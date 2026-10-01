# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 SM120 sparse MLA under DCP, emulated on one GPU.

The per-rank partial attention (FlashInfer's SM120 kernel with the LSE) plus
the LSE combine must reproduce the unsplit attention, and the prefill KV
gather must remap global compressed ids onto the gathered pages."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

_SM12X = current_platform.is_cuda() and current_platform.is_device_capability_family(
    120
)


def _packed_cache(pages: int, page: int, device: str) -> torch.Tensor:
    from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache

    n = pages * page
    kv = (torch.randn(n, 512, device=device) * 0.5).to(torch.bfloat16)
    cache = torch.zeros(pages, page * 584, dtype=torch.uint8, device=device)
    quantize_and_insert_k_cache(kv, cache, torch.arange(n, device=device), page)
    return cache.view(pages, page, 1, 584)


@pytest.mark.skipif(not _SM12X, reason="requires SM12x")
@pytest.mark.parametrize("combine", ["ag_rs", "a2a"])
def test_split_attention_and_lse_combine_match_unsplit(combine):
    from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
        _get_flashinfer_dsv4_workspace,
        _sm120_sparse_attention_with_lse,
    )
    from vllm.v1.attention.ops.dcp import (
        CPTritonContext,
        _dcp_a2a_lse_pack_dim,
        _dcp_a2a_pack_send,
        _dcp_a2a_unpack_combine,
        correct_attn_out,
    )

    torch.manual_seed(0)
    device = "cuda"
    world, heads, tokens, page = 2, 32, 6, 64
    swa = _packed_cache(4, page, device)
    comp = _packed_cache(16, page, device)
    q = (torch.randn(tokens, heads, 512, device=device) * 0.05).to(torch.bfloat16)
    sinks = torch.randn(heads, device=device)
    workspace = _get_flashinfer_dsv4_workspace(torch.device(device))
    # Row 0 has no compressed candidates at all; row 1 none on rank 1.
    swa_n = torch.tensor([5, 64, 128, 1, 40, 100], device=device, dtype=torch.int32)
    ext_n = torch.tensor([0, 1, 300, 512, 64, 33], device=device, dtype=torch.int32)
    swa_idx = torch.full((tokens, 128), -1, dtype=torch.int32, device=device)
    ext_idx = torch.full((tokens, 512), -1, dtype=torch.int32, device=device)
    for t in range(tokens):
        swa_idx[t, : swa_n[t]] = torch.randperm(4 * page, device=device)[: swa_n[t]]
        ext_idx[t, : ext_n[t]] = torch.randperm(16 * page, device=device)[: ext_n[t]]

    def attend(swa_i, swa_l, ext_i, ext_l, sink):
        out = torch.empty(tokens, heads, 512, dtype=torch.bfloat16, device=device)
        lse = torch.empty(tokens, heads, dtype=torch.float32, device=device)
        _sm120_sparse_attention_with_lse(
            q, swa, workspace, swa_i, swa_l, comp, ext_i, ext_l, out,
            1 / math.sqrt(512), sink, lse,
        )  # fmt: skip
        return out, lse

    ref, _ = attend(swa_idx, swa_n, ext_idx, ext_n, sinks)
    partial = []
    for rank in range(world):
        idx = torch.full_like(ext_idx, -1)
        lens = torch.zeros_like(ext_n)
        for t in range(tokens):
            mine = ext_idx[t, : ext_n[t]][rank::world]
            idx[t, : mine.numel()] = mine
            lens[t] = mine.numel()
        # The window and the sink live on rank 0 only.
        if rank == 0:
            partial.append(attend(swa_idx, swa_n, idx, lens, sinks))
        else:
            empty = torch.zeros_like(swa_n)
            partial.append(attend(torch.full_like(swa_idx, -1), empty, idx, lens, None))
    assert torch.all(partial[1][1][0] == -1e30)

    hpr = heads // world
    if combine == "ag_rs":
        lses = torch.stack([lse for _, lse in partial])
        total = torch.zeros(tokens, heads, 512, device=device)
        for rank, (out, _) in enumerate(partial):
            corrected, _ = correct_attn_out(
                out.clone(), lses, rank, CPTritonContext(), is_lse_base_on_e=False
            )
            total += corrected.float()
        results = [total[:, r * hpr : (r + 1) * hpr] for r in range(world)]
    else:
        pack = _dcp_a2a_lse_pack_dim(torch.bfloat16)
        sends = []
        for out, lse in partial:
            send = torch.empty(
                (world, tokens, hpr, 512 + pack), dtype=torch.bfloat16, device=device
            )
            _dcp_a2a_pack_send(out, lse, send, world, hpr, 512, pack)
            sends.append(send)
        results = [
            _dcp_a2a_unpack_combine(
                torch.stack([send[r] for send in sends]), 512, pack, False, False
            ).float()
            for r in range(world)
        ]
    for rank, got in enumerate(results):
        want = ref[:, rank * hpr : (rank + 1) * hpr].float()
        # bf16 partial outputs: within a few bf16 ulps of |out| < 1.
        torch.testing.assert_close(got, want, atol=1e-2, rtol=0)


def test_prefill_gather_remaps_global_ids(monkeypatch):
    from vllm.models.deepseek_v41.nvidia import flashinfer_sparse as fs

    world, states_per_page, ratio = 2, 4, 2
    seq_lens = [3, 37, 64]  # tokens: 1, 18 and 32 states
    num_decodes = 1
    num_prefills = len(seq_lens) - num_decodes
    max_entries = 8
    # Distinct physical pages per (request, virtual block) on each rank.
    block_table = torch.arange(len(seq_lens) * max_entries).view(len(seq_lens), -1)
    pool_pages = block_table.numel()
    # Each rank's pool holds (rank, local state) tags: [pages, page, 1, 2].
    pools = [
        torch.full((pool_pages, states_per_page, 1, 2), -1, dtype=torch.int64)
        for _ in range(world)
    ]
    for req, sl in enumerate(seq_lens):
        for state in range(sl // ratio):
            virtual = states_per_page * world
            off = state % virtual
            owner = off % world
            local = (state // virtual) * states_per_page + off // world
            page = block_table[req, local // states_per_page]
            pools[owner][page, local % states_per_page, 0] = torch.tensor([req, state])

    calls = iter(range(10**6))
    gathered: dict[int, torch.Tensor] = {}

    def all_gather(local, dim=0):
        # Emulate the other rank's contribution with its own pool.
        i = next(calls)
        page_ids = gathered[i]
        return torch.cat([pool.index_select(0, page_ids) for pool in pools], dim)

    layer = object.__new__(fs.DeepseekV4FlashInferSM120Attention)
    layer.__dict__.update(
        dcp_world_size=world,
        cp_kv_cache_interleave_size=1,
        compress_ratio=ratio,
        index_source_layer_id=0,
        dcp_group=SimpleNamespace(all_gather=all_gather),
    )
    for i in range(num_prefills):
        p = -(-(seq_lens[num_decodes + i] // ratio) // (states_per_page * world))
        gathered[i] = block_table[num_decodes + i, :p]
    step = object()
    monkeypatch.setattr(
        fs, "get_forward_context", lambda: SimpleNamespace(attn_metadata=step)
    )
    monkeypatch.setattr(fs.DeepseekV4FlashInferSM120Attention, "_dcp_gather_cache", {})
    monkeypatch.setattr(
        fs.DeepseekV4FlashInferSM120Attention,
        "_as_sparse_cache",
        staticmethod(lambda kv: kv),
    )

    # Prefill tokens: request 1 then request 2, each asking for some global ids.
    topk = torch.tensor(
        [[0, 17, 5, -1], [8, 9, -1, -1], [31, 0, 16, 7], [-1, -1, -1, -1]],
        dtype=torch.int32,
    )
    token_to_req = torch.tensor([1, 1, 2, 2])
    valid = torch.tensor([True, True, True, False])
    metadata = SimpleNamespace(
        seq_lens_cpu=torch.tensor(seq_lens), block_table=block_table
    )
    kv, slots, lens = layer._dcp_prefill_gather(
        pools[0], topk, token_to_req, valid, metadata, num_decodes, num_prefills, 4
    )
    assert lens.tolist() == [3, 2, 4, 0]
    flat = kv.reshape(-1, 2)
    for row, req in enumerate(token_to_req.tolist()):
        for col, state in enumerate(topk[row].tolist()):
            if state < 0:
                assert slots[row, col] == -1
                continue
            assert flat[slots[row, col]].tolist() == [req, state]
