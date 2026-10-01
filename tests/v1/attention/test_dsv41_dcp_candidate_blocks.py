# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 DCP candidate blocks: the per-rank MAX-reduced selection over
state-sharded logits must pick the same global blocks as DCP1."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.attention.dsa import candidate_blocks as cb
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens


def _shard(logits: torch.Tensor, ends: torch.Tensor, world: int, rank: int):
    """Rank ``rank``'s local columns (interleave 1: state s -> s // world)."""
    local = logits[:, rank::world].contiguous()
    local_ends = get_dcp_local_seq_lens(ends, world, rank, 1)
    return local, local_ends


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("row_repeat", [1, 3])
def test_dcp_candidates_match_dcp1(monkeypatch, world, row_repeat):
    torch.manual_seed(0)
    device = "cuda"
    block, topk_blocks = 8, 6
    # Per-group global context in states; includes empty and partial blocks.
    group_ends = torch.tensor([0, 1, 7, 8, 9, 33, 64, 70], device=device)
    ends = group_ends.repeat_interleave(row_repeat)
    rows = ends.numel()
    width = int(group_ends.max())
    logits = torch.randn(rows, width, device=device)
    cols = torch.arange(width, device=device)
    logits.masked_fill_(cols[None, :] >= ends[:, None], -float("inf"))

    expected = torch.full((rows, topk_blocks), -7, dtype=torch.int32, device=device)
    cb.select_candidate_blocks(
        logits, None, group_ends, topk_blocks, block, expected, row_repeat
    )

    nblocks = -(-width // block)
    shards = [_shard(logits, ends, world, r) for r in range(world)]
    reduced = torch.full((rows, nblocks), -float("inf"), device=device)
    for local, local_ends in shards:
        scores = cb._block_scores(
            local, None, local_ends[::row_repeat], block // world, row_repeat, False
        )
        n = min(scores.shape[1], nblocks)
        reduced[:, :n] = torch.maximum(reduced[:, :n], scores[:, :n])

    def fake_all_reduce(tensor, op=None, group=None):
        assert tensor.shape == reduced.shape
        tensor.copy_(reduced)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    group = SimpleNamespace(world_size=world, device_group=None)
    for local, local_ends in shards:
        out = torch.full_like(expected, -7)
        cb.select_candidate_blocks(
            local,
            None,
            local_ends[::row_repeat],
            topk_blocks,
            block // world,
            out,
            row_repeat,
            dcp_group=group,
            global_row_ke=group_ends,
            global_block_size=block,
            max_global_blocks=nblocks,
        )
        for got, want in zip(out.tolist(), expected.tolist()):
            assert [b for b in got if b >= 0] == [b for b in want if b >= 0]
            assert all(b == -1 for b in got if b < 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dcp_candidates_empty_local_shard_joins_reduce(monkeypatch):
    """A rank with no local states still contributes and gets the pin."""
    device = "cuda"
    calls = []

    def fake_all_reduce(tensor, op=None, group=None):
        calls.append(tensor.shape)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    out = torch.full((2, 4), -7, dtype=torch.int32, device=device)
    cb.select_candidate_blocks(
        torch.empty((2, 0), device=device),
        None,
        torch.zeros(2, dtype=torch.int32, device=device),
        4,
        4,
        out,
        dcp_group=SimpleNamespace(world_size=2, device_group=None),
        global_row_ke=torch.tensor([1, 0], device=device),
        global_block_size=8,
        max_global_blocks=1,
    )
    assert calls == [(2, 1)]
    assert out.tolist() == [[0, -1, -1, -1], [-1, -1, -1, -1]]


def test_decode_global_row_ends():
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _dcp_decode_global_row_ends as row_ends,
    )

    lens = torch.tensor([10, 7])
    # Flattened / plain decode: one end per row.
    assert row_ends(torch.tensor([10, 7, 9]), 3, 1, 3, 1, 2).tolist() == [5, 3, 4]
    # Shared bounds per row_repeat group.
    assert row_ends(lens, 2, 3, 6, 3, 2).tolist() == [5, 3]
    # Native spec decode: request lengths -> per-row causal lengths.
    assert row_ends(lens, 2, 3, 6, 1, 1).tolist() == [8, 9, 10, 5, 6, 7]
    # Graph-padding rows have no context.
    assert row_ends(lens, 4, 1, 4, 1, 2).tolist() == [5, 3, 0, 0]
