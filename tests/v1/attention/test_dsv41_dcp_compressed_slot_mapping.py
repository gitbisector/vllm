# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 DCP: compressed-state sharding and decode-length localization."""

import pytest
import torch

from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

PAD = -1


def _owner_and_local(pos: int, block_size: int, world: int, interleave: int):
    """The runner's DCP position sharding (worker/gpu/block_table.py)."""
    virtual = block_size * world
    off = pos % virtual
    owner = (off // interleave) % world
    local_off = (off // (interleave * world)) * interleave + off % interleave
    return owner, (pos // virtual) * block_size + local_off


def _token_slot_mapping(positions, token_block_size, world, rank, interleave):
    """The sharded group's token slot mapping: PAD for tokens other ranks own."""
    return torch.tensor(
        [
            local if owner == rank else PAD
            for owner, local in (
                _owner_and_local(p, token_block_size, world, interleave)
                for p in positions
            )
        ],
        dtype=torch.int64,
    )


def _run(seq_lens, query_lens, ratio, states_per_page, world, rank, interleave):
    device = "cuda"
    query_start_loc = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(query_lens).cumsum(0)
    positions = [p for sl, ql in zip(seq_lens, query_lens) for p in range(sl - ql, sl)]
    max_states = max(seq_lens) // ratio + 1
    num_entries = -(-max_states // (states_per_page * world))
    # Distinct physical pages per (request, block-table entry).
    block_table = (
        torch.arange(len(seq_lens) * num_entries, dtype=torch.int32).view(
            len(seq_lens), num_entries
        )
        + 5
    )
    slot_mapping = _token_slot_mapping(
        positions, states_per_page * ratio, world, rank, interleave
    )
    out = get_compressed_slot_mapping(
        len(positions),
        slot_mapping.to(device),
        query_start_loc.to(device),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        block_table.to(device),
        states_per_page,
        ratio,
        dcp_world_size=world,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
    ).cpu()
    return positions, block_table, out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_len10_ratio2_world2_rank0_owns_three_states():
    """len=10, R=2, W=2: states 0..4; rank 0 owns {0, 2, 4}, rank 1 {1, 3}.

    The state closing token of state s is 2s+1, always odd, so rank 1 owns
    every one of them in TOKEN space: the token slot mapping is PAD on rank 0
    for all of them and must not gate the state write.
    """
    written = {}
    for rank in range(2):
        _, block_table, out = _run([10], [10], 2, 4, 2, rank, 1)
        page = int(block_table[0, 0])
        slots = [s for s in out.tolist() if s != PAD]
        written[rank] = slots
        # Writers are the state-closing tokens 2s+1 of this rank's states.
        expected_tokens = [2 * s + 1 for s in range(5) if s % 2 == rank]
        assert [i for i, s in enumerate(out.tolist()) if s != PAD] == expected_tokens
        assert slots == [page * 4 + i for i in range(len(expected_tokens))]
    assert len(written[0]) == 3 and len(written[1]) == 2
    # Each rank's share matches the decode bound localize(len // R) -- not
    # localize(len) // R, which gives rank 0 only 5 // 2 == 2 states.
    for rank in range(2):
        local = get_dcp_local_seq_lens(torch.tensor([10 // 2]), 2, rank, 1)
        assert int(local[0]) == len(written[rank])
    assert int(get_dcp_local_seq_lens(torch.tensor([10]), 2, 0, 1)[0]) // 2 == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("world,interleave", [(2, 1), (2, 2), (4, 1)])
def test_state_sharding_matches_reference(ratio, world, interleave):
    states_per_page = 8
    # Full prefills, a chunked prefill and decodes with spec tokens.
    seq_lens = [10, 37, 129, 300, 301, 6]
    query_lens = [10, 37, 50, 6, 1, 6]
    all_states: list[tuple[int, int]] = []
    for rank in range(world):
        positions, block_table, out = _run(
            seq_lens, query_lens, ratio, states_per_page, world, rank, interleave
        )
        token_req = [r for r, ql in enumerate(query_lens) for _ in range(ql)]
        expected = []
        for pos, req in zip(positions, token_req):
            if (pos + 1) % ratio:
                expected.append(PAD)
                continue
            state = pos // ratio
            owner, local = _owner_and_local(state, states_per_page, world, interleave)
            if owner != rank:
                expected.append(PAD)
                continue
            all_states.append((req, state))
            page = int(block_table[req, local // states_per_page])
            expected.append(page * states_per_page + local % states_per_page)
        assert out.tolist() == expected
        slots = [s for s in expected if s != PAD]
        assert len(slots) == len(set(slots))
        # Full-prefill rows: this rank's share equals the localized bound.
        for req, (sl, ql) in enumerate(zip(seq_lens, query_lens)):
            if ql != sl:
                continue
            n = sum(
                1
                for (p, r), s in zip(zip(positions, token_req), out.tolist())
                if r == req and s != PAD
            )
            local = get_dcp_local_seq_lens(
                torch.tensor([sl // ratio]), world, rank, interleave
            )
            assert n == int(local[0])
    # Ranks write disjoint states covering every closed state.
    assert len(all_states) == len(set(all_states))
    closed = {
        (req, pos // ratio)
        for req, (sl, ql) in enumerate(zip(seq_lens, query_lens))
        for pos in range(sl - ql, sl)
        if (pos + 1) % ratio == 0
    }
    assert set(all_states) == closed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dcp1_keeps_replay_pad():
    """Without DCP a PAD token slot (bounded replay) still suppresses the write."""
    device = "cuda"
    slot_mapping = torch.arange(8, dtype=torch.int64, device=device)
    slot_mapping[:4] = PAD
    out = get_compressed_slot_mapping(
        8,
        slot_mapping,
        torch.tensor([0, 8], dtype=torch.int32, device=device),
        torch.tensor([8], dtype=torch.int32, device=device),
        torch.tensor([[3]], dtype=torch.int32, device=device),
        block_size=4,
        compress_ratio=2,
    )
    assert out.tolist() == [-1, -1, -1, -1, -1, 3 * 4 + 2, -1, 3 * 4 + 3]
