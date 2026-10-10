# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for ``scatter_states``.

Rows whose state index is ``<= 0`` (``NULL_BLOCK_ID``) must be skipped: in dummy
runs every block table row is zero, and writing there would clobber the null
block shared with other cache groups.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="scatter_states requires CUDA"
)

SENTINEL = 7.0
STATE_SHAPE = (4, 33)  # row_size not a power of two to exercise the mask
NUM_SLOTS = 8


def _make_inputs(indices: list[int], index_dtype: torch.dtype):
    state = torch.full(
        (NUM_SLOTS, *STATE_SHAPE), SENTINEL, dtype=torch.float32, device="cuda"
    )
    src = torch.arange(1, len(indices) + 1, dtype=torch.float32, device="cuda")
    src = src.view(-1, 1, 1).expand(-1, *STATE_SHAPE).contiguous()
    idx = torch.tensor(indices, dtype=index_dtype, device="cuda")
    return state, src, idx


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_scatter_states_skips_null(index_dtype):
    indices = [3, NULL_BLOCK_ID, 5, 1]
    state, src, idx = _make_inputs(indices, index_dtype)
    expected = state.clone()
    for row, slot in enumerate(indices):
        if slot > 0:
            expected[slot] = src[row]

    scatter_states(state, src, idx)

    torch.testing.assert_close(state, expected)
    assert torch.all(state[NULL_BLOCK_ID] == SENTINEL)


@pytest.mark.parametrize("num_rows", [1, 4])
def test_scatter_states_all_null_is_noop(num_rows):
    # The dummy run shape: every block table row is zero.
    state, src, idx = _make_inputs([NULL_BLOCK_ID] * num_rows, torch.int32)
    expected = state.clone()

    scatter_states(state, src, idx)

    torch.testing.assert_close(state, expected)
