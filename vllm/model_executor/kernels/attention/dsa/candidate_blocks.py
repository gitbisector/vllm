# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

# Scratch cap for the DCP block-score all-reduce (bytes per row slice).
_DCP_REDUCE_BYTES = 128 * 1024 * 1024


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    PIN_NEWEST: tl.constexpr = True,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        (blocks[:, None] < nblocks)
        & (offsets[None, :] < BLOCK_SIZE)
        & (cols < end)
        & (cols < width),
        other=-float("inf"),
    )
    reduced = tl.reduce(values, 1, _max_with_nan)
    if PIN_NEWEST:
        reduced = tl.where(
            (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE),
            float("inf"),
            reduced,
        )
    tl.store(scores + row * nblocks + blocks, reduced, blocks < nblocks)


@triton.jit(do_not_specialize=["k"])
def _store_candidates_kernel(
    values,
    indices,
    output,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
    index = tl.load(indices + row * k + cols, cols < k, other=-1)
    # NaN scores can occur during warmup; only -inf denotes padding.
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(value != -float("inf"), index, -1),
        cols < OUT_K,
    )


@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(
    candidates,
    starts,
    flags,
    stride_row,
    stride_col,
    stride_start,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(nblocks + 1, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots <= nblocks)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    ).to(tl.int64)
    # Preserve the packed-column clamp for candidates beyond the logits width.
    block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    tl.debug_barrier()
    tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))


@triton.jit(do_not_specialize=["width", "nblocks"])
def _mask_candidates_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    tl.store(
        logits + row * stride_row + cols * stride_col,
        -float("inf"),
        (cols < width) & ~(valid & keep),
    )


def _block_scores(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    block_size: int,
    row_repeat: int,
    pin_newest: bool,
) -> torch.Tensor:
    rows, width = logits.shape
    nblocks = triton.cdiv(width, block_size)
    scores = logits.new_empty((rows, nblocks))
    _block_scores_kernel[(rows, triton.cdiv(nblocks, 128))](
        logits,
        row_ks,
        row_ke,
        scores,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        128,
        PIN_NEWEST=pin_newest,
    )
    return scores


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
    *,
    dcp_group: "GroupCoordinator | None" = None,
    global_row_ke: torch.Tensor | None = None,
    global_block_size: int | None = None,
    max_global_blocks: int | None = None,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.

    Under DCP (``dcp_group`` given) ``logits`` hold this rank's local packed
    columns and ``block_size`` is the LOCAL block, ``global_block_size //
    world``: with interleave 1, local block ``b`` is exactly this rank's states
    of global block ``b``, so the global block score is the MAX over ranks of
    the local block scores (one all-reduce over ``[rows, max_global_blocks]``)
    and the block ids need no translation. The newest block is pinned from the
    GLOBAL per-row context ``global_row_ke`` (states, per row or per
    ``row_repeat`` group) after the reduce, so every rank pins the same block.
    Collective: every DCP rank must call this with the same ``rows`` and
    ``max_global_blocks`` (an empty local shard is fine).
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows:
        return
    if dcp_group is not None:
        _select_candidate_blocks_dcp(
            logits,
            row_ks,
            row_ke,
            topk_blocks,
            block_size,
            out,
            row_repeat,
            dcp_group,
            global_row_ke,
            global_block_size,
            max_global_blocks,
        )
        return
    if not width:
        out.fill_(-1)
        return
    scores = _block_scores(logits, row_ks, row_ke, block_size, row_repeat, True)
    nblocks = scores.shape[1]
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
        top.values,
        top.indices,
        out,
        *out.stride(),
        top.values.shape[1],
        topk_blocks,
        256,
    )


def _select_candidate_blocks_dcp(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int,
    dcp_group: "GroupCoordinator",
    global_row_ke: torch.Tensor | None,
    global_block_size: int | None,
    max_global_blocks: int | None,
) -> None:
    assert global_row_ke is not None
    assert global_block_size is not None and max_global_blocks is not None
    assert global_block_size == block_size * dcp_group.world_size, (
        "DCP candidate blocks need block_size == global_block_size // world"
    )
    rows, width = logits.shape
    nblocks = max(int(max_global_blocks), 1)
    ends = global_row_ke.reshape(-1)
    if row_repeat > 1:
        ends = ends.repeat_interleave(row_repeat)
    ends = ends[:rows].to(torch.int64)
    # Pin each row's newest GLOBAL block, as DCP1 pins (end - start - 1) //
    # block from the global bounds; rows without context select nothing.
    pin = ((ends - 1) // global_block_size).clamp_(0, nblocks - 1).view(-1, 1)
    local = (
        _block_scores(logits, row_ks, row_ke, block_size, row_repeat, False)
        if width
        else None
    )
    # Reduce in row slices: a 16K-token prefill chunk at 128K context would
    # otherwise need a ~1 GiB scratch per rank. The slice count derives from
    # rows and nblocks, identical on every rank, so the collectives line up.
    k = min(topk_blocks, nblocks)
    rows_per_slice = max(1, _DCP_REDUCE_BYTES // (nblocks * 4))
    top_values = logits.new_empty((rows, k))
    top_indices = torch.empty((rows, k), dtype=torch.int64, device=logits.device)
    for r0 in range(0, rows, rows_per_slice):
        r1 = min(r0 + rows_per_slice, rows)
        scores = logits.new_full((r1 - r0, nblocks), -float("inf"))
        if local is not None:
            n = min(local.shape[1], nblocks)
            scores[:, :n] = local[r0:r1, :n]
        torch.distributed.all_reduce(
            scores, op=torch.distributed.ReduceOp.MAX, group=dcp_group.device_group
        )
        scores.scatter_(1, pin[r0:r1], float("inf"))
        scores.masked_fill_((ends[r0:r1] <= 0).view(-1, 1), -float("inf"))
        # Keep the existing top-k tie behavior.
        top = scores.topk(k, dim=-1)
        top_values[r0:r1] = top.values
        top_indices[r0:r1] = top.indices
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
        top_values,
        top_indices,
        out,
        *out.stride(),
        k,
        topk_blocks,
        256,
    )


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> None:
    """Mask packed logits outside causal bounds and request-local candidates.

    Under DCP pass the LOCAL block size (global // world): local block ``b``
    is this rank's shard of global block ``b``, so the candidate ids apply
    unchanged.
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, nblocks + 1), device=logits.device, dtype=torch.uint8)
    start_stride = row_ks.stride(0) if row_ks is not None else 0
    _candidate_flags_kernel[(rows,)](
        candidate_blocks,
        row_ks,
        flags,
        *candidate_blocks.stride(),
        start_stride,
        width,
        nblocks,
        block_size,
        candidate_blocks.shape[1],
        row_ks is not None,
        row_repeat,
    )
    _mask_candidates_kernel[(rows, triton.cdiv(width, 1024))](
        logits,
        row_ks,
        row_ke,
        flags,
        *logits.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        1024,
    )
