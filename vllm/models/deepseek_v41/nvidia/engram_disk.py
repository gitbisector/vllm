# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram tables served from the checkpoint on disk (``DSV41_ENGRAM_DISK=1``).

The two DeepSeek-V4.1-Flash Engram tables (~188 GiB) do not fit next to the
model on 128 GB unified-memory nodes (DGX Spark / GB10), where "pinned host
memory" is the same pool the GPU uses. In disk mode no rank allocates its
table shard; rows are read on demand from the safetensors shards (preadv, or
one NumPy gather through a read-only memory map), dequantized on the CPU
(fp8 e4m3 x ue8m0 block scales -> bf16) and copied into the layer's staging
buffer.

Under the V2 runner :class:`EngramDiskStager` stages every engram layer's
rows from ``prepare_inputs``, before the forward, so the forward holds no
host round trip and FULL CUDA graphs work. The V1 runner looks rows up in
the forward instead (eager only).

Optional ``DSV41_ENGRAM_DIR`` points at a node-local sparse copy of the
Engram shards holding only this rank's rows at their original offsets, plus
``engram-local.json`` with the copied row range per layer; it is used only
where that range covers the rank's rows.

Ported from tonyd2wild's DeepSeek-V4.1-Flash-vLLM-DGX-Spark patches (MIT):
engram-offset-fix, engram-parallel-reads, cudagraph-prestage.
"""

import contextlib
import json
import mmap
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.common.engram import NgramHashState

logger = init_logger(__name__)

# Fork-local switches; the names and defaults are the production interface.
ENGRAM_DISK = os.environ.get("DSV41_ENGRAM_DISK", "0") == "1"
# One process-wide read pool shared by every table: a step's reads for all
# engram layers (weight and scale rows) go out as one batch.
_READ_THREADS = int(os.environ.get("DSV41_ENGRAM_DISK_THREADS", "32"))
_READ_CHUNK = int(os.environ.get("DSV41_ENGRAM_DISK_CHUNK", "16"))
# Gather rows through a read-only memory map (one NumPy fancy-index copy per
# task) instead of one preadv per row.
_MMAP = os.environ.get("DSV41_ENGRAM_MMAP", "0") == "1"
# The gather releases the GIL, so more threads = more page faults in flight.
_MMAP_THREADS = int(os.environ.get("DSV41_ENGRAM_MMAP_THREADS", "96"))
# Gathers smaller than this many unique rows per table keep the preadv path
# (0 = memory map for every size).
_MMAP_MIN_ROWS = int(os.environ.get("DSV41_ENGRAM_MMAP_MIN_ROWS", "0"))
# Gathers up to this many rows skip the pool: an async readahead of every row
# (POSIX_FADV_WILLNEED, all reads in flight together), then one gather per
# table on the calling thread. Per-task faults serialize decode-sized gathers.
_MMAP_INLINE_ROWS = int(os.environ.get("DSV41_ENGRAM_MMAP_INLINE_ROWS", "2048"))
# Hash-ahead prefetch of the next prefill chunk's rows into the page cache
# while the current chunk computes (the prompt is known up front).
_PREFETCH = os.environ.get("DSV41_ENGRAM_PREFETCH", "0") == "1"
_PREFETCH_MODE = os.environ.get("DSV41_ENGRAM_PREFETCH_MODE", "fadvise")
_PREFETCH_THREADS = int(os.environ.get("DSV41_ENGRAM_PREFETCH_THREADS", "32"))
_PREFETCH_TASK_ROWS = 512

_POOL: ThreadPoolExecutor | None = None

# A read job: (fd, byte offset of row 0, row ids, row bytes, destination
# buffer[, memory-mapped [num_rows, row_bytes] view]).
ReadJob = tuple[Any, ...]


def skip_engram_checkpoint_tensor(name: str) -> bool:
    """The default loader's ``skip_checkpoint_tensor``: in disk mode the
    Engram tables stay on disk and are never read at load time."""
    return ENGRAM_DISK and name.endswith(
        (".engram.embed.weight", ".engram.embed.scale")
    )


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(
            max_workers=_MMAP_THREADS if _MMAP else _READ_THREADS,
            thread_name_prefix="engram-disk",
        )
    return _POOL


def _prefetch_rows(table: "DiskEngramTable", rows: list[int]) -> None:
    """Warm rank-local `rows` of `table` in the page cache: weight rows by
    exact byte range, scale rows by 4 KiB page (many rows share one)."""
    if not rows:
        return
    dim, sb = table.dim, table.sb
    scale_pages = {(table.s_off + r * sb) >> 12 for r in rows}
    if _PREFETCH_MODE == "pread":
        buf = bytearray(dim)
        for r in rows:
            with contextlib.suppress(OSError):
                os.preadv(table.w_fd, [buf], table.w_off + r * dim)
        sbuf = bytearray(4096)
        for p in scale_pages:
            with contextlib.suppress(OSError):
                os.preadv(table.s_fd, [sbuf], p << 12)
        return
    adv = os.POSIX_FADV_WILLNEED
    for r in rows:
        with contextlib.suppress(OSError):
            os.posix_fadvise(table.w_fd, table.w_off + r * dim, dim, adv)
    for p in scale_pages:
        with contextlib.suppress(OSError):
            os.posix_fadvise(table.s_fd, p << 12, 4096, adv)


def _map_file(path: str) -> tuple[mmap.mmap, np.ndarray]:
    """Read-only shared mapping of a whole shard with readahead disabled.

    MADV_RANDOM is the mmap twin of the preadv path's POSIX_FADV_RANDOM: a
    fault brings exactly the page of the row. Without it every cold-row fault
    pulled an NFS readahead window (rsize 1 MiB): 9K prefill 14.7 s vs 8.1 s.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)
    with contextlib.suppress(AttributeError, OSError):  # old kernels
        mm.madvise(mmap.MADV_RANDOM)
    return mm, np.frombuffer(mm, dtype=np.uint8)


def _mmap_rows(
    view: np.ndarray, rel: list[int], lo: int, hi: int, row_bytes: int, buf
) -> None:
    """Copy rows rel[lo:hi] of the [num_rows, row_bytes] mapped view into buf
    (a flat uint8 memoryview): memcpy for resident pages, faults for cold
    ones; NumPy releases the GIL."""
    dst = np.frombuffer(buf, dtype=np.uint8)[lo * row_bytes : hi * row_bytes]
    np.take(
        view,
        np.asarray(rel[lo:hi], dtype=np.int64),
        axis=0,
        out=dst.reshape(hi - lo, row_bytes),
    )


def _pread_rows(
    fd: int, base: int, rel: list[int], lo: int, hi: int, row_bytes: int, buf
) -> None:
    for i in range(lo, hi):
        off = base + rel[i] * row_bytes
        view = buf[i * row_bytes : (i + 1) * row_bytes]
        got = 0
        while got < row_bytes:
            n = os.preadv(fd, [view[got:]], off + got)
            if n <= 0:
                raise OSError("engram disk table: short read")
            got += n


def _parallel_read(jobs: list[ReadJob]) -> None:
    """Read every row of every job with all rows in flight at once.

    A preadv task carries ceil(total / threads) rows, capped at
    DSV41_ENGRAM_DISK_CHUNK. Only the calling thread submits, so the shared
    pool cannot deadlock.
    """
    total = sum(len(job[2]) for job in jobs)
    if total == 0:
        return

    def run(job: ReadJob, lo: int, hi: int) -> None:
        fd, base, rel, row_bytes, buf = job[:5]
        view = job[5] if len(job) > 5 else None
        if view is not None:
            _mmap_rows(view, rel, lo, hi, row_bytes, buf)
        else:
            _pread_rows(fd, base, rel, lo, hi, row_bytes, buf)

    if total == 1:
        for job in jobs:
            run(job, 0, len(job[2]))
        return
    mapped = [len(job) > 5 and job[5] is not None for job in jobs]
    if any(mapped):
        if total <= _MMAP_INLINE_ROWS and all(mapped):
            adv = os.POSIX_FADV_WILLNEED
            for fd, base, rel, row_bytes, _, _ in jobs:
                for r in rel:
                    with contextlib.suppress(OSError):
                        os.posix_fadvise(fd, base + r * row_bytes, row_bytes, adv)
            for job in jobs:
                run(job, 0, len(job[2]))
            return
        # Cold rows fault in sequentially within a task, so spread the rows
        # over every pool thread.
        chunk = max(1, -(-total // _MMAP_THREADS))
    else:
        chunk = max(1, min(_READ_CHUNK, -(-total // _READ_THREADS)))
    pool = _pool()
    futures = [
        pool.submit(run, job, lo, min(lo + chunk, len(job[2])))
        for job in jobs
        for lo in range(0, len(job[2]), chunk)
    ]
    for future in futures:
        future.result()


def _local_dir(
    model_dir: str, layer_id: int, row_start: int, num_rows: int | None
) -> str:
    """DSV41_ENGRAM_DIR when its recorded row range for this layer covers
    this table's rows, else model_dir: a changed rank map must never read the
    sparse copy's empty holes."""
    local = os.environ.get("DSV41_ENGRAM_DIR", "")
    if not local:
        return model_dir
    try:
        with open(os.path.join(local, "engram-local.json")) as f:
            lo, hi = json.load(f)["layers"][str(layer_id)]
    except Exception as e:
        logger.warning(
            "DSV41_ENGRAM_DIR=%s unusable for layer %d (%s); reading %s",
            local,
            layer_id,
            e,
            model_dir,
        )
        return model_dir
    if num_rows is None or not (lo <= row_start and row_start + num_rows <= hi):
        logger.warning(
            "DSV41_ENGRAM_DIR=%s holds layer %d rows [%d, %d), this rank needs "
            "[%d, +%s); reading %s",
            local,
            layer_id,
            lo,
            hi,
            row_start,
            num_rows,
            model_dir,
        )
        return model_dir
    logger.info(
        "Engram layer %d rows [%d, %d) read from node-local %s",
        layer_id,
        row_start,
        row_start + num_rows,
        local,
    )
    return local


class DiskEngramTable:
    """This rank's rows of one Engram table, read from the safetensors shards.

    The checkpoint holds the full table (every rank's hash heads) per layer
    while callers pass rank-local row ids, so reads start at `row_start`.
    """

    def __init__(
        self,
        model_dir: str,
        layer_id: int,
        dim: int,
        block_size: int,
        row_start: int = 0,
        num_rows: int | None = None,
    ) -> None:
        model_dir = _local_dir(model_dir, layer_id, row_start, num_rows)
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        wname = f"layers.{layer_id}.engram.embed.weight"
        sname = f"layers.{layer_id}.engram.embed.scale"
        w_path = os.path.join(model_dir, weight_map[wname])
        s_path = os.path.join(model_dir, weight_map[sname])
        self.w_fd, self.w_off, w_shape = self._open(w_path, wname)
        self.s_fd, self.s_off, s_shape = self._open(s_path, sname)
        self.dim = dim
        self.sb = dim // block_size
        assert w_shape[1] == dim, (w_shape, dim)
        assert s_shape[1] == self.sb, (s_shape, self.sb)
        assert s_shape[0] == w_shape[0], (s_shape, w_shape)
        if num_rows is None:
            num_rows = w_shape[0] - row_start
        assert row_start >= 0 and row_start + num_rows <= w_shape[0], (
            row_start,
            num_rows,
            w_shape,
        )
        self.w_off += row_start * dim
        self.s_off += row_start * self.sb
        self.row_start = row_start
        self.num_rows = num_rows
        # [num_rows, row_bytes] views of this rank's rows through read-only
        # maps of each shard; pages come and go with the page cache.
        self.w_view: np.ndarray | None = None
        self.s_view: np.ndarray | None = None
        if _MMAP:
            self._w_mm, w_buf = _map_file(w_path)
            self._s_mm, s_buf = _map_file(s_path)
            self.w_view = np.ndarray(
                (num_rows, dim), dtype=np.uint8, buffer=w_buf, offset=self.w_off
            )
            self.s_view = np.ndarray(
                (num_rows, self.sb), dtype=np.uint8, buffer=s_buf, offset=self.s_off
            )
        logger.info(
            "Engram DISK mode: layer %d rows [%d, %d) read from %s (off=%d) and "
            "%s (off=%d); %s, %d threads",
            layer_id,
            row_start,
            row_start + num_rows,
            weight_map[wname],
            self.w_off,
            weight_map[sname],
            self.s_off,
            "mmap" if _MMAP else "preadv",
            _MMAP_THREADS if _MMAP else _READ_THREADS,
        )

    @staticmethod
    def _open(path: str, tname: str) -> tuple[int, int, tuple[int, ...]]:
        fd = os.open(path, os.O_RDONLY)
        with contextlib.suppress(AttributeError, OSError):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))[tname]
        return fd, 8 + n + meta["data_offsets"][0], tuple(meta["shape"])

    def read_jobs(
        self, rel: list[int], w: torch.Tensor, s: torch.Tensor
    ) -> list[ReadJob]:
        """Read jobs for rank-local rows `rel` into uint8 w [R, dim], s [R, sb]."""
        w_buf = memoryview(w.numpy()).cast("B")
        s_buf = memoryview(s.numpy()).cast("B")
        if self.w_view is not None and len(rel) >= _MMAP_MIN_ROWS:
            return [
                (self.w_fd, self.w_off, rel, self.dim, w_buf, self.w_view),
                (self.s_fd, self.s_off, rel, self.sb, s_buf, self.s_view),
            ]
        return [
            (self.w_fd, self.w_off, rel, self.dim, w_buf),
            (self.s_fd, self.s_off, rel, self.sb, s_buf),
        ]

    def dequant(self, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """fp8 e4m3 rows (as uint8) x ue8m0 block scales -> [R, dim] fp32."""
        num_rows = w.shape[0]
        vals = (
            w.view(torch.float8_e4m3fn)
            .to(torch.float32)
            .view(num_rows, self.sb, self.dim // self.sb)
        )
        # The ue8m0 byte is the fp32 exponent field: 2^(e-127).
        scale = (s.to(torch.int32) << 23).view(torch.float32)
        return (vals * scale[:, :, None]).reshape(num_rows, self.dim)

    def gather_dequant(self, rel: torch.Tensor, owned: torch.Tensor) -> torch.Tensor:
        """rel: [R] int64 CPU local row ids; owned: [R] bool -> [R, dim] bf16."""
        return gather_dequant_many([(self, rel, owned)])[0]


def gather_dequant_many(
    requests: list[tuple[DiskEngramTable, torch.Tensor, torch.Tensor]],
) -> list[torch.Tensor]:
    """[(table, rel [R] int64 CPU, owned [R] bool)] -> one [R, dim] bf16 CPU
    tensor each; rows not owned are zero. Rows are de-duplicated per table and
    the reads of every table go out in one parallel batch."""
    plans, jobs = [], []
    for table, rel, owned in requests:
        uniq, inverse = torch.unique(rel, return_inverse=True)
        w = torch.empty((uniq.numel(), table.dim), dtype=torch.uint8)
        s = torch.empty((uniq.numel(), table.sb), dtype=torch.uint8)
        jobs += table.read_jobs(uniq.tolist(), w, s)
        plans.append((table, w, s, inverse, owned))
    _parallel_read(jobs)
    outs = []
    for table, w, s, inverse, owned in plans:
        out = table.dequant(w, s)[inverse]
        out[~owned] = 0
        outs.append(out.to(torch.bfloat16))
    return outs


def disk_rel_owned(
    local: torch.Tensor,
    local_heads: int,
    vocab_start: int,
    vocab_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """local: [T, <=local_heads] int64 CPU hash ids of this rank's heads ->
    flat (rel [T * local_heads] rank-local row ids, owned [T * local_heads]).
    Padded heads (beyond the table's heads) and ids outside this rank's range
    are not owned, so they read as zeros like the lookup kernel's."""
    num_tokens = local.shape[0]
    if local.shape[1] < local_heads:
        pad = torch.full((num_tokens, local_heads - local.shape[1]), -1)
        local = torch.cat([local, pad.to(local.dtype)], dim=1)
    rows = local.reshape(-1)
    owned = (rows >= vocab_start) & (rows < vocab_end)
    rel = torch.where(owned, rows - vocab_start, torch.zeros_like(rows))
    return rel, owned


class EngramDiskStager:
    """Stage disk-mode Engram rows outside the model forward.

    The V2 model state calls :meth:`stage` from ``prepare_inputs`` once per
    step, after this step's input ids, positions, query start offsets and
    lookback window are written. It hashes the step on the GPU (same kernel
    and inputs as the forward), copies this rank's hash columns to pinned
    host memory, waits on one event, reads every row of every engram layer in
    one parallel batch, dequantizes on the CPU and copies (async, from pinned
    memory) into each layer's persistent ``staged_rows``. The forward (eager,
    piecewise or FULL graph) only reads ``staged_rows``.
    """

    def __init__(self, hash_state: "NgramHashState", engrams: list[Any]) -> None:
        assert engrams and all(e.embed_tokens.disk is not None for e in engrams)
        self.hash_state = hash_state
        self.engrams = sorted(engrams, key=lambda e: e.layer_hash_index)
        emb = self.engrams[0].embed_tokens
        self.local_heads = emb.part_n_hash_cols
        self.head_start = emb.head_start
        self.head_end = min(emb.head_start + emb.part_n_hash_cols, emb.n_hash_cols)
        self.dim = emb.dim
        self.max_tokens = self.engrams[0].staged_rows.shape[0]
        num_layers = hash_state.multipliers.shape[0]
        self.hash_host = torch.empty(
            (self.max_tokens, num_layers, self.head_end - self.head_start),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.rows_host = [
            torch.empty(
                (self.max_tokens, self.local_heads, self.dim),
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
            )
            for _ in self.engrams
        ]
        self.hashes_ready = torch.cuda.Event()
        self.num_staged = 0
        # DSV41_ENGRAM_TIMING=1: host wall clock of stage() phases. Two
        # buckets (decode: <= 16 tokens, prefill): sync, gather+dequant, H2D
        # tail, total, calls, rows, unique rows.
        self._timing = os.environ.get("DSV41_ENGRAM_TIMING", "0") == "1"
        self._t_acc: list[list[float]] = [[0.0] * 7 for _ in range(2)]
        self._prefetch = _PREFETCH
        if self._prefetch:
            self.hash_host_next = torch.empty_like(self.hash_host)
            self.hashes_next_ready = torch.cuda.Event()
            self.last_pos_host = torch.zeros(
                256, dtype=torch.int64, device="cpu", pin_memory=True
            )
            self._prefetch_pool = ThreadPoolExecutor(
                max_workers=_PREFETCH_THREADS, thread_name_prefix="engram-prefetch"
            )
            self._prefetch_futs: list = []
            self._prefetch_stats = [0, 0, 0.0]  # lookaheads, rows, host seconds
            logger.info(
                "Engram DISK prefetch: hash-ahead of the next prefill chunk "
                "(%s, %d threads)",
                _PREFETCH_MODE,
                _PREFETCH_THREADS,
            )
        logger.info(
            "Engram DISK rows staged before the forward (graph-safe): %d layers, "
            "heads [%d, %d) of %d, up to %d tokens/step",
            len(self.engrams),
            self.head_start,
            self.head_end,
            emb.n_hash_cols,
            self.max_tokens,
        )

    def _requests(
        self, host: torch.Tensor
    ) -> list[tuple[DiskEngramTable, torch.Tensor, torch.Tensor]]:
        requests = []
        for engram in self.engrams:
            emb = engram.embed_tokens
            rel, owned = disk_rel_owned(
                host[:, engram.layer_hash_index, :].to(torch.int64),
                self.local_heads,
                emb.vocab_start_idx,
                emb.vocab_end_idx,
            )
            requests.append((emb.disk, rel, owned))
        return requests

    def _hash_ahead(
        self, input_batch, req_states, positions, query_start_loc, lookback_token_ids
    ) -> None:
        """For the batch's longest still-prefilling request, hash the next
        chunk of its (known) prompt with the same kernel and warm this rank's
        rows in the page cache while the current step computes."""
        from vllm.models.deepseek_v41.common.mm_preprocess import (
            image_sentinel_mask,
        )

        t0 = time.perf_counter()
        n_reqs = int(input_batch.num_reqs)
        if n_reqs <= 0:
            return
        idx_np = input_batch.idx_mapping_np[:n_reqs]
        prompt_len = req_states.prompt_len.np[idx_np]
        # Valid after stage()'s hashes_ready.synchronize().
        next_start = self.last_pos_host[:n_reqs].numpy() + 1
        remaining = prompt_len - next_start
        i = int(remaining.argmax())
        if remaining[i] <= 0:
            return
        req_idx = int(idx_np[i])
        start = int(next_start[i])
        length = int(min(remaining[i], self.max_tokens))
        tokens = req_states.all_token_ids.gpu
        ids = tokens[req_idx, start : start + length]
        pos = torch.arange(
            start, start + length, dtype=positions.dtype, device=positions.device
        )
        qsl = torch.tensor(
            [0, length], dtype=query_start_loc.dtype, device=query_start_loc.device
        )
        depth = int(lookback_token_ids.shape[1])
        lookback = torch.full(
            (1, depth),
            -1,
            dtype=lookback_token_ids.dtype,
            device=lookback_token_ids.device,
        )
        k = min(depth, start)
        if k > 0:
            lookback[0, :k] = tokens[req_idx, start - k : start].flip(0)
        hashes = self.hash_state(
            ids,
            pos,
            qsl,
            image_sentinel_mask(ids),
            lookback,
            image_sentinel_mask(lookback),
            None,
            None,
        )
        host = self.hash_host_next[:length]
        host.copy_(hashes[:, :, self.head_start : self.head_end], non_blocking=True)
        self.hashes_next_ready.record()
        self.hashes_next_ready.synchronize()
        self._prefetch_futs = [f for f in self._prefetch_futs if not f.done()]
        total_rows = 0
        for table, rel, owned in self._requests(host):
            rows = torch.unique(rel[owned]).tolist()
            total_rows += len(rows)
            for lo in range(0, len(rows), _PREFETCH_TASK_ROWS):
                self._prefetch_futs.append(
                    self._prefetch_pool.submit(
                        _prefetch_rows, table, rows[lo : lo + _PREFETCH_TASK_ROWS]
                    )
                )
        st = self._prefetch_stats
        st[0] += 1
        st[1] += total_rows
        st[2] += time.perf_counter() - t0
        if st[0] % 16 == 0 or length >= 4096:
            logger.info(
                "Engram prefetch: next chunk %d tokens from %d (prompt %d), %d "
                "unique rows queued, host %.1f ms (%d lookaheads so far, avg "
                "%.0f rows, %.1f ms)",
                length,
                start,
                int(prompt_len[i]),
                total_rows,
                1e3 * (time.perf_counter() - t0),
                st[0],
                st[1] / st[0],
                1e3 * st[2] / st[0],
            )

    @torch.inference_mode()
    def stage(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        num_tokens: int,
        input_batch=None,
        req_states=None,
    ) -> int:
        """Stage rows for the first `num_tokens` (unpadded) tokens of this
        step. Returns the number staged; 0 while the KV cache is unbound
        (profiling), matching the forward, which skips engram then."""
        if torch.cuda.is_current_stream_capturing():
            return 0
        n = min(int(num_tokens), self.max_tokens)
        if n <= 0 or not self.hash_state.ensure_cache():
            return 0
        from vllm.models.deepseek_v41.common.mm_preprocess import (
            image_sentinel_mask,
        )

        timing = self._timing
        t0 = time.perf_counter() if timing else 0.0
        ids = input_ids[:n]
        hashes = self.hash_state(
            ids,
            positions[:n],
            query_start_loc,
            image_sentinel_mask(ids),
            lookback_token_ids,
            image_sentinel_mask(lookback_token_ids),
            None,
            None,
        )
        host = self.hash_host[:n]
        host.copy_(hashes[:, :, self.head_start : self.head_end], non_blocking=True)
        prefetch = self._prefetch and input_batch is not None and req_states is not None
        if prefetch:
            # Last position of every request in this step -> next chunk start.
            n_reqs = int(input_batch.num_reqs)
            if n_reqs > self.last_pos_host.shape[0]:
                self.last_pos_host = torch.zeros(
                    n_reqs * 2, dtype=torch.int64, device="cpu", pin_memory=True
                )
            ends = torch.as_tensor(
                input_batch.query_start_loc_np[1 : n_reqs + 1].astype("int64") - 1,
                device=positions.device,
            )
            self.last_pos_host[:n_reqs].copy_(positions[ends], non_blocking=True)
        self.hashes_ready.record()
        # The one host sync per step. It also orders this step's rewrite of
        # the pinned buffers after the previous step's async H2D copies.
        self.hashes_ready.synchronize()
        t1 = time.perf_counter() if timing else 0.0
        requests = self._requests(host)
        rows_per_layer = gather_dequant_many(requests)
        t2 = time.perf_counter() if timing else 0.0
        for engram, rows, buf in zip(self.engrams, rows_per_layer, self.rows_host):
            staged = buf[:n]
            staged.copy_(rows.view(n, self.local_heads, self.dim))
            engram.staged_rows[:n].copy_(staged, non_blocking=True)
        self.num_staged = n
        if timing:
            self._log_timing(n, requests, t0, t1, t2, time.perf_counter())
        if prefetch:
            try:
                self._hash_ahead(
                    input_batch,
                    req_states,
                    positions,
                    query_start_loc,
                    lookback_token_ids,
                )
            except Exception as e:  # prefetch is best effort
                logger.warning_once("Engram prefetch disabled after error: %r", e)
                self._prefetch = False
        return n

    def _log_timing(
        self,
        n: int,
        requests: list[tuple[DiskEngramTable, torch.Tensor, torch.Tensor]],
        t0: float,
        t1: float,
        t2: float,
        t3: float,
    ) -> None:
        rows = sum(int(r[1].numel()) for r in requests)
        uniq = sum(int(torch.unique(r[1]).numel()) for r in requests)
        mode = "mmap" if _MMAP and rows >= _MMAP_MIN_ROWS * len(requests) else "preadv"
        if n >= 1024:
            logger.info(
                "Engram prefill stage [%s]: %d tokens, %d rows (%d unique), "
                "gather+dequant %.1f ms, total %.1f ms",
                mode,
                n,
                rows,
                uniq,
                1e3 * (t2 - t1),
                1e3 * (t3 - t0),
            )
        acc = self._t_acc[0 if n <= 16 else 1]
        for i, value in enumerate((t1 - t0, t2 - t1, t3 - t2, t3 - t0, 1, rows, uniq)):
            acc[i] += value
        calls = acc[4]
        if calls % (200 if n <= 16 else 20) == 0:
            logger.info(
                "Engram stage timing [%s, %s]: total %.2f ms/step (sync wait "
                "%.2f, disk+dequant %.2f, h2d+rest %.2f), %.1f rows/step (%.1f "
                "unique), avg of %d steps",
                mode,
                "decode" if n <= 16 else "prefill",
                1e3 * acc[3] / calls,
                1e3 * acc[0] / calls,
                1e3 * acc[1] / calls,
                1e3 * acc[2] / calls,
                acc[5] / calls,
                acc[6] / calls,
                calls,
            )
            acc[:] = [0.0] * 7
