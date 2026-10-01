# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read only this TP rank's rows of dim-0-sharded tensors (``DSV41_SLICE_READ``).

DeepSeek-V4.1-Flash under ``--load-format fastsafetensors`` with
``VLLM_FASTSAFETENSORS_ALL_LOCAL=1``: every rank reads the whole checkpoint,
though for the row-sharded classes below it keeps a quarter of it at TP4.
This wraps fastsafetensors' chunk planner so each planned chunk's byte runs
cover only this rank's rows of those classes (production: 284 -> 147 GiB read
per rank).

Sparse fill: the staging buffer, the chunk span and every yielded tensor keep
their full checkpoint shape. vLLM's own weight loaders narrow exactly the
rows that were read; the unread bytes of a yielded tensor are stale staging
memory that those loaders never consume. Nothing is pre-narrowed, which
matters because ``RoutedExperts._load_w13`` derives the per-rank size from
the tensor's own shape.

Only name patterns audited against the loaders are sliced (``RULES``), and
only after the live model's parameters are checked to be narrowed the way
each rule assumes (same TP rank, rows // tp per rank). A failed check disables
that class -- it is read whole, never wrongly sliced.

It patches ``fastsafetensors.parallel_loader.plan_chunks``, a private symbol,
so it requires fastsafetensors 0.4.x and fails loudly on anything else.
Slicing needs a planned load (a device memory budget) and per-rank reads
(all_local); otherwise the load reads whole tensors as usual.

``DSV41_SLICE_VERIFY=<dir>`` writes a sha256 of every loaded parameter per rank,
to compare a sliced load against a full one byte for byte.
"""

from __future__ import annotations

import contextlib
import inspect
import os
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import regex as re

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from torch import nn

logger = init_logger(__name__)

# Set by the VL model around its target load_weights, so the planner (which
# runs lazily, inside the first next() of the weight stream) can check the live
# parameters. None -- e.g. during the draft pass -- disables slicing.
MODEL: nn.Module | None = None

ALN = 4096  # O_DIRECT alignment of fastsafetensors' dma_load_runs

# Checkpoint (pre-mapper) names whose TP shard is dim-0 rows
# [r * R / tp, (r + 1) * R / tp), with the loader that performs the narrow.
# Deliberately not here:
#   experts w2 / w2.scale, attn.wo_b, shared_experts.w2 -> dim-1 shard: a
#       stride shorter than ALN, so O_DIRECT reads the whole tensor anyway
#   linear .scale tensors -> tiny
#   shared_experts.w1/w3 -> _pad_shared_expert_weight may torch.cat the whole
#       tensor, unread rows included
#   mtp.* -> the draft (DSpark) loaders are not audited
#   embed / head -> load from lazy mmap, not fastsafetensors
RULES = [
    # RoutedExperts._load_w13: shard_dim 0, loaded_per_rank = shape[0] // tp
    (
        "expert_w13_weight",
        re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[13]\.weight$"),
    ),
    (
        "expert_w13_scale",
        re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[13]\.scale$"),
    ),
    # ColumnParallelLinear -> _ColumnvLLMParameter.load_column_parallel_weight:
    # narrow(output_dim=0, tp_rank * shard_size, shard_size)
    ("attn_wq_b_weight", re.compile(r"^layers\.\d+\.attn\.wq_b\.weight$")),
    ("attn_wo_a_weight", re.compile(r"^layers\.\d+\.attn\.wo_a\.weight$")),
]

# DeepSeek-V4.1-Flash dims the model check expects (per-rank = these // tp).
_MOE_INTERMEDIATE = 2304
_WQ_B_ROWS = 32768
_WO_A_ROWS = 8192

# fastsafetensors.parallel_loader.plan_chunks as of 0.4.x.
_PLAN_CHUNKS_PARAMS = ["metadata", "max_batch_bytes", "keep_tensor", "merge_gap"]


def rule_for(name: str) -> str | None:
    for label, pat in RULES:
        if pat.match(name):
            return label
    return None


def row_slice(
    shape: list[int], nbytes: int, rank: int, tp: int
) -> tuple[int, int] | None:
    """[start, end) byte offsets, relative to the tensor start, of this rank's
    dim-0 shard, or None if it cannot be sliced exactly as vLLM narrows it.

    Requires rows % tp == 0: the loaders use floor division (RoutedExperts) or
    the parameter's own row count (column-parallel); with an exact multiple
    both are rows // tp and the slice is unambiguous."""
    if tp <= 1 or not shape:
        return None
    rows = shape[0]
    if rows < tp or rows % tp or nbytes % rows:
        return None
    row_bytes = nbytes // rows
    per = rows // tp
    return rank * per * row_bytes, (rank + 1) * per * row_bytes


def tensor_runs(
    name: str,
    frame: dict,
    header_length: int,
    rank: int,
    tp: int,
    enabled: bool = True,
) -> tuple[list[tuple[int, int]], bool]:
    """Absolute file runs to read for one tensor, and whether it was sliced.

    ``frame`` is the safetensors header entry ({shape, data_offsets});
    ``header_length`` is fastsafetensors' SafeTensorsMetadata.header_length
    (8 + JSON header length)."""
    s = header_length + frame["data_offsets"][0]
    e = header_length + frame["data_offsets"][1]
    if s == e:
        return [], False
    if enabled and rule_for(name) is not None:
        rs = row_slice(frame["shape"], e - s, rank, tp)
        if rs is not None:
            return [(s + rs[0], s + rs[1])], True
    return [(s, e)], False


def merge(runs: Iterable[tuple[int, int]], gap: int = ALN) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(runs):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def rewrite_chunk(
    frames: dict[str, dict],
    header_length: int,
    names: set[str],
    ranges: list[tuple[int, int]],
    rank: int,
    tp: int,
    merge_gap: int = ALN,
) -> tuple[list[tuple[int, int]], dict]:
    """Replace a planned chunk's byte runs with this rank's runs.

    The planner's ``ranges`` cover whole tensors. The copier maps the staging
    buffer's start to ``min(run starts)`` and materializes every tensor at
    ``header + data_offsets[0] - base``. If a sliced tensor opened the chunk,
    dropping its leading rows would move that base forward and every
    full-shape view in the chunk would start before the buffer. So the runs
    are anchored: they always contain the chunk's first and last byte, which
    keeps the base and the span identical to the unsliced plan, at a cost of
    at most two extra aligned pages per chunk.
    """
    orig_start = min(s for s, _ in ranges)
    orig_end = max(e for _, e in ranges)
    runs: list[tuple[int, int]] = []
    stats = {"sliced": 0, "whole": 0, "sliced_saved": 0}
    for n in names:
        tr, sliced = tensor_runs(n, frames[n], header_length, rank, tp)
        runs.extend(tr)
        if sliced:
            stats["sliced"] += 1
            f = frames[n]["data_offsets"]
            stats["sliced_saved"] += (f[1] - f[0]) - sum(e - s for s, e in tr)
        else:
            stats["whole"] += 1
    runs.append((orig_start, orig_start + 1))
    runs.append((orig_end - 1, orig_end))
    out = merge(runs, merge_gap)
    assert out[0][0] == orig_start and out[-1][1] == orig_end, (out[:1], out[-1:])
    return out, stats


def _check_fastsafetensors(fpl) -> None:
    """Fail loudly unless the private planner is the one this was written for."""
    from importlib.metadata import version

    ver = version("fastsafetensors")
    params = list(inspect.signature(fpl.plan_chunks).parameters)
    if not ver.startswith("0.4.") or params != _PLAN_CHUNKS_PARAMS:
        raise RuntimeError(
            f"DSV41_SLICE_READ=1 patches fastsafetensors' private plan_chunks "
            f"and supports fastsafetensors 0.4.x with plan_chunks"
            f"({', '.join(_PLAN_CHUNKS_PARAMS)}); found {ver} with "
            f"plan_chunks({', '.join(params)}). Unset DSV41_SLICE_READ."
        )


def _checked_rules(model: nn.Module, rank: int, tp: int) -> set[str]:
    """Rule labels whose assumption holds on the live model."""
    ok: set[str] = set()
    expert_ok = attn_ok = None
    for mod_name, mod in model.named_modules():
        w13 = getattr(mod, "w13_weight", None)
        if expert_ok is None and w13 is not None and ".mtp." not in mod_name:
            cfg = getattr(mod, "moe_config", None)
            ptp = getattr(getattr(cfg, "moe_parallel_config", None), "tp_size", None)
            prank = getattr(cfg, "tp_rank", None)
            # Audited: the fused leg (RoutedExperts + Mxfp4MoEMethod) only; the
            # mega-MoE leg has other loaders. moe tp_rank is the flattened
            # dp/pcp/tp rank, so comparing it with the TP rank also disables
            # slicing under dp or pcp. w13_weight is (E, 2 * I_per_rank, H/2).
            fused_leg = (
                any(c.__name__ == "RoutedExperts" for c in type(mod).__mro__)
                and type(getattr(mod, "quant_method", None)).__name__
                == "Mxfp4MoEMethod"
            )
            expert_ok = (
                fused_leg
                and ptp == tp
                and prank == rank
                and w13.shape[-2] == 2 * (_MOE_INTERMEDIATE // tp)
            )
            logger.info(
                "DSV41 slice-read: %s w13 %s moe tp=%s rank=%s -> %s",
                mod_name,
                tuple(w13.shape),
                ptp,
                prank,
                expert_ok,
            )
        if (
            attn_ok is None
            and mod_name.endswith(".attn.wq_b")
            and ".mtp." not in mod_name
        ):
            wo_a = model.get_submodule(mod_name[: -len("wq_b")] + "wo_a")
            attn_ok = (
                type(mod).__name__ == "ColumnParallelLinear"
                and type(wo_a).__name__ == "ColumnParallelLinear"
                and getattr(mod, "tp_rank", None) == rank
                and getattr(mod, "tp_size", None) == tp
                and getattr(wo_a, "tp_rank", None) == rank
                and mod.weight.shape[0] == _WQ_B_ROWS // tp
                and wo_a.weight.shape[0] == _WO_A_ROWS // tp
            )
            logger.info(
                "DSV41 slice-read: %s %s tp_rank=%s weight %s -> %s",
                mod_name,
                type(mod).__name__,
                getattr(mod, "tp_rank", None),
                tuple(mod.weight.shape),
                attn_ok,
            )
    if expert_ok:
        ok |= {"expert_w13_weight", "expert_w13_scale"}
    if attn_ok:
        ok |= {"attn_wq_b_weight", "attn_wo_a_weight"}
    return ok


@contextlib.contextmanager
def sliced_planner(
    rank: int, tp: int, all_local: bool, planned: bool
) -> Iterator[None]:
    """Patch fastsafetensors.parallel_loader.plan_chunks for the duration of a
    ParallelLoader construction (it plans in __init__)."""
    model = MODEL
    if not envs.DSV41_SLICE_READ or model is None:
        yield
        return
    import fastsafetensors.parallel_loader as fpl

    _check_fastsafetensors(fpl)
    if tp <= 1 or not all_local or not planned:
        # Without all_local a rank's chunk is broadcast to ranks that need
        # other rows; without a plan there are no chunks to rewrite.
        logger.warning(
            "DSV41 slice-read: disabled (tp=%d, all_local=%s, planned=%s); "
            "it needs tp > 1, VLLM_FASTSAFETENSORS_ALL_LOCAL=1 and a planned "
            "load",
            tp,
            all_local,
            planned,
        )
        yield
        return
    rules = _checked_rules(model, rank, tp)
    if not rules:
        logger.warning("DSV41 slice-read: no rule passed the model check; full read")
        yield
        return

    orig = fpl.plan_chunks
    totals = {"chunks": 0, "sliced": 0, "saved": 0}

    def plan_chunks(metadata, max_batch_bytes, keep_tensor=None, merge_gap=ALN):
        chunks = orig(
            metadata, max_batch_bytes, keep_tensor=keep_tensor, merge_gap=merge_gap
        )
        frames = {
            n: {"shape": list(f.shape), "data_offsets": list(f.data_offsets)}
            for n, f in metadata.tensors.items()
        }
        out = []
        for names, ranges in chunks:
            sliceable = {n for n in names if rule_for(n) in rules}
            if not sliceable:
                out.append((names, ranges))
                continue
            # rewrite_chunk consults RULES itself; pass classes that failed the
            # model check through as whole tensors.
            runs = []
            for n in names - sliceable:
                runs.extend(
                    tensor_runs(
                        n, frames[n], metadata.header_length, rank, tp, enabled=False
                    )[0]
                )
            new, st = rewrite_chunk(
                frames, metadata.header_length, sliceable, ranges, rank, tp, merge_gap
            )
            totals["chunks"] += 1
            totals["sliced"] += st["sliced"]
            totals["saved"] += st["sliced_saved"]
            out.append((names, merge(new + runs, merge_gap)))
        return out

    fpl.plan_chunks = plan_chunks
    try:
        yield
    finally:
        fpl.plan_chunks = orig
        logger.info(
            "DSV41 slice-read: rank %d/%d rules=%s chunks=%d sliced_tensors=%d "
            "skipped_read=%.2f GiB",
            rank,
            tp,
            sorted(rules),
            totals["chunks"],
            totals["sliced"],
            totals["saved"] / (1 << 30),
        )


def proc_read_bytes() -> int:
    """This process's block-device reads so far (/proc/self/io read_bytes;
    counts O_DIRECT, excludes page-cache hits), or -1 if unavailable."""
    try:
        with open("/proc/self/io") as fh:
            for line in fh:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def dump_checksums(
    model: nn.Module, loaded_names: Iterable[str], rank: int, meta: dict
) -> None:
    """sha256 of every loaded parameter's bytes, written to
    $DSV41_SLICE_VERIFY/rank{rank}.tsv (+ .json meta).

    Call right after the target load_weights, before the loader's
    process_weights_after_loading (quant repacks), so it hashes exactly what
    the weight loaders wrote. Memory: 4 threads x one 64 MiB host buffer; the
    device copy is chunked. hashlib releases the GIL on large updates, so the
    threads hash in parallel.
    """
    verify_dir = envs.DSV41_SLICE_VERIFY
    if not verify_dir:
        return
    import hashlib
    import json
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import torch

    t0 = time.time()
    chunk = 64 << 20
    params = dict(model.named_parameters())
    loaded = set(loaded_names)
    names = sorted(n for n in loaded if n in params)
    tls = threading.local()

    def one(name: str) -> tuple[str, str, str, int, str]:
        t = params[name].data
        if not t.is_contiguous():
            t = t.contiguous()
        flat = t.reshape(-1).view(torch.uint8)
        buf = getattr(tls, "buf", None)
        if buf is None:
            buf = tls.buf = torch.empty(chunk, dtype=torch.uint8)
        h = hashlib.sha256()
        n = flat.numel()
        for o in range(0, n, chunk):
            k = min(chunk, n - o)
            buf[:k].copy_(flat[o : o + k])
            h.update(memoryview(buf[:k].numpy()))
        dtype = str(t.dtype).replace("torch.", "")
        return name, dtype, "x".join(map(str, t.shape)), n, h.hexdigest()

    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, names))
    os.makedirs(verify_dir, exist_ok=True)
    with open(os.path.join(verify_dir, f"rank{rank}.tsv"), "w") as fh:
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
        for n in sorted(loaded - set(params)):
            fh.write(f"{n}\tNOTAPARAM\t-\t0\t-\n")
    meta = dict(
        meta,
        rank=rank,
        params_hashed=len(rows),
        bytes_hashed=sum(r[3] for r in rows),
        hash_seconds=round(time.time() - t0, 1),
        slice_read=envs.DSV41_SLICE_READ,
        skip_target_mtp=envs.DSV41_TARGET_SKIP_MTP,
    )
    with open(os.path.join(verify_dir, f"rank{rank}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    logger.info(
        "DSV41 slice-verify: rank %d hashed %d params (%.2f GiB) in %.1f s -> %s",
        rank,
        len(rows),
        meta["bytes_hashed"] / (1 << 30),
        meta["hash_seconds"],
        verify_dir,
    )
