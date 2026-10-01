# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of DeepSeek-V4.1 slice-read: the byte-range planning, the planner
patch and its model check (against a stub fastsafetensors), and the checksum
dump."""

import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.nvidia import slice_read as sr

TP = 4
HEADER = 1000  # SafeTensorsMetadata.header_length


def _frames() -> dict[str, dict]:
    """A synthetic shard: a sliced expert w1 first (so it opens the chunk), a
    replicated norm, a sliced wq_b, and a dim-1 sharded w2 last."""
    shapes = {
        "layers.0.ffn.experts.0.w1.weight": ([16, 64], 1),
        "layers.0.attn.norm.weight": ([64], 2),
        "layers.0.attn.wq_b.weight": ([32, 128], 1),
        "layers.0.ffn.experts.0.w2.weight": ([64, 8], 1),
    }
    frames, off = {}, 0
    for name, (shape, itemsize) in shapes.items():
        n = itemsize
        for d in shape:
            n *= d
        frames[name] = {"shape": shape, "data_offsets": [off, off + n]}
        off += n
    return frames


def _abs(frames, name):
    s, e = frames[name]["data_offsets"]
    return HEADER + s, HEADER + e


def _covered(runs, s, e) -> bool:
    return any(a <= s and e <= b for a, b in runs)


def test_row_slice_requires_exact_multiple():
    assert sr.row_slice([16, 64], 1024, 1, 4) == (256, 512)
    assert sr.row_slice([18, 64], 18 * 64, 1, 4) is None  # rows % tp
    assert sr.row_slice([2, 64], 128, 0, 4) is None  # rows < tp
    assert sr.row_slice([16, 64], 1024, 0, 1) is None  # no TP


def test_rule_set():
    assert sr.rule_for("layers.3.ffn.experts.17.w3.scale") == "expert_w13_scale"
    assert sr.rule_for("layers.3.attn.wo_a.weight") == "attn_wo_a_weight"
    assert sr.rule_for("layers.3.ffn.experts.17.w2.weight") is None
    assert sr.rule_for("layers.3.attn.wo_b.weight") is None
    assert sr.rule_for("mtp.0.attn.wq_b.weight") is None


@pytest.mark.parametrize("rank", range(TP))
def test_rewrite_chunk_is_anchored_and_reads_own_rows(rank):
    frames = _frames()
    names = set(frames)
    start = HEADER
    end = HEADER + max(f["data_offsets"][1] for f in frames.values())
    runs, stats = sr.rewrite_chunk(
        frames, HEADER, names, [(start, end)], rank, TP, merge_gap=0
    )
    # The span, and so the copier's base offset, is unchanged.
    assert runs[0][0] == start and runs[-1][1] == end
    assert stats["sliced"] == 2 and stats["whole"] == 2
    for name in ("layers.0.ffn.experts.0.w1.weight", "layers.0.attn.wq_b.weight"):
        s, e = _abs(frames, name)
        row = (e - s) // frames[name]["shape"][0]
        per = frames[name]["shape"][0] // TP
        assert _covered(runs, s + rank * per * row, s + (rank + 1) * per * row)
    for name in ("layers.0.attn.norm.weight", "layers.0.ffn.experts.0.w2.weight"):
        assert _covered(runs, *_abs(frames, name))
    assert sum(b - a for a, b in runs) < end - start


class _Frame:
    def __init__(self, d):
        self.shape = d["shape"]
        self.data_offsets = d["data_offsets"]


def _stub_plan_chunks(metadata, max_batch_bytes, keep_tensor=None, merge_gap=4096):
    s = min(
        metadata.header_length + f.data_offsets[0] for f in metadata.tensors.values()
    )
    e = max(
        metadata.header_length + f.data_offsets[1] for f in metadata.tensors.values()
    )
    return [(set(metadata.tensors), [(s, e)])]


@pytest.fixture
def fake_fst(monkeypatch):
    fst = types.ModuleType("fastsafetensors")
    fpl = types.ModuleType("fastsafetensors.parallel_loader")
    fpl.plan_chunks = _stub_plan_chunks  # type: ignore[attr-defined]
    fst.parallel_loader = fpl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastsafetensors", fst)
    monkeypatch.setitem(sys.modules, "fastsafetensors.parallel_loader", fpl)
    monkeypatch.setattr("importlib.metadata.version", lambda _: "0.4.0")
    monkeypatch.setenv("DSV41_SLICE_READ", "1")
    return fpl


def _model(rank: int, tp: int) -> torch.nn.Module:
    """Modules shaped like the audited DeepSeek-V4.1-Flash target layers."""
    RoutedExperts = type("RoutedExperts", (torch.nn.Module,), {})
    Mxfp4MoEMethod = type("Mxfp4MoEMethod", (), {})
    ColumnParallelLinear = type("ColumnParallelLinear", (torch.nn.Module,), {})

    experts = RoutedExperts()
    experts.w13_weight = torch.nn.Parameter(
        torch.empty(1, 2 * (2304 // tp), 1), requires_grad=False
    )
    experts.quant_method = Mxfp4MoEMethod()
    experts.moe_config = SimpleNamespace(
        tp_rank=rank, moe_parallel_config=SimpleNamespace(tp_size=tp)
    )

    def linear(rows):
        m = ColumnParallelLinear()
        m.weight = torch.nn.Parameter(torch.empty(rows, 1), requires_grad=False)
        m.tp_rank, m.tp_size = rank, tp
        return m

    attn = torch.nn.Module()
    attn.wq_b = linear(32768 // tp)
    attn.wo_a = linear(8192 // tp)
    layer = torch.nn.Module()
    layer.attn = attn
    layer.experts = experts
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList([layer])
    return root


def _metadata():
    frames = _frames()
    return SimpleNamespace(
        header_length=HEADER, tensors={n: _Frame(f) for n, f in frames.items()}
    )


def test_sliced_planner_patches_and_restores(fake_fst, monkeypatch):
    monkeypatch.setattr(sr, "MODEL", _model(1, TP))
    full = _stub_plan_chunks(_metadata(), 1 << 20)[0][1]
    with sr.sliced_planner(1, TP, all_local=True, planned=True):
        assert fake_fst.plan_chunks is not _stub_plan_chunks
        [(names, runs)] = fake_fst.plan_chunks(_metadata(), 1 << 20, merge_gap=0)
    assert fake_fst.plan_chunks is _stub_plan_chunks
    assert names == set(_frames())
    assert runs[0][0] == full[0][0] and runs[-1][1] == full[-1][1]
    assert sum(b - a for a, b in runs) < full[0][1] - full[0][0]


def test_failed_model_check_reads_whole(fake_fst, monkeypatch):
    # Built for rank 0 but loading as rank 1: every rule must be refused.
    monkeypatch.setattr(sr, "MODEL", _model(0, TP))
    with sr.sliced_planner(1, TP, all_local=True, planned=True):
        assert fake_fst.plan_chunks is _stub_plan_chunks


@pytest.mark.parametrize(
    "all_local,planned,registered",
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_disabled_without_preconditions(
    fake_fst, monkeypatch, all_local, planned, registered
):
    monkeypatch.setattr(sr, "MODEL", _model(1, TP) if registered else None)
    with sr.sliced_planner(1, TP, all_local=all_local, planned=planned):
        assert fake_fst.plan_chunks is _stub_plan_chunks


def test_unsupported_fastsafetensors_fails_loudly(fake_fst, monkeypatch):
    monkeypatch.setattr(sr, "MODEL", _model(1, TP))
    monkeypatch.setattr("importlib.metadata.version", lambda _: "0.5.0")
    with (
        pytest.raises(RuntimeError, match="0.4.x"),
        sr.sliced_planner(1, TP, all_local=True, planned=True),
    ):
        pass
    assert fake_fst.plan_chunks is _stub_plan_chunks


def test_dump_checksums(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setenv("DSV41_SLICE_VERIFY", str(tmp_path))
    m = torch.nn.Module()
    m.a = torch.nn.Parameter(
        torch.arange(300, dtype=torch.float32), requires_grad=False
    )
    sr.dump_checksums(m, {"a", "not_a_param"}, 2, {"k": 1})
    rows = (tmp_path / "rank2.tsv").read_text().splitlines()
    name, dtype, shape, n, digest = rows[0].split("\t")
    assert (name, dtype, shape, int(n)) == ("a", "float32", "300", 1200)
    assert digest == hashlib.sha256(m.a.data.numpy().tobytes()).hexdigest()
    assert rows[1].startswith("not_a_param\tNOTAPARAM")
    assert os.path.exists(tmp_path / "rank2.json")


def test_vl_mapper_drops_mtp():
    # DSV41_TARGET_SKIP_MTP only skips mtp.* when the target's mapper drops it.
    from vllm.models.deepseek_v41.nvidia.vl_model import (
        _make_deepseek_v4_vl_weights_mapper,
    )

    mapper = _make_deepseek_v4_vl_weights_mapper("fp4", "weight_scale")
    assert mapper.map_name("mtp.0.attn.wq_b.weight") is None
    assert mapper.map_name("layers.0.attn.wq_b.weight") is not None


def test_real_fastsafetensors_reads_rank_rows(tmp_path, monkeypatch):
    """The patched planner through fastsafetensors' real copier, on CPU."""
    pytest.importorskip("fastsafetensors")
    from fastsafetensors import SingleGroup
    from fastsafetensors.parallel_loader import ParallelLoader
    from safetensors.torch import save_file

    rank = 2
    g = torch.Generator().manual_seed(0)

    def i8(*shape):
        return torch.randint(-128, 127, shape, generator=g, dtype=torch.int8)

    # Sliced tensors open and close the shard, which the anchoring must handle.
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": i8(2304, 640),
        "layers.0.attn.norm.weight": torch.randn(512, generator=g).bfloat16(),
        "layers.0.attn.wq_b.weight": i8(32768, 16),
        "layers.0.ffn.experts.0.w2.weight": i8(512, 1152),
        "layers.0.ffn.experts.0.w3.weight": i8(2304, 640),
    }
    path = str(tmp_path / "model-00001-of-00001.safetensors")
    save_file(tensors, path)

    monkeypatch.setenv("DSV41_SLICE_READ", "1")
    monkeypatch.setattr(sr, "MODEL", _model(rank, TP))
    with sr.sliced_planner(rank, TP, all_local=True, planned=True):
        loader = ParallelLoader(
            pg=SingleGroup(),
            hf_weights_files=[path],
            device="cpu",
            nogds=True,
            all_local=True,
            device_memory_budget=64 << 20,
            max_batch_bytes=2 * 2304 * 640,
        )
    loaded = {n: t.clone() for n, t in loader.iterate_weights()}

    for name, expected in tensors.items():
        got = loaded[name]
        assert got.shape == expected.shape
        if sr.rule_for(name) is None:
            assert torch.equal(got, expected), name
        else:
            per = expected.shape[0] // TP
            rows = slice(rank * per, (rank + 1) * per)
            assert torch.equal(got[rows], expected[rows]), name
