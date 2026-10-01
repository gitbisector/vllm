# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks of DSV41_ENGRAM_DISK row reads against a tiny checkpoint."""

import json
import struct
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="posix_fadvise / MADV_RANDOM are Linux-only"
)

LAYER = 3
ROWS, DIM, BLOCK = 40, 64, 32
SB = DIM // BLOCK


def _write_shard(path, tensors):
    """tensors: [(name, dtype string, shape, raw bytes)] in file order."""
    header, offset = {}, 0
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for *_, data in tensors:
            f.write(data)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Weight and scale in two shards, each behind a decoy tensor, so neither
    starts at data offset 0."""
    root = tmp_path_factory.mktemp("ckpt")
    gen = torch.Generator().manual_seed(0)
    weight = (torch.randn(ROWS, DIM, generator=gen) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(120, 135, (ROWS, SB), dtype=torch.uint8, generator=gen)
    wname = f"layers.{LAYER}.engram.embed.weight"
    sname = f"layers.{LAYER}.engram.embed.scale"
    decoy = bytes(range(24))
    _write_shard(
        root / "a.safetensors",
        [
            ("decoy.a", "U8", (24,), decoy),
            (wname, "F8_E4M3", (ROWS, DIM), weight.view(torch.uint8).numpy().tobytes()),
        ],
    )
    _write_shard(
        root / "b.safetensors",
        [
            ("decoy.b", "U8", (24,), decoy),
            (sname, "F8_E8M0", (ROWS, SB), scale.numpy().tobytes()),
        ],
    )
    index = {"weight_map": {wname: "a.safetensors", sname: "b.safetensors"}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    # Independent reference: fp8 value x 2^(e - 127) per block.
    exp = torch.pow(2.0, scale.to(torch.float32) - 127)
    ref = weight.to(torch.float32) * exp.repeat_interleave(BLOCK, dim=1)
    return root, ref


@pytest.fixture
def disk(monkeypatch):
    from vllm.models.deepseek_v41.nvidia import engram_disk

    monkeypatch.setattr(engram_disk, "_POOL", None)
    monkeypatch.delenv("DSV41_ENGRAM_DIR", raising=False)
    return engram_disk


@pytest.mark.parametrize(
    "mmap, inline_rows",
    [(False, 2048), (True, 2048), (True, 0)],
    ids=["preadv", "mmap-inline", "mmap-pool"],
)
def test_rows_match_reference(checkpoint, disk, monkeypatch, mmap, inline_rows):
    root, ref = checkpoint
    monkeypatch.setattr(disk, "_MMAP", mmap)
    monkeypatch.setattr(disk, "_MMAP_INLINE_ROWS", inline_rows)
    row_start, num_rows = 10, 20
    table = disk.DiskEngramTable(str(root), LAYER, DIM, BLOCK, row_start, num_rows)
    assert (table.w_view is not None) == mmap
    # Repeats exercise the de-duplication; owned=False rows read as zeros.
    rel = torch.tensor([0, 5, 19, 5, 0, 7, 12, 19], dtype=torch.int64)
    owned = torch.tensor([True, True, True, True, False, True, False, True])
    out = table.gather_dequant(rel, owned)
    assert out.dtype == torch.bfloat16 and out.shape == (8, DIM)
    expected = ref[row_start + rel].to(torch.bfloat16)
    expected[~owned] = 0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_rel_owned_pads_and_masks(disk):
    local = torch.tensor([[3, 9], [12, 4]], dtype=torch.int64)
    rel, owned = disk.disk_rel_owned(local, 3, vocab_start=4, vocab_end=12)
    assert owned.tolist() == [False, True, False, False, True, False]
    assert rel.tolist() == [0, 5, 0, 0, 0, 0]


def test_local_dir_guard(checkpoint, disk, monkeypatch, tmp_path):
    root, _ = checkpoint
    monkeypatch.setenv("DSV41_ENGRAM_DIR", str(tmp_path))
    (tmp_path / "engram-local.json").write_text(
        json.dumps({"layers": {str(LAYER): [10, 30]}})
    )
    assert disk._local_dir(str(root), LAYER, 10, 20) == str(tmp_path)
    # A rank map the copy does not cover, or an unknown layer, reads model_dir.
    assert disk._local_dir(str(root), LAYER, 20, 20) == str(root)
    assert disk._local_dir(str(root), LAYER + 1, 10, 20) == str(root)


def test_skip_checkpoint_tensor(disk, monkeypatch):
    names = [
        f"layers.{LAYER}.engram.embed.weight",
        f"layers.{LAYER}.engram.embed.scale",
    ]
    monkeypatch.setattr(disk, "ENGRAM_DISK", False)
    assert not any(map(disk.skip_engram_checkpoint_tensor, names))
    monkeypatch.setattr(disk, "ENGRAM_DISK", True)
    assert all(map(disk.skip_engram_checkpoint_tensor, names))
    assert not disk.skip_engram_checkpoint_tensor(f"layers.{LAYER}.engram.wkv.weight")
