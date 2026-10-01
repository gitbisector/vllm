# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import tempfile

import huggingface_hub.constants
import pytest
import torch

from vllm.model_executor.model_loader.weight_utils import (
    _fastsafetensors_needs_plan,
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform

GiB = 1 << 30


def _download_gpt2(tmpdir):
    huggingface_hub.constants.HF_HUB_OFFLINE = False
    download_weights_from_hf(
        "openai-community/gpt2", allow_patterns=["*.safetensors"], cache_dir=tmpdir
    )
    safetensors = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
    assert len(safetensors) > 0
    return safetensors


def _assert_matches_safetensors(tmpdir):
    safetensors = _download_gpt2(tmpdir)

    fastsafetensors_tensors = {
        name: tensor
        for name, tensor in fastsafetensors_weights_iterator(safetensors, True)
    }
    hf_safetensors_tensors = {
        name: tensor for name, tensor in safetensors_weights_iterator(safetensors, True)
    }

    assert len(fastsafetensors_tensors) == len(hf_safetensors_tensors)
    for name, fastsafetensors_tensor in fastsafetensors_tensors.items():
        fastsafetensors_tensor = fastsafetensors_tensor.to("cpu")
        assert fastsafetensors_tensor.dtype == hf_safetensors_tensors[name].dtype
        assert fastsafetensors_tensor.shape == hf_safetensors_tensors[name].shape
        assert torch.all(fastsafetensors_tensor.eq(hf_safetensors_tensors[name]))


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
@pytest.mark.parametrize("queue_size", [0, 1])
def test_fastsafetensors_model_loader(monkeypatch, queue_size):
    monkeypatch.setenv("VLLM_FASTSAFETENSORS_QUEUE_SIZE", str(queue_size))
    with tempfile.TemporaryDirectory() as tmpdir:
        _assert_matches_safetensors(tmpdir)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
def test_fastsafetensors_sub_shard_chunking(monkeypatch):
    """A budget below the shard size must still yield identical tensors.

    GPT-2 is a single 522 MiB shard whose largest tensor (wte.weight) is
    147 MiB. The planner's floor is *twice* that, not once: it double-buffers
    the transient copy, and does so even after collapsing the pipeline to
    fully serial, so the smallest satisfiable budget is 2 x 147 = 294 MiB.

    384 MiB therefore sits above the floor and below the shard, forcing the
    planner to split the shard -- it loads as 4 chunks. Below 294 MiB the shard
    cannot be planned and loads from lazy mmap instead, which
    test_fastsafetensors_budget_below_largest_tensor_loads_lazily covers.
    """
    monkeypatch.setenv(
        "VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET", str(384 * 1024 * 1024)
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _assert_matches_safetensors(tmpdir)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
def test_fastsafetensors_budget_below_largest_tensor_loads_lazily(monkeypatch):
    """A shard the planner cannot place loads from lazy mmap, not an error.

    GPT-2's largest tensor (wte.weight) is 147 MiB, so no plan fits a 1 MiB
    budget: a chunk must hold that tensor twice. Rather than fail the load,
    the shard goes through the lazy iterator, which copies tensor by tensor
    and needs no staging buffer, and must yield identical tensors.
    """
    monkeypatch.setenv("VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET", str(1024 * 1024))
    with tempfile.TemporaryDirectory() as tmpdir:
        _assert_matches_safetensors(tmpdir)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
def test_fastsafetensors_accumulate_resident_skips_derived_budget(monkeypatch):
    """A consumer that keeps less than it reads must not be planned for.

    The planner charges every byte read as resident, so a consumer whose
    parameters materialize during the load -- online quantization stores a
    smaller quantized parameter than the checkpoint bytes it consumes -- would
    be over-charged and refused a load that fits. The derived budget is
    therefore skipped for those, and the load runs unplanned.
    """
    monkeypatch.delenv("VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        safetensors = _download_gpt2(tmpdir)
        names = {
            name
            for name, _ in fastsafetensors_weights_iterator(
                safetensors, True, accumulate_resident=True
            )
        }
    assert names


@pytest.mark.parametrize(
    ("budget", "largest_span", "group_size", "resident", "plan"),
    [
        # Qwen3.6-35B-A3B-AWQ: 2.8 GiB shards, 85 GiB free. Every shard fits
        # whole, so planning would only add header parsing and chunking.
        (85 * GiB, 2.8 * GiB, 1, False, False),
        # Muse-Glimmer-30B: a 46.5 GiB shard against 38 GiB free must chunk.
        (38 * GiB, 46.5 * GiB, 1, False, True),
        # Broadcast adds a receive buffer: 5 x 2.8 fits 16 GiB, 7 x 2.8 does not.
        (16 * GiB, 2.8 * GiB, 1, False, False),
        (16 * GiB, 2.8 * GiB, 2, False, True),
        # Resident growth shrinks headroom by an amount only the planner knows.
        (85 * GiB, 2.8 * GiB, 1, True, True),
    ],
)
def test_fastsafetensors_needs_plan(budget, largest_span, group_size, resident, plan):
    """The planner runs only where it would change the load."""
    assert (
        _fastsafetensors_needs_plan(
            int(budget),
            int(largest_span),
            queue_size=0,
            group_size=group_size,
            accumulate_resident=resident,
        )
        is plan
    )
