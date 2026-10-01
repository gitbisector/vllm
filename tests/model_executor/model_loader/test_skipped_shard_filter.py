# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for dropping shards whose every tensor the model skips."""

import json
import os
import tempfile

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.model_executor.model_loader.weight_utils import (
    filter_skipped_safetensors_files,
)


def _skip_non_mtp(name: str) -> bool:
    return not name.startswith("mtp.")


def _write_index(folder: str, weight_map: dict[str, str]) -> None:
    with open(os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def _shards(folder: str, n: int) -> list[str]:
    return [
        os.path.join(folder, f"model-{i:05d}-of-{n:05d}.safetensors") for i in range(n)
    ]


def test_keeps_only_shards_holding_unskipped_tensors():
    with tempfile.TemporaryDirectory() as folder:
        files = _shards(folder, 4)
        _write_index(
            folder,
            {
                "layers.0.attn.wq_b.weight": os.path.basename(files[0]),
                "layers.1.attn.wq_b.weight": os.path.basename(files[1]),
                # A shard shared by target and draft tensors is kept.
                "layers.2.attn.wq_b.weight": os.path.basename(files[2]),
                "mtp.0.attn.wq_b.weight": os.path.basename(files[2]),
                "mtp.1.attn.wq_b.weight": os.path.basename(files[3]),
            },
        )
        kept = filter_skipped_safetensors_files(
            files, folder, SAFE_WEIGHTS_INDEX_NAME, _skip_non_mtp
        )
        assert kept == files[2:]


def test_keeps_files_the_index_does_not_list():
    with tempfile.TemporaryDirectory() as folder:
        files = _shards(folder, 2)
        _write_index(folder, {"layers.0.w": os.path.basename(files[0])})
        kept = filter_skipped_safetensors_files(
            files, folder, SAFE_WEIGHTS_INDEX_NAME, _skip_non_mtp
        )
        assert kept == files[1:]


def test_reads_everything_when_nothing_would_be_kept():
    with tempfile.TemporaryDirectory() as folder:
        files = _shards(folder, 2)
        _write_index(
            folder,
            {
                "layers.0.w": os.path.basename(files[0]),
                "layers.1.w": os.path.basename(files[1]),
            },
        )
        kept = filter_skipped_safetensors_files(
            files, folder, SAFE_WEIGHTS_INDEX_NAME, _skip_non_mtp
        )
        assert kept == files


def test_without_index_is_a_no_op():
    with tempfile.TemporaryDirectory() as folder:
        files = _shards(folder, 2)
        kept = filter_skipped_safetensors_files(
            files, folder, SAFE_WEIGHTS_INDEX_NAME, _skip_non_mtp
        )
        assert kept == files
