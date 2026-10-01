# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 FlashInfer sparse MLA backend."""

import os
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.common.ops import (
    build_flashinfer_mixed_sparse_indices,
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v41.nvidia.ops.o_proj import (
    dsv41_o_proj,
    register_dsv41_o_proj_warmup,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_dspark_swa_index_width,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.ops.dcp import (
    _CORRECT_ATTN_CP_OUT_KERNEL,
    CPTritonContext,
    cp_lse_ag_out_rs,
    dcp_a2a_lse_reduce,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)

# DCP prefill: gather the DCP ranks' compressed pages of each prefill request
# and attend the local heads over them (no q gather / LSE combine).
_DCP_PREFILL_GATHER = os.environ.get("DSV41_DCP_PREFILL_GATHER", "0") == "1"
# DCP combine: "a2a" (one all-to-all of packed output + LSE) or the default
# LSE all-gather + head reduce-scatter (--dcp-comm-backend a2a selects a2a too).
_DCP_COMBINE = os.environ.get("DSV41_DCP_COMBINE", "")


def _sm120_sparse_attention_with_lse(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    workspace: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor | None,
    compressed_kv_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    out: torch.Tensor,
    sm_scale: float,
    sinks: torch.Tensor | None,
    lse: torch.Tensor,
) -> None:
    """FlashInfer's SM120 packed sparse-MLA attention, returning the LSE.

    Mirrors the SM120 branch of ``trtllm_batch_decode_sparse_mla_dsv4``
    (flashinfer 0.7.0.post1, ``flashinfer/mla/_core.py``), whose public entry
    hard-codes ``return_lse=False``. ``query``/``out`` are ``[T, H, 512]``
    bf16, ``lse`` is ``[T, H]`` fp32 and receives the BASE-2 LSE of the
    ``sm_scale``-scaled logits with the sink folded in; a row with no candidate
    and no sink gets ``-1e30`` and a zero output, which the combine treats as
    an empty shard.
    """
    from flashinfer.mla._core import (
        _check_sm120_dsv4_kv_cache_layout,
        _SparseMLASegment,
        _trtllm_batch_decode_sparse_mla_sm120,
    )

    num_tokens = query.shape[0]
    swa_kv_cache = _check_sm120_dsv4_kv_cache_layout(
        swa_kv_cache, "NHD", "swa_kv_cache"
    )
    segments = [
        _SparseMLASegment(indices=swa_indices.reshape(num_tokens, -1), lengths=swa_lens)
    ]
    if extra_indices is not None:
        assert compressed_kv_cache is not None
        compressed_kv_cache = _check_sm120_dsv4_kv_cache_layout(
            compressed_kv_cache, "NHD", "compressed_kv_cache"
        )
        segments.append(
            _SparseMLASegment(
                indices=extra_indices.reshape(num_tokens, -1),
                lengths=extra_lens,
                kv_cache=compressed_kv_cache,
            )
        )
    _trtllm_batch_decode_sparse_mla_sm120(
        query=query.unsqueeze(1),
        kv_cache=swa_kv_cache,
        workspace_buffer=workspace,
        sparse_mla_segments=segments,
        out=out.unsqueeze(1),
        sm_scale=float(sm_scale),
        sinks=sinks,
        lse=lse,
        return_lse=True,
        kv_scale_format="auto",
        kv_cache_format="fp8",
    )


_FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
# FlashInfer's SM120 DSv4 sparse-MLA kernels are instantiated only for pages of
# 64 rows (_DECODE_DSV4_PAGE_BLOCK_SIZE); a page of any other size has no kernel.
_SM120_PAGE_BLOCK_SIZE = 64
_flashinfer_dsv4_workspace_by_device: dict[torch.device, torch.Tensor] = {}


def _get_flashinfer_dsv4_workspace(device: torch.device) -> torch.Tensor:
    workspace = _flashinfer_dsv4_workspace_by_device.get(device)
    if workspace is None:
        workspace = torch.zeros(
            _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        _flashinfer_dsv4_workspace_by_device[device] = workspace
    return workspace


def _packed_block_span(pool: torch.Tensor) -> int:
    """Per-block stride of ``pool`` in tokens (``stride(0)//stride(-2)``): ==
    block_size for unpacked KV, larger when packed (#44577). Raises if not
    token-aligned."""
    block_stride = pool.stride(0)
    token_stride = pool.stride(-2)
    if block_stride % token_stride != 0:
        raise NotImplementedError(
            "FLASHINFER_MLA_SPARSE_DSV4 packed KV requires the per-block stride "
            f"({block_stride}) to be a multiple of the per-token stride "
            f"({token_stride}); this layout is not supported yet."
        )
    return block_stride // token_stride


# Sparse MLA h_q counts accepted natively (flashinfer>=0.6.14, #3545).
_SPARSE_MLA_SUPPORTED_Q_HEADS = (8, 16, 32, 64, 128)


def _pad_to_supported_q_heads(num_heads: int) -> int:
    for supported in _SPARSE_MLA_SUPPORTED_Q_HEADS:
        if num_heads <= supported:
            return supported
    raise ValueError(
        f"DeepseekV4 FlashInfer MLA Sparse does not support {num_heads} heads "
        "(sparse MLA kernel requires h_q in {8, 16, 32, 64, 128})."
    )


def _required_sm120_sparse_topk(vllm_config: VllmConfig, window_size: int) -> int:
    """Return the SM120 DSV4 SWA specialization needed by this model."""
    if not vllm_config.attention_config.use_non_causal:
        return window_size
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return window_size
    return get_dspark_swa_index_width(
        window_size,
        speculative_config.num_speculative_tokens,
    )


class DeepseekV4FlashInferMLASparseBackend(DeepseekV4SparseMLABackend):
    """FlashInfer backend using the DSv4 sparse metadata/cache layout.

    Inherits the base and backend reuses its``DeepseekV4SparseMLAMetadataBuilder``
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        if current_platform.is_cuda() and current_platform.is_device_capability_family(
            120
        ):
            # SM120 pages hold 64 compressed states: 64 tokens at ratio 1, 128
            # at ratio 2. Without a spec, accept the page of either ratio.
            tokens_per_state = getattr(kv_cache_spec, "tokens_per_state", None)
            if isinstance(tokens_per_state, int):
                return [_SM120_PAGE_BLOCK_SIZE * tokens_per_state]
            return [128, _SM120_PAGE_BLOCK_SIZE]
        return [128]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_DSV41"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [10, 12]

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if device_capability.major == 10:
            if kv_cache_dtype == "fp8_ds_mla":
                return (
                    "FLASHINFER_MLA_SPARSE_DSV4 SM10x uses the plain "
                    "per-tensor FP8 KV layout, not fp8_ds_mla"
                )
            if kv_cache_dtype not in (None, "auto", "bfloat16", "fp8", "fp8_e4m3"):
                return "kv_cache_dtype not supported"
            return None
        if device_capability.major == 12:
            if kv_cache_dtype not in ("fp8", "fp8_e4m3", "fp8_ds_mla"):
                return "kv_cache_dtype not supported"
            from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

            if not has_flashinfer_sparse_mla_sm120():
                return (
                    "FLASHINFER_MLA_SPARSE_DSV4 SM120 requires FlashInfer's "
                    "sparse MLA decode API"
                )
            return None
        return "FLASHINFER_MLA_SPARSE_DSV4 requires SM10x or SM12x"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4FlashInferSparseMLAMetadataBuilder"]:
        return DeepseekV4FlashInferSparseMLAMetadataBuilder


class DeepseekV4FlashInferSparseMLAMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    """Varlen-capable metadata builder for the FlashInfer sparse MLA backend."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekSparseSWAFlashInferMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    """SWA metadata for the FlashInfer sparse decode path (varlen decode)."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Graphs retain these addresses while each build refreshes their contents.
        self._decode_topk_lens = torch.empty(
            self._max_tokens, dtype=torch.int32, device=self.device
        )
        self._decode_seq_lens = torch.empty_like(self._decode_topk_lens)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        replay_start: torch.Tensor | None = None,
    ) -> "DeepseekSparseSWAMetadata":
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build, replay_start
        )
        num_tokens = metadata.num_decode_tokens
        if not common_attn_metadata.causal and num_tokens > 0:
            assert metadata.decode_swa_lens is not None
            assert metadata.seq_lens is not None
            assert metadata.token_to_req_indices is not None
            topk_lens = self._decode_topk_lens[:num_tokens]
            seq_lens = self._decode_seq_lens[:num_tokens]
            torch.clamp(metadata.decode_swa_lens, min=self.window_size, out=topk_lens)
            torch.index_select(
                metadata.seq_lens,
                0,
                metadata.token_to_req_indices[:num_tokens],
                out=seq_lens,
            )
            metadata.flashinfer_decode_topk_lens = topk_lens
            metadata.flashinfer_decode_seq_lens = seq_lens
        return metadata


class DeepseekSparseSWAFlashInferBackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekSparseSWAFlashInferMetadataBuilder]:
        return DeepseekSparseSWAFlashInferMetadataBuilder


class DeepseekSparseSWAFlashInferSM120Backend(DeepseekSparseSWAFlashInferBackend):
    """SWA cache on SM12x: FlashInfer's SM120 kernels take only 64-token pages,
    so the page must not be any other multiple of 32."""

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [_SM120_PAGE_BLOCK_SIZE]


class DeepseekV4FlashInferMLAAttention(DeepseekV4Attention):
    """FlashInfer TRTLLM-gen sparse MLA attention layer for SM100 DeepSeek V4."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    swa_backend_cls = DeepseekSparseSWAFlashInferBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = False

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return _pad_to_supported_q_heads(num_heads)

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return dsv41_o_proj(self, attn_out, positions)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe(
            self._o_proj_block_size
        )
        register_dsv41_o_proj_warmup(self)
        # Per-tensor FP8 scale buffers + precomputed scalar BMM scales. Only the
        # per-tensor FP8 cache path consumes these; bf16 reads ``self.scale``.
        if self.kv_cache_torch_dtype != torch.float8_e4m3fn:
            return
        fp8_q_scale = 1.0
        fp8_kv_scale = 1.0
        self.register_buffer(
            "_flashinfer_fp8_q_scale",
            torch.tensor([fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_q_scale_inv",
            torch.tensor([1.0 / fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_kv_scale",
            torch.tensor([fp8_kv_scale], dtype=torch.float32),
            persistent=False,
        )
        # TRTLLM-gen takes scalar scale args on a distinct C++ path vs
        # one-element tensors, so these are Python floats.
        self._flashinfer_fp8_bmm1_scale = self.scale * fp8_q_scale * fp8_kv_scale
        self._flashinfer_fp8_bmm2_scale = fp8_kv_scale

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # The TRTLLM-gen kernel requires h_q in {64, 128}, so the output buffer
        # is allocated at the padded head count while q arrives at the local
        # head count; _forward pads q to match before the launcher.
        assert output.shape[0] == q.shape[0] and output.shape[-1] == q.shape[-1], (
            f"output buffer shape {output.shape} incompatible with q shape {q.shape}"
        )
        assert output.shape[1] >= q.shape[1], (
            f"output heads {output.shape[1]} must be >= q heads {q.shape[1]}"
        )
        # Per-tensor FP8 q produces a bf16 attention output.
        expected_output_dtype = (
            torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        )
        assert output.dtype == expected_output_dtype, (
            f"output dtype {output.dtype} must match expected {expected_output_dtype} "
            f"for q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # Warmup dummy run: FlashInfer reads the cache directly and lazily
            # allocates its workspace, so nothing to reserve here.
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # Compressed-cache metadata lives on the kv-source layer's prefix;
        # consumers share that cache and its block table.
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio == 0
        # SWA-only layers have no compressed KV cache; consumers read the kv
        # source's cache.
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        self._forward(
            q=q,
            kv_cache=self_kv_cache,
            swa_k_cache=swa_kv_cache,
            swa_metadata=swa_metadata,
            attn_metadata=flashmla_metadata,
            swa_only=swa_only,
            output=output,
        )

    def _build_sparse_index_metadata(
        self,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the combined sparse-index tensors for the mixed batch.

        Returns ``(compressed_kv_cache, seq_lens, sparse_indices,
        sparse_topk_lens)``.
        """
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens

        assert swa_metadata.seq_lens is not None
        assert swa_metadata.query_start_loc is not None
        assert swa_metadata.token_to_req_indices is not None
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.block_table is not None
        assert swa_metadata.replay_start is not None

        decode_swa_indices = swa_metadata.decode_swa_indices.reshape(
            num_decode_tokens, swa_metadata.decode_swa_width
        )
        decode_compressed_topk_lens = None
        decode_compressed_indices_are_local = False
        decode_is_valid_token = None

        if swa_only:
            assert self.topk_indices_buffer is not None
            compressed_kv_cache = swa_k_cache
            decode_compressed_indices = None
            prefill_topk_indices = self.topk_indices_buffer[
                num_decode_tokens:num_tokens, :0
            ]
            compressed_block_table = None
            compressed_block_size = swa_metadata.block_size
            top_k = 0
        else:
            assert kv_cache is not None
            assert attn_metadata is not None
            assert self.topk_indices_buffer is not None
            assert swa_metadata.is_valid_token is not None
            compressed_kv_cache = kv_cache
            compressed_block_table = attn_metadata.block_table[:num_reqs]
            compressed_block_size = attn_metadata.block_size // self.compress_ratio

            # Local indices filled by the index-source layer's indexer.
            if num_prefill_tokens > 0:
                prefill_topk_indices = self.topk_indices_buffer[
                    num_decode_tokens:num_tokens
                ]
                top_k = prefill_topk_indices.shape[-1]
            else:
                prefill_topk_indices = self.topk_indices_buffer[:0, :0]
                top_k = 0

            decode_compressed_indices_are_local = True
            decode_is_valid_token = swa_metadata.is_valid_token[:num_decode_tokens]
            if num_decode_tokens > 0:
                decode_compressed_indices = self.topk_indices_buffer[:num_decode_tokens]
            else:
                # Keep the logical width aligned with the mixed-batch case so
                # pure-prefill steps reuse the same Triton specialization.
                decode_compressed_indices = prefill_topk_indices[:0]

        query_start_loc = swa_metadata.query_start_loc[: num_reqs + 1]
        seq_lens = swa_metadata.seq_lens[:num_reqs]
        assert seq_lens.dtype == torch.int32
        # SWA-only layers all build the same mixed sparse indices, so the first
        # one caches them for the step; indexer layers depend on their own topk
        # indices and stay uncached.
        cached_sparse = (
            swa_metadata.flashinfer_sparse_index_cache.get("swa_only")
            if swa_only
            else None
        )
        if cached_sparse is None:
            swa_block_span = _packed_block_span(swa_k_cache)
            compressed_block_span = _packed_block_span(compressed_kv_cache)
            sparse_indices, sparse_topk_lens = build_flashinfer_mixed_sparse_indices(
                decode_swa_indices,
                decode_compressed_indices,
                decode_compressed_topk_lens,
                prefill_topk_indices[:num_prefill_tokens],
                query_start_loc,
                seq_lens,
                swa_metadata.token_to_req_indices[:num_tokens],
                swa_metadata.block_table[:num_reqs],
                swa_metadata.block_size,
                compressed_block_table,
                compressed_block_size,
                self.window_size,
                self.compress_ratio,
                top_k,
                decode_compressed_indices_are_local=decode_compressed_indices_are_local,
                decode_is_valid_token=decode_is_valid_token,
                swa_block_span=swa_block_span,
                compressed_block_span=compressed_block_span,
                replay_start=swa_metadata.replay_start[:num_reqs],
            )
            if swa_only:
                swa_metadata.flashinfer_sparse_index_cache["swa_only"] = (
                    sparse_indices,
                    sparse_topk_lens,
                )
        else:
            sparse_indices, sparse_topk_lens = cached_sparse
        return compressed_kv_cache, seq_lens, sparse_indices, sparse_topk_lens

    def _forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        assert self.kv_cache_torch_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens
        if num_tokens == 0:
            return

        (
            compressed_kv_cache,
            seq_lens,
            sparse_indices,
            sparse_topk_lens,
        ) = self._build_sparse_index_metadata(
            kv_cache=kv_cache,
            swa_k_cache=swa_k_cache,
            swa_metadata=swa_metadata,
            attn_metadata=attn_metadata,
            swa_only=swa_only,
        )

        # CUDA graph execution can pad q/output past the scheduled token count;
        # restrict to the real tokens (the launcher validates sparse indices).
        query = q[:num_tokens]
        output = output[:num_tokens]
        bmm1_scale: float | torch.Tensor = self.scale
        bmm2_scale: float | torch.Tensor = 1.0
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert query.dtype == torch.float8_e4m3fn
            bmm1_scale = self._flashinfer_fp8_bmm1_scale
            bmm2_scale = self._flashinfer_fp8_bmm2_scale
        else:
            assert query.dtype == torch.bfloat16
            query = query.contiguous()

        # The TRTLLM-gen sparse-MLA kernel requires h_q in {64, 128}; zero-pad
        # the query heads to the allocated output head count. Padded heads attend
        # to the shared KV and are sliced off downstream (output is padded too).
        padded_heads = output.shape[1]
        if query.shape[1] < padded_heads:
            padded_query = query.new_zeros(
                (query.shape[0], padded_heads, query.shape[2])
            )
            padded_query[:, : query.shape[1], :] = query
            query = padded_query

        workspace = _get_flashinfer_dsv4_workspace(q.device)
        query_start_loc = swa_metadata.query_start_loc
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc is not None and query_start_loc_cpu is not None

        # Keep the TRTLLM-gen decode/prefill split: the launcher is tuned for
        # uniform-q batches, and this avoids flattening mixed batches into one call.
        if num_decode_tokens > 0:
            decode_query = query[:num_decode_tokens]
            decode_output = output[:num_decode_tokens]
            decode_cu = query_start_loc[: num_decodes + 1]
            decode_seq_lens = seq_lens[:num_decodes]
            decode_topk_lens = sparse_topk_lens[:num_decode_tokens]
            max_decode_query_len = swa_metadata.max_decode_query_len
            if swa_metadata.decode_swa_width > self.window_size:
                # DSpark's non-causal window extends past the fixed 128 SWA
                # columns into the aliased compressed pool. Exclude padding,
                # and expose the full block to each query instead of letting
                # TRTLLM derive a causal SWA length from its query position.
                assert swa_only
                assert swa_metadata.flashinfer_decode_topk_lens is not None
                assert swa_metadata.flashinfer_decode_seq_lens is not None
                decode_topk_lens = swa_metadata.flashinfer_decode_topk_lens
                decode_seq_lens = swa_metadata.flashinfer_decode_seq_lens
                decode_query = decode_query.unsqueeze(1)
                decode_output = decode_output.unsqueeze(1)
                decode_cu = None
                max_decode_query_len = 1
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=decode_query,
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[:num_decode_tokens],
                sparse_indices_are_storage_offsets=True,
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=decode_topk_lens,
                seq_lens=decode_seq_lens,
                out=decode_output,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=decode_cu,
                max_q_len=max_decode_query_len,
            )

        if num_prefill_tokens > 0:
            # The prefill query view re-anchors at offset 0, so rebase the
            # cumulative query offsets to start at 0.
            prefill_cu = (
                query_start_loc[num_decodes : num_reqs + 1]
                - query_start_loc[num_decodes]
            )
            prefill_cu_cpu = query_start_loc_cpu[num_decodes : num_reqs + 1]
            prefill_lens_cpu = prefill_cu_cpu[1:] - prefill_cu_cpu[:-1]
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=query[num_decode_tokens:num_tokens],
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[num_decode_tokens:num_tokens],
                sparse_indices_are_storage_offsets=True,
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens[num_decode_tokens:num_tokens],
                seq_lens=seq_lens[num_decodes:num_reqs],
                out=output[num_decode_tokens:num_tokens],
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=prefill_cu,
                max_q_len=int(prefill_lens_cpu.max().item()),
            )


class DeepseekV4FlashInferSM120Attention(DeepseekV4Attention):
    """DeepSeek V4 sparse MLA attention through FlashInfer's SM120 kernels."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    swa_backend_cls = DeepseekSparseSWAFlashInferSM120Backend
    use_fp8_ds_mla_layout: ClassVar[bool] = True
    kv_page_states: ClassVar[int | None] = _SM120_PAGE_BLOCK_SIZE
    # DCP: per-rank partial attention returning the LSE, then a combine.
    can_return_lse_for_decode: ClassVar[bool] = True
    # Under DCP the per-token q gather / combine has no cross-token dependency:
    # bound the transient [T, heads * dcp, 512] buffers inside a prefill chunk.
    DCP_PREFILL_TOKEN_CHUNK: ClassVar[int] = 2048
    # DCP prefill gather (DSV41_DCP_PREFILL_GATHER), within one step: the
    # gathered compressed KV of the current source cache and the remapped top-k
    # of the current (index source, ratio, page), shared by the layers that use
    # them. Sources only advance with depth, so a superseded entry is never
    # reused and is dropped; the last compressed layer releases the rest.
    _dcp_gather_cache: ClassVar[dict[str, Any]] = {"step": None}

    @staticmethod
    def _get_workspace(device: torch.device) -> torch.Tensor:
        return _get_flashinfer_dsv4_workspace(device)

    @staticmethod
    def _as_sparse_cache(kv_cache: torch.Tensor) -> torch.Tensor:
        if kv_cache.dtype == torch.float8_e4m3fn:
            kv_cache = kv_cache.view(torch.uint8)
        if kv_cache.dim() == 4:
            return kv_cache
        return kv_cache.unsqueeze(-2)

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return _pad_to_supported_q_heads(num_heads)

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return dsv41_o_proj(self, attn_out, positions)

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config

        required_topk = _required_sm120_sparse_topk(vllm_config, self.window_size)
        # DCP: the kernel sees the gathered heads [rank 0 heads | rank 1 heads |
        # ...]; the local slice must be unpadded so the head reduce-scatter
        # hands each rank exactly its own heads back.
        self.n_dcp_heads = self.n_local_heads * self.dcp_world_size
        if self.dcp_world_size > 1:
            if self.padded_heads != self.n_local_heads:
                raise NotImplementedError(
                    "DeepSeek-V4.1 DCP needs an unpadded local head count in "
                    f"{_SPARSE_MLA_SUPPORTED_Q_HEADS}; got {self.n_local_heads} "
                    f"(padded to {self.padded_heads})."
                )
            if self.n_dcp_heads not in _SPARSE_MLA_SUPPORTED_Q_HEADS:
                raise NotImplementedError(
                    f"DeepSeek-V4.1 DCP gathered head count {self.n_dcp_heads} "
                    "is not a supported sparse-MLA h_q "
                    f"{_SPARSE_MLA_SUPPORTED_Q_HEADS}."
                )
        # The local heads run SWA-only layers and the DCP prefill gather; the
        # gathered heads run the DCP combine path.
        for kernel_heads in dict.fromkeys((self.padded_heads, self.n_dcp_heads)):
            if not has_flashinfer_sparse_mla_sm120_config(kernel_heads, required_topk):
                raise RuntimeError(
                    "FLASHINFER_MLA_SPARSE_DSV4 on SM120 requires a FlashInfer "
                    "DSV4 sparse MLA decode specialization for "
                    f"(num_q_heads={kernel_heads}, top_k={required_topk}). "
                    "Install a FlashInfer build containing "
                    "flashinfer-ai/flashinfer#4380."
                )
        # DCP state: the group (a subset of the TP ranks), the gathered sink
        # (built on the first forward, after weight loading) and the neutral
        # SWA index rows of the ranks other than 0.
        self.dcp_group = get_dcp_group() if self.dcp_world_size > 1 else None
        # The last backbone layer with a compressed cache: after its prefill no
        # layer of the step reads the gathered KV again.
        config = vllm_config.model_config.hf_config
        ratios = list(getattr(config, "compress_ratios", None) or ())
        self._is_last_compressed_layer = self.layer_id == max(
            (i for i, r in enumerate(ratios[: config.num_hidden_layers]) if r > 0),
            default=-1,
        )
        self._dcp_sink: torch.Tensor | None = None
        self._dcp_empty_swa: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # One Triton launch context per layer for the LSE-combine kernel.
        self._cp_ctx = CPTritonContext() if self.dcp_world_size > 1 else None
        self._dcp_combine_a2a = (
            _DCP_COMBINE == "a2a"
            or vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        if self.dcp_world_size > 1 and self.compress_ratio > 0:
            logger.info_once(
                "DeepSeek-V4.1 SM120 attention under DCP%d: %d local heads -> %d "
                "gathered, SWA window + sink on DCP rank 0, base-2 LSE %s "
                "combine, prefill %s",
                self.dcp_world_size,
                self.n_local_heads,
                self.n_dcp_heads,
                "a2a" if self._dcp_combine_a2a else "ag_rs",
                "KV gather" if _DCP_PREFILL_GATHER else "q gather + combine",
            )
            if (
                vllm_config.kernel_config.enable_jit_warmup
                and not self._dcp_combine_a2a
            ):
                _CORRECT_ATTN_CP_OUT_KERNEL.register_warmup(
                    vllm_config,
                    output_dtype=torch.bfloat16,
                    num_heads=self.n_dcp_heads,
                    head_dim=self.head_dim,
                    is_base_e=False,
                )
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe(
            self._o_proj_block_size
        )
        # Per-tensor FP8 cache path scales.
        if self.kv_cache_torch_dtype != torch.float8_e4m3fn:
            return
        fp8_q_scale = 1.0
        fp8_kv_scale = 1.0
        self.register_buffer(
            "_flashinfer_fp8_q_scale",
            torch.tensor([fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_q_scale_inv",
            torch.tensor([1.0 / fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_kv_scale",
            torch.tensor([fp8_kv_scale], dtype=torch.float32),
            persistent=False,
        )
        # FlashInfer expects scalar scale arguments for this path.
        self._flashinfer_fp8_bmm1_scale = self.scale * fp8_q_scale * fp8_kv_scale
        self._flashinfer_fp8_bmm2_scale = fp8_kv_scale

    def _reserve_empty_forward_workspace(self) -> None:
        self._get_workspace(
            torch.device("cuda", torch.accelerator.current_device_index())
        )

    def _forward_sparse_impl(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        self_kv_cache: torch.Tensor | None,
        swa_kv_cache: torch.Tensor,
        swa_only: bool,
    ) -> None:
        num_decode_tokens = swa_metadata.num_decode_tokens
        if swa_metadata.num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
            if self._is_last_compressed_layer:
                # Release the DCP prefill gather before the step ends, so it
                # never outlives the step into decode-only steps.
                type(self)._dcp_gather_cache.clear()
        if swa_metadata.num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Output may be padded to backend-supported head counts.
        assert output.shape[0] == q.shape[0] and output.shape[-1] == q.shape[-1], (
            f"output buffer shape {output.shape} incompatible with q shape {q.shape}"
        )
        assert output.shape[1] >= q.shape[1], (
            f"output heads {output.shape[1]} must be >= q heads {q.shape[1]}"
        )
        # Per-tensor FP8 q produces a bf16 attention output.
        expected_output_dtype = (
            torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        )
        assert output.dtype == expected_output_dtype, (
            f"output dtype {output.dtype} must match expected {expected_output_dtype} "
            f"for q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            self._reserve_empty_forward_workspace()
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # Compressed-cache metadata lives on the kv-source layer's prefix;
        # consumers share that cache and its block table.
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio == 0
        # SWA-only layers have no compressed KV cache; consumers read the kv
        # source's cache.
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        self._forward_sparse_impl(
            q=q,
            output=output,
            flashmla_metadata=flashmla_metadata,
            swa_metadata=swa_metadata,
            self_kv_cache=self_kv_cache,
            swa_kv_cache=swa_kv_cache,
            swa_only=swa_only,
        )

    def _prepare_query(self, q: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert q.dtype == torch.float8_e4m3fn
            q = q.to(torch.bfloat16)
        else:
            assert q.dtype == torch.bfloat16
        padded_heads = output.shape[1]
        if q.shape[1] < padded_heads:
            padded_query = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
            padded_query[:, : q.shape[1], :] = q
            q = padded_query
        return q.contiguous()

    # ---- DCP ---------------------------------------------------------------

    def _compressed_topk_to_slots(
        self,
        topk_indices: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        is_valid_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Indexer top-k (compressed ids) -> physical slots + per-row count.

        DCP1: the request-local ids map through the block table. DCP>1: the ids
        are GLOBAL compressed ids (the indexer's cross-rank merge); de-interleave
        them in state coordinates, drop the states other ranks own and compact
        this rank's slots to a prefix (-1 tail, count). ``block_size`` is the
        number of states per page on this rank.
        """
        if self.dcp_world_size == 1:
            return compute_global_topk_indices_and_lens(
                topk_indices,
                token_to_req_indices,
                block_table,
                block_size,
                is_valid_token,
            )
        num_tokens = topk_indices.shape[0]
        slots, lens = triton_filter_and_convert_dcp_index(
            token_to_req_indices[:num_tokens].contiguous(),
            block_table,
            topk_indices,
            dcp_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            BLOCK_SIZE=block_size,
            BLOCK_STRIDE_ROWS=block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            BLOCK_N=128,
            return_valid_counts=True,
        )
        # CUDA-graph padding rows attend nothing (parity with the DCP1 kernel).
        lens.masked_fill_(~is_valid_token[:num_tokens], 0)
        return slots, lens

    def _dcp_prefill_gather(
        self,
        compressed_k_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        token_to_req: torch.Tensor,
        is_valid_token: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata,
        num_decodes: int,
        num_prefills: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """DCP prefill without q gather / LSE combine.

        Gathers the DCP ranks' compressed pages of every prefill request into
        one rank-major paged buffer (request i holds ``W * pages_i`` pages:
        rank 0's, then rank 1's, ...) and remaps the GLOBAL top-k ids into it.
        Returns (gathered cache, slots, per-row counts). The gathered KV is
        shared per source cache and the remap per (index source, ratio, page)
        across the layers of one step: every layer reads the one top-k buffer
        its index source wrote, which the next index source overwrites.
        """
        cache = type(self)._dcp_gather_cache
        step = get_forward_context().attn_metadata
        if cache.get("step") is not step:
            cache.clear()
            cache.update(step=step, kv={}, idx={}, geom={})
        assert self.dcp_group is not None
        world = self.dcp_world_size
        interleave = self.cp_kv_cache_interleave_size
        geom_key = (self.compress_ratio, block_size)
        geom = cache["geom"].get(geom_key)
        if geom is None:
            assert attn_metadata.seq_lens_cpu is not None
            seq_lens = attn_metadata.seq_lens_cpu[
                num_decodes : num_decodes + num_prefills
            ].tolist()
            # States after this step (state = pos // ratio) in virtual blocks of
            # block_size * world states = one page per rank; at least one page so
            # the collective never carries zero bytes.
            pages = [
                max(1, -(-(int(sl) // self.compress_ratio) // (block_size * world)))
                for sl in seq_lens
            ]
            base = [0]
            for p in pages:
                base.append(base[-1] + world * p)
            device = topk_indices.device
            geom = (
                pages,
                torch.tensor(pages, device=device, dtype=torch.int64),
                torch.tensor(base[:-1], device=device, dtype=torch.int64),
            )
            cache["geom"][geom_key] = geom
        pages, pages_t, base_t = geom
        kv_key = compressed_k_cache.data_ptr()
        kv = cache["kv"].get(kv_key)
        if kv is None:
            # Free the previous source's gather before allocating this one.
            cache["kv"].clear()
            pool = self._as_sparse_cache(compressed_k_cache)
            parts = []
            for i, p in enumerate(pages):
                page_ids = attn_metadata.block_table[num_decodes + i, :p].long()
                local = pool.index_select(0, page_ids).contiguous()
                parts.append(self.dcp_group.all_gather(local, dim=0))
            kv = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            cache["kv"][kv_key] = kv
        idx_key = (self.index_source_layer_id, *geom_key)
        idx = cache["idx"].get(idx_key)
        if idx is None:
            cache["idx"].clear()
            req = (token_to_req.long() - num_decodes).clamp_(0, max(len(pages) - 1, 0))
            ids = topk_indices.long()
            valid = ids >= 0
            state = ids.clamp_min(0)
            span = block_size * world
            vblock = state // span
            off = state % span
            owner = (off // interleave) % world
            local_off = (off // (interleave * world)) * interleave + off % interleave
            slot = (
                base_t[req][:, None] + owner * pages_t[req][:, None] + vblock
            ) * block_size + local_off
            slots = torch.where(valid, slot, torch.full_like(slot, -1)).to(torch.int32)
            lens = valid.sum(dim=-1, dtype=torch.int32)
            lens.masked_fill_(~is_valid_token[: lens.shape[0]], 0)
            idx = (slots, lens)
            cache["idx"][idx_key] = idx
        return kv, idx[0], idx[1]

    def _dcp_attn_sink(self) -> torch.Tensor:
        """The sink of the gathered heads, in gather order (collective)."""
        if self._dcp_sink is None:
            assert self.dcp_group is not None
            local = self.attn_sink.data[: self.n_local_heads].contiguous()
            self._dcp_sink = self.dcp_group.all_gather(local, dim=0)
        return self._dcp_sink

    def _dcp_neutral_swa(
        self, num_tokens: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All -1 SWA indices and zero lengths (ranks other than 0)."""
        buf = self._dcp_empty_swa.get(width)
        if buf is None or buf[0].shape[0] < num_tokens:
            rows = max(num_tokens, self.max_num_batched_tokens)
            buf = (
                torch.full((rows, 1, width), -1, dtype=torch.int32, device=device),
                torch.zeros(rows, dtype=torch.int32, device=device),
            )
            self._dcp_empty_swa[width] = buf
        return buf[0][:num_tokens], buf[1][:num_tokens]

    def _run_sm120_dcp(
        self,
        q: torch.Tensor,
        swa_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        extra_cache: torch.Tensor | None,
        extra_indices: torch.Tensor | None,
        extra_lens: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        """One DCP attention step for ``q``/``output`` of ``[T, n_local_heads, 512]``.

        Gathers the local heads to ``[T, n_local_heads * W, 512]``, attends this
        rank's compressed-state shard with the LSE, then combines across the DCP
        group and scatters the heads back. The replicated SWA window and the
        sink are attended on DCP rank 0 only, so with ``L_r`` the per-rank
        base-2 LSEs (sink folded into ``L_0``) the combine ``L = log2(sum_r
        2^L_r)``, ``o = sum_r 2^(L_r - L) o_r`` reproduces the DCP1 softmax.
        """
        assert self.dcp_group is not None
        num_tokens = q.shape[0]
        # Collective: every rank builds the gathered sink; only rank 0 applies
        # it, together with the SWA window.
        sink = self._dcp_attn_sink()
        q_gathered = self.dcp_group.all_gather(q, dim=1)
        num_heads = q_gathered.shape[1]
        out_gathered = torch.empty(
            (num_tokens, num_heads, self.head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
        lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
        if self.dcp_rank != 0:
            swa_indices, swa_lens = self._dcp_neutral_swa(
                num_tokens, swa_indices.shape[-1], q.device
            )
        _sm120_sparse_attention_with_lse(
            q_gathered,
            swa_cache,
            self._get_workspace(q.device),
            swa_indices,
            swa_lens,
            extra_cache,
            extra_indices,
            extra_lens,
            out_gathered,
            self.scale,
            sink if self.dcp_rank == 0 else None,
            lse,
        )
        # Rows with no local candidate (and no sink) carry the kernel's -1e30
        # LSE and a zero output, which the max-subtracted exp2 of either combine
        # turns into zero weight.
        if self._dcp_combine_a2a:
            combined = dcp_a2a_lse_reduce(
                out_gathered, lse, self.dcp_group, is_lse_base_on_e=False
            )
        else:
            combined = cp_lse_ag_out_rs(
                out_gathered,
                lse,
                self.dcp_group,
                ctx=self._cp_ctx,
                is_lse_base_on_e=False,
            )
        output.copy_(combined)

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "Compressed-layer decode requires top-k indices from the indexer."
                )
            # Indices filled by the index-source layer's indexer (request-local
            # at DCP1, global compressed ids under DCP).
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            block_size = attn_metadata.block_size // self.compress_ratio
            global_indices, extra_sparse_lengths = self._compressed_topk_to_slots(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
            )
            extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        if self.dcp_world_size > 1 and not swa_only:
            # SWA-only layers keep the DCP1 path: their cache is replicated and
            # every rank runs its own heads over the whole window.
            self._run_sm120_dcp(
                q,
                swa_cache,
                swa_indices,
                swa_lens,
                extra_cache,
                extra_sparse_indices,
                extra_sparse_lengths,
                output,
            )
            return
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=swa_cache,
            workspace_buffer=self._get_workspace(q.device),
            sparse_indices=swa_indices,
            compressed_kv_cache=extra_cache,
            out=output,
            bmm1_scale=self.scale,
            sinks=self.attn_sink,
            kv_layout="NHD",
            swa_topk_lens=swa_lens,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_lengths,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio == 0

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        extra_sparse_indices: torch.Tensor | None = None
        extra_sparse_lengths: torch.Tensor | None = None
        gathered_kv: torch.Tensor | None = None
        if not swa_only:
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "Compressed-layer prefill requires top-k indices from the indexer."
                )
            if attn_metadata is None:
                raise RuntimeError("Compressed-layer prefill metadata is missing.")
            if swa_metadata.token_to_req_indices is None:
                raise RuntimeError(
                    "Compressed-layer prefill request mapping is missing."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "Compressed-layer prefill validity metadata is missing."
                )
            # Local indices filled by the index-source layer's indexer.
            local_topk_indices = self.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            block_size = attn_metadata.block_size // self.compress_ratio
            if self.dcp_world_size > 1 and _DCP_PREFILL_GATHER:
                assert compressed_k_cache is not None
                gathered_kv, extra_sparse_indices, extra_sparse_lengths = (
                    self._dcp_prefill_gather(
                        compressed_k_cache,
                        local_topk_indices,
                        swa_metadata.token_to_req_indices[prefill_token_slice],
                        swa_metadata.is_valid_token[prefill_token_slice],
                        attn_metadata,
                        num_decodes,
                        num_prefills,
                        block_size,
                    )
                )
            else:
                extra_sparse_indices, extra_sparse_lengths = (
                    self._compressed_topk_to_slots(
                        local_topk_indices,
                        swa_metadata.token_to_req_indices[prefill_token_slice],
                        attn_metadata.block_table,
                        block_size,
                        swa_metadata.is_valid_token[prefill_token_slice],
                    )
                )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None

        q = self._prepare_query(q, output)
        swa_kv_paged = self._as_sparse_cache(swa_k_cache)
        if swa_only:
            extra_kv_paged = None
        else:
            if compressed_k_cache is None:
                raise RuntimeError(
                    "Compressed sparse MLA layers require their compressed KV cache."
                )
            extra_kv_paged = (
                gathered_kv
                if gathered_kv is not None
                else self._as_sparse_cache(compressed_k_cache)
            )

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            extra_sparse_indices_chunk = (
                extra_sparse_indices[query_start:query_end]
                if extra_sparse_indices is not None
                else None
            )
            extra_sparse_lengths_chunk = (
                extra_sparse_lengths[query_start:query_end]
                if extra_sparse_lengths is not None
                else None
            )

            q_chunk = q[query_start:query_end]
            swa_indices_chunk = swa_metadata.prefill_swa_indices[query_start:query_end]
            swa_lens_chunk = swa_metadata.prefill_swa_lens[query_start:query_end]
            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:
                raise RuntimeError(
                    "Compressed sparse MLA prefill requires compressed sparse indices."
                )
            if self.dcp_world_size > 1 and not swa_only and gathered_kv is None:
                # Chunk bounds come from the host query_start_loc (identical on
                # every rank), so the collectives line up across ranks.
                assert extra_sparse_indices is not None
                assert extra_sparse_lengths is not None
                qs, qe = int(query_start), int(query_end)
                for sub_start in range(qs, qe, self.DCP_PREFILL_TOKEN_CHUNK):
                    sub_end = min(sub_start + self.DCP_PREFILL_TOKEN_CHUNK, qe)
                    self._run_sm120_dcp(
                        q[sub_start:sub_end],
                        swa_kv_paged,
                        swa_metadata.prefill_swa_indices[sub_start:sub_end],
                        swa_metadata.prefill_swa_lens[sub_start:sub_end],
                        extra_kv_paged,
                        extra_sparse_indices[sub_start:sub_end],
                        extra_sparse_lengths[sub_start:sub_end],
                        output[sub_start:sub_end],
                    )
                continue
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q_chunk,
                swa_kv_cache=swa_kv_paged,
                workspace_buffer=self._get_workspace(q.device),
                sparse_indices=swa_indices_chunk,
                compressed_kv_cache=extra_kv_paged,
                out=output[query_start:query_end],
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                swa_topk_lens=swa_lens_chunk,
                extra_sparse_indices=extra_sparse_indices_chunk,
                extra_sparse_topk_lens=extra_sparse_lengths_chunk,
            )
