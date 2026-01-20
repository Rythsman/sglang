from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.distributed import get_dcp_rank, get_dcp_world_size
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.utils import compute_dcp_local_seq_lens
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class FlashAttentionMLADcpBackend(FlashAttentionBackend):
    """FA3 backend for MLA decode under DCP.

    Scope (intentionally minimal):
    - DCP + MLA
    - BF16 attention compute
    - Decode-only (no extend/prefill/verify/spec).

    Unsupported paths are guarded by asserts to avoid accidental usage.
    """

    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner, fa_impl_ver=3, **kwargs)

        # DCP must be initialized if we instantiate this backend.
        self._dcp_world_size = get_dcp_world_size()
        self._dcp_rank = get_dcp_rank()

        # Safety scope for initial bring-up.
        assert self.use_mla, "FlashAttentionMLADcpBackend requires MLA models."
        assert self._dcp_world_size > 1, "FlashAttentionMLADcpBackend requires DCP."

        # Initial plan: only bf16 attention compute.
        assert (
            model_runner.dtype == torch.bfloat16
        ), f"Only bf16 is supported for DCP+MLA FA3, got {model_runner.dtype=}"
        assert (
            model_runner.kv_cache_dtype == torch.bfloat16
        ), f"Only bf16 KV cache is supported for DCP+MLA FA3, got {model_runner.kv_cache_dtype=}"
        assert (
            model_runner.server_args.kv_cache_dtype == "auto"
        ), "FP8 KV cache is not supported for DCP+MLA FA3 yet."

        # Keep page_size=1 for the first stage to match DCP token sharding semantics.
        assert (
            self.page_size == 1
        ), f"Only page_size=1 is supported for DCP+MLA FA3 currently, got {self.page_size=}"

        # Disable features not covered by the initial DCP bring-up.
        assert (
            self.topk <= 1
        ), "Speculative decoding is not supported for DCP+MLA FA3 yet."
        assert (
            not self.has_local_attention
        ), "Local attention is not supported for DCP+MLA FA3 yet."
        assert not self.has_swa, "SWA is not supported for DCP+MLA FA3 yet."

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Decode-only for the first stage.
        assert forward_batch.forward_mode.is_decode_or_idle(), (
            "DCP+MLA FA3 currently supports decode-only. "
            "Use --prefill-attention-backend flashinfer for prefill/extend."
        )
        assert (
            forward_batch.spec_info is None
        ), "Speculative decoding is not supported for DCP+MLA FA3 yet."

        super().init_forward_metadata(forward_batch)

        # Localize metadata for the current DCP rank.
        metadata = self.forward_metadata
        assert metadata is not None
        assert metadata.page_table is not None
        assert metadata.cache_seqlens_int32 is not None

        # Only bf16 compute (extra safety at runtime).
        assert (
            forward_batch.input_ids.device.type == "cuda"
        ), "DCP+MLA FA3 requires CUDA."

        seq_lens = metadata.cache_seqlens_int32
        local_lens = compute_dcp_local_seq_lens(
            seq_lens, self._dcp_rank, self._dcp_world_size
        ).to(device=seq_lens.device)

        # Build local page_table by filtering global token locations.
        global_page_table = metadata.page_table
        bs = global_page_table.shape[0]
        max_k = global_page_table.shape[1]

        max_local_k = int(local_lens.max().item()) if bs > 0 else 0
        local_page_table = torch.zeros(
            (bs, max_local_k),
            dtype=torch.int32,
            device=global_page_table.device,
        )

        # NOTE: keep order stable; DCP token sharding is by (idx % world_size).
        for i in range(bs):
            seq_len_i = int(seq_lens[i].item())
            if seq_len_i <= 0:
                continue
            row = global_page_table[i, : min(seq_len_i, max_k)].to(torch.int64)
            mask = (row % self._dcp_world_size) == self._dcp_rank
            row_local = (row[mask] // self._dcp_world_size).to(torch.int32)
            n = min(row_local.numel(), max_local_k)
            if n > 0:
                local_page_table[i, :n] = row_local[:n]

        metadata.page_table = local_page_table
        metadata.cache_seqlens_int32 = local_lens
        metadata.max_seq_len_k = max_local_k
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(local_lens, dim=0, dtype=torch.int32), (1, 0)
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
    ):
        """Initialize forward metadata for capturing CUDA graph.

        Note: CUDA graph capture only targets decode for this backend.
        """
        assert (
            forward_mode.is_decode_or_idle()
        ), "DCP+MLA FA3 CUDA graph capture supports decode-only."
        assert encoder_lens is None, "Encoder-decoder is not supported for DCP+MLA FA3."
        assert spec_info is None, "Speculative decoding is not supported for DCP+MLA FA3."

        super().init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )

        # Localize metadata for the current DCP rank. Keep the same tensor objects
        # (storage pointers) so the captured graph can be safely replayed.
        metadata = self.forward_metadata
        assert metadata is not None
        assert metadata.cache_seqlens_int32 is not None
        assert metadata.cu_seqlens_k is not None
        assert metadata.page_table is not None

        self._localize_decode_metadata_inplace_for_dcp(
            metadata=metadata,
            bs=bs,
            req_pool_indices=req_pool_indices[:bs],
            seq_lens=seq_lens[:bs],
            seq_lens_cpu=None,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        assert (
            forward_mode.is_decode_or_idle()
        ), "DCP+MLA FA3 CUDA graph replay supports decode-only."
        assert encoder_lens is None, "Encoder-decoder is not supported for DCP+MLA FA3."
        assert spec_info is None, "Speculative decoding is not supported for DCP+MLA FA3."
        assert seq_lens_cpu is not None, "seq_lens_cpu is required for CUDA graph replay."

        # Reuse the metadata object created during capture for this batch size.
        metadata = self.decode_cuda_graph_metadata[bs]
        self.forward_metadata = metadata

        self._localize_decode_metadata_inplace_for_dcp(
            metadata=metadata,
            bs=bs,
            req_pool_indices=req_pool_indices[:bs],
            seq_lens=seq_lens[:bs],
            seq_lens_cpu=seq_lens_cpu[:bs],
        )

    def _localize_decode_metadata_inplace_for_dcp(
        self,
        metadata,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> None:
        """Update decode metadata in-place for one DCP rank.

        This method is used for both CUDA graph capture and replay. It keeps the
        same output tensors (page table, cache_seqlens, cu_seqlens_k) and only
        updates their contents.

        Args:
            metadata: FlashAttentionMetadata.
            bs: Batch size.
            req_pool_indices: Shape [bs], CUDA int tensor.
            seq_lens: Shape [bs], CUDA int tensor.
            seq_lens_cpu: Optional CPU view of seq_lens (used to compute max length
                without a device sync). If None, will fall back to a device query.
        """
        if bs <= 0:
            metadata.max_seq_len_k = 0
            return

        # (1) Compute local lengths and cu_seqlens_k.
        local_lens = compute_dcp_local_seq_lens(
            seq_lens, self._dcp_rank, self._dcp_world_size
        )
        metadata.cache_seqlens_int32.copy_(local_lens)
        metadata.cu_seqlens_k[0] = 0
        metadata.cu_seqlens_k[1:].copy_(
            torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
        )

        max_local_k = int(local_lens.max().item())
        metadata.max_seq_len_k = max_local_k

        # (2) Gather global token indices (page_size == 1) without mutating
        # cache_seqlens / cu_seqlens_k.
        max_len = (
            int(seq_lens_cpu.max().item())
            if seq_lens_cpu is not None
            else int(seq_lens.max().item())
        )
        max_len = max(0, max_len)
        if max_len == 0 or max_local_k == 0:
            # Keep page_table contents unspecified; kernels won't read out of range.
            return

        strided_indices = self.decode_cuda_graph_metadata["strided_indices"][:max_len]
        global_tokens = self.req_to_token[
            req_pool_indices[:, None],
            strided_indices[None, :],
        ]

        # (3) Filter and localize tokens: keep idx % world == rank, map to idx // world,
        # and compact each row to the left into metadata.page_table.
        # Make sure we compute positions on int64 to avoid overflow on long contexts.
        tokens_i64 = global_tokens.to(torch.int64)
        mask = (tokens_i64 % self._dcp_world_size) == self._dcp_rank
        pos = mask.to(torch.int32).cumsum(dim=1) - 1  # valid only where mask==True
        local_tokens = (tokens_i64 // self._dcp_world_size).to(torch.int32)

        # Clear only the prefix that the kernel may read.
        metadata.page_table[:, :max_local_k].zero_()

        # Scatter compacted tokens.
        b_idx = torch.arange(bs, device=global_tokens.device, dtype=torch.int64).view(
            -1, 1
        )
        b_idx = b_idx.expand_as(pos)
        metadata.page_table[b_idx[mask], pos[mask].to(torch.int64)] = local_tokens[mask]

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # DCP decode path must return LSE for cross-rank merge.
        assert (
            forward_batch.forward_mode.is_decode()
        ), "DCP+MLA FA3 only supports ForwardMode.DECODE here."
        assert self.use_mla, "Expected MLA model."
        assert self._dcp_world_size > 1, "Expected DCP to be enabled."
        assert (
            q.dtype == torch.bfloat16
        ), f"Only bf16 attention compute is supported, got {q.dtype=}"
        assert (
            sinks is None
        ), "Attention sinks are not supported for DCP+MLA FA3 yet."

        # Save KV cache (copied from FlashAttentionBackend, restricted scope).
        if k is not None:
            assert v is not None
            if save_kv_cache:
                assert (
                    not layer.is_cross_attention
                ), "Cross attention is not supported for DCP+MLA FA3 yet."
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    k_rope,
                )

        # Recompute the absorbed-MLA attention with return_softmax_lse=True.
        # This avoids invasive changes in FlashAttentionBackend and keeps the
        # DCP-specific behavior isolated.
        metadata = self.forward_metadata
        assert metadata is not None

        # Only support the absorbed MLA path (non-chunked, non-verify).
        assert (
            forward_batch.attn_attend_prefix_cache is None
        ), "Chunked prefix cache is not supported for DCP+MLA FA3 yet."
        assert (
            not forward_batch.forward_mode.is_target_verify()
        ), "Target verify is not supported for DCP+MLA FA3 yet."
        assert (
            forward_batch.spec_info is None
        ), "Speculative decoding is not supported for DCP+MLA FA3 yet."

        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
            q.dtype
        )
        k_rope_cache = kv_cache[:, :, layer.v_head_dim :].view(
            -1, 1, layer.tp_k_head_num, layer.head_dim - layer.v_head_dim
        )
        c_kv_cache = kv_cache[:, :, : layer.v_head_dim].view(
            -1, 1, layer.tp_v_head_num, layer.v_head_dim
        )

        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]

        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        result = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope_cache,
            v_cache=c_kv_cache,
            qv=q_nope,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k_new=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seq_len_q,
            softmax_scale=layer.scaling,
            causal=True,
            softcap=layer.logit_cap,
            return_softmax_lse=True,
            num_splits=self.num_splits,
        )

        o, lse, *rest = result
        return o, lse

