from __future__ import annotations

import math
from typing import List, Tuple

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from sglang.test.test_utils import CustomTestCase


def _compute_sm_scale() -> float:
    """Match the scale formula used by existing MLA benchmarks."""
    x = 0.1 * math.log(40.0) + 1.0
    return x * x * ((512.0 + 64.0) ** -0.5)


def _maybe_import_fa3_kernel():
    try:
        from sgl_kernel.flash_attn import (
            flash_attn_with_kvcache,  # pylint: disable=import-outside-toplevel
        )

        return flash_attn_with_kvcache
    except Exception:  # noqa: BLE001
        return None


def _build_dcp_local_lens_and_indices(
    kv_lens_cpu: torch.Tensor, dcp_size: int, rank: int
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Build per-seq DCP indices (CPU) using the same formula as flashinfer_mla."""
    local_lens = ((kv_lens_cpu - rank - 1) // dcp_size) + 1
    local_lens = torch.clamp(local_lens, min=0).to(torch.int32)

    indices: List[torch.Tensor] = []
    offset = 0
    for original_len_i, local_len_i in zip(kv_lens_cpu.tolist(), local_lens.tolist()):
        if local_len_i > 0:
            idx = (
                torch.arange(local_len_i, dtype=torch.int64) * dcp_size + rank + offset
            )
        else:
            idx = torch.empty((0,), dtype=torch.int64)
        indices.append(idx)
        offset += int(original_len_i)
    return indices, local_lens


def _as_bh_lse(lse: torch.Tensor, batch_size: int, num_heads: int) -> torch.Tensor:
    """Normalize LSE to shape [B, H]."""
    if lse.dim() != 2:
        raise ValueError(f"Expected lse to be 2D, got {lse.shape=}")
    if lse.shape == (batch_size, num_heads):
        return lse
    if lse.shape == (num_heads, batch_size):
        return lse.T.contiguous()
    raise ValueError(f"Unexpected lse shape: {lse.shape}, expected (B,H) or (H,B)")


class TestFA3MLADcpAccuracy(CustomTestCase):
    def test_fa3_mla_dcp_merge_matches_full(self):
        if torch is None:
            self.skipTest("PyTorch is required for this test.")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this test.")

        flash_attn_with_kvcache = _maybe_import_fa3_kernel()
        if flash_attn_with_kvcache is None:
            self.skipTest("FA3 kernel (sgl_kernel.flash_attn) is not available.")

        device = torch.device("cuda:0")
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Keep this test small for CI.
        batch_size = 1
        seq_len = 8192
        tp_size = 32
        dcp_size = 8
        head_dim_ckv = 512
        head_dim_kpe = 64
        page_size = 1

        # Initial bring-up scope: bf16 only.
        dtype = torch.bfloat16

        num_heads = 128 // tp_size * dcp_size
        sm_scale = _compute_sm_scale()

        # Query is one token per sequence (decode-like).
        q_nope = torch.randn(
            (batch_size, num_heads, head_dim_ckv), device=device, dtype=dtype
        )
        q_pe = torch.randn(
            (batch_size, num_heads, head_dim_kpe), device=device, dtype=dtype
        )

        kv_all = torch.randn(
            (batch_size * seq_len, 1, head_dim_ckv + head_dim_kpe),
            device=device,
            dtype=dtype,
        )
        ckv_cache_global = kv_all[..., :head_dim_ckv].contiguous()
        kpe_cache_global = kv_all[..., head_dim_ckv:].contiguous()

        # page_table indexes tokens when page_size == 1.
        page_table_global = torch.stack(
            [
                torch.arange(
                    b * seq_len, (b + 1) * seq_len, device=device, dtype=torch.int32
                )
                for b in range(batch_size)
            ],
            dim=0,
        )
        cache_seqlens_global = torch.full(
            (batch_size,), seq_len, device=device, dtype=torch.int32
        )
        cu_seqlens_q = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        cu_seqlens_k_global = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens_global, dim=0, dtype=torch.int32), (1, 0)
        )

        # Baseline: full attention.
        torch.cuda.synchronize()
        o_base = flash_attn_with_kvcache(
            q=q_pe,
            k_cache=kpe_cache_global.view(-1, page_size, 1, head_dim_kpe),
            v_cache=ckv_cache_global.view(-1, page_size, 1, head_dim_ckv),
            qv=q_nope,
            page_table=page_table_global,
            cache_seqlens=cache_seqlens_global,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_global,
            max_seqlen_q=1,
            softmax_scale=sm_scale,
            causal=True,
            return_softmax_lse=False,
        )
        torch.cuda.synchronize()

        # DCP simulation: rank-local attention + LSE merge.
        kv_lens_cpu = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device="cpu"
        )
        outs: List[torch.Tensor] = []
        lses: List[torch.Tensor] = []

        for rank in range(dcp_size):
            per_seq_indices, local_lens_cpu = _build_dcp_local_lens_and_indices(
                kv_lens_cpu, dcp_size, rank
            )
            # print(f"{rank=}, {per_seq_indices=}, {local_lens_cpu=}", flush=True)
            local_lens = local_lens_cpu.to(device=device, non_blocking=True)
            max_local = int(local_lens_cpu.max().item())

            # Build compacted KV caches and local page table.
            local_ckv_chunks: List[torch.Tensor] = []
            local_kpe_chunks: List[torch.Tensor] = []
            page_table_local = torch.zeros(
                (batch_size, max_local), device=device, dtype=torch.int32
            )
            offset = 0
            for b in range(batch_size):
                idx = per_seq_indices[b].to(device=device, non_blocking=True)
                if idx.numel() == 0:
                    continue
                local_ckv_chunks.append(ckv_cache_global.index_select(0, idx))
                local_kpe_chunks.append(kpe_cache_global.index_select(0, idx))
                page_table_local[b, : idx.numel()] = torch.arange(
                    offset, offset + idx.numel(), device=device, dtype=torch.int32
                )
                offset += idx.numel()

            local_ckv_cache = (
                torch.cat(local_ckv_chunks, dim=0)
                if local_ckv_chunks
                else torch.empty((0, 1, head_dim_ckv), device=device, dtype=dtype)
            )
            local_kpe_cache = (
                torch.cat(local_kpe_chunks, dim=0)
                if local_kpe_chunks
                else torch.empty((0, 1, head_dim_kpe), device=device, dtype=dtype)
            )

            cu_seqlens_k_local = torch.nn.functional.pad(
                torch.cumsum(local_lens, dim=0, dtype=torch.int32), (1, 0)
            )

            o_r, lse_r, *rest = flash_attn_with_kvcache(
                q=q_pe,
                k_cache=local_kpe_cache.view(-1, page_size, 1, head_dim_kpe),
                v_cache=local_ckv_cache.view(-1, page_size, 1, head_dim_ckv),
                qv=q_nope,
                page_table=page_table_local,
                cache_seqlens=local_lens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k_local,
                max_seqlen_q=1,
                softmax_scale=sm_scale,
                causal=True,
                return_softmax_lse=True,
            )

            outs.append(o_r)
            lses.append(_as_bh_lse(lse_r, batch_size, num_heads))

        lse_stack = torch.stack(lses, dim=0)  # [R, B, H]
        o_stack = torch.stack(outs, dim=0)  # [R, B, H, D]
        lse_global = torch.logsumexp(lse_stack, dim=0)  # [B, H]
        weights = torch.exp(lse_stack - lse_global.unsqueeze(0))  # [R, B, H]
        o_merged = (o_stack * weights.unsqueeze(-1)).sum(dim=0)

        # bf16 tolerance: keep it a bit loose, but should be stable.
        diff = (o_merged - o_base).float()
        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()
        self.assertLess(max_abs, 1e-3, msg=f"{max_abs=}, {mean_abs=}")
