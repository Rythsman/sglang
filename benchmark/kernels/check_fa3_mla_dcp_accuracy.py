from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

from sgl_kernel.flash_attn import flash_attn_with_kvcache


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    seq_len: int
    tp_size: int
    dcp_size: int
    head_dim_ckv: int
    head_dim_kpe: int
    page_size: int
    dtype: torch.dtype
    sm_scale: float
    causal: bool


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _compute_sm_scale() -> float:
    x = 0.1 * math.log(40.0) + 1.0
    return x * x * ((128.0 + 64.0) ** -0.5)


def _build_dcp_token_indices(
    kv_lens_cpu: torch.Tensor, dcp_size: int, rank: int
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Build per-seq token indices for one DCP rank (CPU).

    Returns:
        per_seq_indices: list of length B, each is int64 1D tensor of global token ids.
        local_lens_cpu: [B] int32 CPU tensor.
    """
    if kv_lens_cpu.dtype != torch.int32 or kv_lens_cpu.device.type != "cpu":
        raise ValueError("kv_lens_cpu must be an int32 CPU tensor")
    local_lens = ((kv_lens_cpu - rank - 1) // dcp_size) + 1
    local_lens = torch.clamp(local_lens, min=0).to(torch.int32)

    per_seq_indices: List[torch.Tensor] = []
    offset = 0
    for original_len_i, local_len_i in zip(kv_lens_cpu.tolist(), local_lens.tolist()):
        if local_len_i > 0:
            idx = (
                torch.arange(local_len_i, dtype=torch.int64) * dcp_size
                + rank
                + offset
            )
        else:
            idx = torch.empty((0,), dtype=torch.int64)
        per_seq_indices.append(idx)
        offset += int(original_len_i)

    return per_seq_indices, local_lens


def _make_inputs(
    cfg: BenchmarkConfig, device: torch.device
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create benchmark tensors.

    Shapes:
      - q_nope: [B, H, D_ckv]
      - q_pe: [B, H, D_kpe]
      - ckv_cache: [B*L, 1, D_ckv]
      - kpe_cache: [B*L, 1, D_kpe]
      - cache_seqlens: [B] int32 cuda
      - cu_seqlens_q: [B+1] int32 cuda
      - page_table_global: [B, L] int32 cuda (token indices)
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    assert cfg.page_size == 1, "This script currently assumes page_size=1."

    num_local_heads = 128 // cfg.tp_size * cfg.dcp_size

    cache_seqlens = torch.full(
        (cfg.batch_size,), cfg.seq_len, device=device, dtype=torch.int32
    )
    cu_seqlens_q = torch.arange(
        0, cfg.batch_size + 1, device=device, dtype=torch.int32
    )

    q_nope = torch.randn(
        cfg.batch_size,
        num_local_heads,
        cfg.head_dim_ckv,
        dtype=cfg.dtype,
        device=device,
    )
    q_pe = torch.randn(
        cfg.batch_size,
        num_local_heads,
        cfg.head_dim_kpe,
        dtype=cfg.dtype,
        device=device,
    )

    kv_all = torch.randn(
        cfg.batch_size * cfg.seq_len,
        1,
        cfg.head_dim_ckv + cfg.head_dim_kpe,
        dtype=cfg.dtype,
        device=device,
    )
    ckv_cache = kv_all[..., : cfg.head_dim_ckv].contiguous()
    kpe_cache = kv_all[..., cfg.head_dim_ckv :].contiguous()

    page_table_global = torch.empty(
        (cfg.batch_size, cfg.seq_len), dtype=torch.int32, device=device
    )
    for b in range(cfg.batch_size):
        page_table_global[b] = torch.arange(
            b * cfg.seq_len, (b + 1) * cfg.seq_len, device=device, dtype=torch.int32
        )

    return q_nope, q_pe, ckv_cache, kpe_cache, cache_seqlens, cu_seqlens_q, page_table_global


def _run_fa3_absorbed(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    sm_scale: float,
    causal: bool,
    return_lse: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run FA3 absorbed MLA attention via flash_attn_with_kvcache."""
    k_cache = kpe_cache.view(-1, 1, 1, kpe_cache.shape[-1])
    v_cache = ckv_cache.view(-1, 1, 1, ckv_cache.shape[-1])

    out = flash_attn_with_kvcache(
        q=q_pe,
        k_cache=k_cache,
        v_cache=v_cache,
        qv=q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=torch.nn.functional.pad(
            torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
        ),
        max_seqlen_q=1,
        softmax_scale=sm_scale,
        causal=causal,
        return_softmax_lse=return_lse,
    )

    if return_lse:
        o, lse, *rest = out
        return o, lse
    return out, torch.empty((0,), device=q_nope.device, dtype=torch.float32)


def _merge_dcp_outputs(
    outs: Sequence[torch.Tensor], lses: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Merge DCP partial outputs using per-rank LSE."""
    lse_stack = torch.stack(lses, dim=0)  # [R, B, H]
    o_stack = torch.stack(outs, dim=0)  # [R, B, H, D]
    lse_global = torch.logsumexp(lse_stack, dim=0)  # [B, H]
    weights = torch.exp(lse_stack - lse_global.unsqueeze(0))  # [R, B, H]
    merged = (o_stack * weights.unsqueeze(-1)).sum(dim=0)
    return merged


def _stats(a: torch.Tensor) -> str:
    x = a.detach().float().abs().flatten()
    return (
        f"max={x.max().item():.6e}, mean={x.mean().item():.6e}, p99={torch.quantile(x, 0.99).item():.6e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check FA3 MLA DCP accuracy (simulated)")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--dcp-size", type=int, default=8)
    parser.add_argument("--head-dim-ckv", type=int, default=512)
    parser.add_argument("--head-dim-kpe", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--causal", action="store_true", default=False)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    device = torch.device("cuda:0")
    dtype = _parse_dtype(args.dtype)

    cfg = BenchmarkConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        tp_size=args.tp_size,
        dcp_size=args.dcp_size,
        head_dim_ckv=args.head_dim_ckv,
        head_dim_kpe=args.head_dim_kpe,
        page_size=args.page_size,
        dtype=dtype,
        sm_scale=_compute_sm_scale(),
        causal=args.causal,
    )

    (
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        cache_seqlens,
        cu_seqlens_q,
        page_table_global,
    ) = _make_inputs(cfg, device)

    # Baseline (no DCP): attend to the full sequence.
    torch.cuda.synchronize()
    o_base, _ = _run_fa3_absorbed(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        page_table=page_table_global,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        sm_scale=cfg.sm_scale,
        causal=cfg.causal,
        return_lse=False,
    )
    torch.cuda.synchronize()

    # DCP simulation: run per-rank over a strided subset of tokens, then merge by LSE.
    kv_lens_cpu = torch.full((cfg.batch_size,), cfg.seq_len, dtype=torch.int32, device="cpu")

    outs: List[torch.Tensor] = []
    lses: List[torch.Tensor] = []
    for rank in range(cfg.dcp_size):
        per_seq_indices, local_lens_cpu = _build_dcp_token_indices(
            kv_lens_cpu, cfg.dcp_size, rank
        )
        local_lens = local_lens_cpu.to(device=device, non_blocking=True)
        max_local = int(local_lens_cpu.max().item())

        # Build compacted local caches and page_table.
        local_ckv_chunks: List[torch.Tensor] = []
        local_kpe_chunks: List[torch.Tensor] = []
        page_table_local = torch.zeros(
            (cfg.batch_size, max_local), dtype=torch.int32, device=device
        )
        offset = 0
        for b in range(cfg.batch_size):
            idx = per_seq_indices[b].to(device=device, non_blocking=True)
            if idx.numel() == 0:
                continue
            local_ckv = ckv_cache.index_select(0, idx.to(torch.int64))
            local_kpe = kpe_cache.index_select(0, idx.to(torch.int64))
            local_ckv_chunks.append(local_ckv)
            local_kpe_chunks.append(local_kpe)
            page_table_local[b, : idx.numel()] = torch.arange(
                offset, offset + idx.numel(), device=device, dtype=torch.int32
            )
            offset += idx.numel()

        local_ckv_cache = (
            torch.cat(local_ckv_chunks, dim=0)
            if local_ckv_chunks
            else torch.empty((0, 1, cfg.head_dim_ckv), device=device, dtype=cfg.dtype)
        )
        local_kpe_cache = (
            torch.cat(local_kpe_chunks, dim=0)
            if local_kpe_chunks
            else torch.empty((0, 1, cfg.head_dim_kpe), device=device, dtype=cfg.dtype)
        )

        o_r, lse_r = _run_fa3_absorbed(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=local_ckv_cache,
            kpe_cache=local_kpe_cache,
            page_table=page_table_local,
            cache_seqlens=local_lens,
            cu_seqlens_q=cu_seqlens_q,
            sm_scale=cfg.sm_scale,
            causal=cfg.causal,
            return_lse=True,
        )
        outs.append(o_r)
        lses.append(lse_r)

    o_merged = _merge_dcp_outputs(outs, lses)
    diff = o_merged - o_base

    print(
        "Config: "
        f"dtype={args.dtype}, batch_size={cfg.batch_size}, seq_len={cfg.seq_len}, "
        f"tp_size={cfg.tp_size}, dcp_size={cfg.dcp_size}, "
        f"head_dim_ckv={cfg.head_dim_ckv}, head_dim_kpe={cfg.head_dim_kpe}, "
        f"page_size={cfg.page_size}, causal={cfg.causal}"
    )
    print(f"abs(diff): {_stats(diff)}")
    print(f"abs(base): {_stats(o_base)}")


if __name__ == "__main__":
    main()

