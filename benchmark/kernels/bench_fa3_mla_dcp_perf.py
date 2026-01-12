from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

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


def _make_inputs(cfg: BenchmarkConfig, device: torch.device):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    assert cfg.page_size == 1, "This benchmark currently assumes page_size=1."

    num_local_heads = 128 // cfg.tp_size * cfg.dcp_size

    cu_seqlens_q = torch.arange(
        0, cfg.batch_size + 1, device=device, dtype=torch.int32
    )
    kv_lens_cpu = torch.full((cfg.batch_size,), cfg.seq_len, dtype=torch.int32, device="cpu")

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
    ckv_cache_global = kv_all[..., : cfg.head_dim_ckv].contiguous()
    kpe_cache_global = kv_all[..., cfg.head_dim_ckv :].contiguous()

    return cu_seqlens_q, kv_lens_cpu, q_nope, q_pe, ckv_cache_global, kpe_cache_global


def _build_local_rank_inputs(
    cfg: BenchmarkConfig,
    device: torch.device,
    kv_lens_cpu: torch.Tensor,
    ckv_cache_global: torch.Tensor,
    kpe_cache_global: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_seq_indices, local_lens_cpu = _build_dcp_token_indices(
        kv_lens_cpu, cfg.dcp_size, rank
    )
    local_lens = local_lens_cpu.to(device=device, non_blocking=True)
    max_local = int(local_lens_cpu.max().item())

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
        local_ckv_chunks.append(ckv_cache_global.index_select(0, idx))
        local_kpe_chunks.append(kpe_cache_global.index_select(0, idx))
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
    return local_lens, local_ckv_cache, local_kpe_cache, page_table_local


def _make_run_once(
    cfg: BenchmarkConfig,
    device: torch.device,
    cu_seqlens_q: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache_global: torch.Tensor,
    kpe_cache_global: torch.Tensor,
    return_lse: bool,
) -> Callable[[], None]:
    # Pre-build per-rank compact caches and page tables.
    rank_inputs = []
    for rank in range(cfg.dcp_size):
        local_lens, local_ckv, local_kpe, page_table_local = _build_local_rank_inputs(
            cfg,
            device,
            kv_lens_cpu,
            ckv_cache_global,
            kpe_cache_global,
            rank,
        )
        rank_inputs.append((local_lens, local_ckv, local_kpe, page_table_local))

    def _run_once() -> None:
        for local_lens, local_ckv, local_kpe, page_table_local in rank_inputs:
            k_cache = local_kpe.view(-1, 1, 1, local_kpe.shape[-1])
            v_cache = local_ckv.view(-1, 1, 1, local_ckv.shape[-1])
            _ = flash_attn_with_kvcache(
                q=q_pe,
                k_cache=k_cache,
                v_cache=v_cache,
                qv=q_nope,
                page_table=page_table_local,
                cache_seqlens=local_lens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=torch.nn.functional.pad(
                    torch.cumsum(local_lens, dim=0, dtype=torch.int32), (1, 0)
                ),
                max_seqlen_q=1,
                softmax_scale=cfg.sm_scale,
                causal=cfg.causal,
                return_softmax_lse=return_lse,
            )

    return _run_once


def _bench_gpu_time_ms(
    fn: Callable[[], None],
    warmup_ms: int,
    repeat_ms: int,
) -> List[float]:
    # Warmup for a target duration.
    torch.cuda.synchronize()
    t0 = time.time()
    while (time.time() - t0) * 1000.0 < warmup_ms:
        fn()
    torch.cuda.synchronize()

    times_ms: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    t0 = time.time()
    while (time.time() - t0) * 1000.0 < repeat_ms:
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    return times_ms


def _stats_ms(ms_list: Sequence[float]) -> str:
    arr = np.asarray(ms_list, dtype=np.float64)
    return (
        f"p50={np.percentile(arr, 50):.4f} ms, "
        f"p10={np.percentile(arr, 10):.4f} ms, "
        f"p90={np.percentile(arr, 90):.4f} ms, "
        f"n={arr.size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FA3 MLA DCP (simulated) GPU time")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--dcp-size", type=int, default=8)
    parser.add_argument("--head-dim-ckv", type=int, default=512)
    parser.add_argument("--head-dim-kpe", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--return-lse", dest="return_lse", action="store_true")
    parser.add_argument("--no-return-lse", dest="return_lse", action="store_false", default=True)
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--repeat-ms", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

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
        cu_seqlens_q,
        kv_lens_cpu,
        q_nope,
        q_pe,
        ckv_cache_global,
        kpe_cache_global,
    ) = _make_inputs(cfg, device)

    run_once = _make_run_once(
        cfg,
        device,
        cu_seqlens_q,
        kv_lens_cpu,
        q_nope,
        q_pe,
        ckv_cache_global,
        kpe_cache_global,
        return_lse=args.return_lse,
    )

    # One extra sync warmup.
    torch.cuda.synchronize()
    run_once()
    torch.cuda.synchronize()

    times_ms = _bench_gpu_time_ms(run_once, warmup_ms=args.warmup_ms, repeat_ms=args.repeat_ms)

    num_local_heads = 128 // cfg.tp_size * cfg.dcp_size
    print(
        "Config: "
        f"dtype={args.dtype}, batch_size={cfg.batch_size}, seq_len={cfg.seq_len}, "
        f"tp_size={cfg.tp_size}, dcp_size={cfg.dcp_size}, num_local_heads={num_local_heads}, "
        f"head_dim_ckv={cfg.head_dim_ckv}, head_dim_kpe={cfg.head_dim_kpe}, "
        f"page_size={cfg.page_size}, return_lse={args.return_lse}, causal={cfg.causal}"
    )
    print(_stats_ms(times_ms))


if __name__ == "__main__":
    main()

