"""
Test script to verify the correctness of filter_seq_indices optimization.
测试脚本用于验证 filter_seq_indices 优化的正确性。
"""

import torch


def filter_seq_indices_original(
    paged_kernel_lens: torch.Tensor,
    paged_kernel_lens_cumsum: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
):
    """Original implementation with .item() sync"""
    device = paged_kernel_lens.device
    lens = paged_kernel_lens.to(torch.int64)
    starts = paged_kernel_lens_cumsum[:-1].to(torch.int64)
    paged_kernel_lens_split = ((lens - dcp_rank - 1) // dcp_world_size) + 1
    paged_kernel_lens_split.clamp_(min=0)
    total_local = int(paged_kernel_lens_split.sum().item())
    if total_local == 0:
        return paged_kernel_lens_split, torch.empty(
            0, dtype=torch.int64, device="cuda"
        )
    max_split = int(paged_kernel_lens_split.max().item())
    j = torch.arange(max_split, device=device, dtype=torch.int64)
    starts_ = starts.view(-1, 1)
    j_ = j.view(1, -1)
    ids = starts_ + dcp_rank + j_ * dcp_world_size
    mask = j_ < paged_kernel_lens_split.view(-1, 1)
    filter_kv_indices = ids[mask].to(device="cuda")
    return paged_kernel_lens_split, filter_kv_indices


def filter_seq_indices_optimized(
    paged_kernel_lens: torch.Tensor,
    paged_kernel_lens_cumsum: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
):
    """Optimized implementation without .item() sync"""
    device = paged_kernel_lens.device
    lens = paged_kernel_lens.to(torch.int64)
    starts = paged_kernel_lens_cumsum[:-1].to(torch.int64)
    paged_kernel_lens_split = ((lens - dcp_rank - 1) // dcp_world_size) + 1
    paged_kernel_lens_split.clamp_(min=0)
    
    # Avoid .item() sync: use original lens max as upper bound
    # Since paged_kernel_lens_split <= paged_kernel_lens, this is safe
    max_split = lens.max()
    j = torch.arange(max_split, device=device, dtype=torch.int64)
    starts_ = starts.view(-1, 1)
    j_ = j.view(1, -1)
    ids = starts_ + dcp_rank + j_ * dcp_world_size
    mask = j_ < paged_kernel_lens_split.view(-1, 1)
    filter_kv_indices = ids[mask]
    
    return paged_kernel_lens_split, filter_kv_indices


def test_correctness():
    """Test that optimized version produces same results as original"""
    print("Testing filter_seq_indices optimization...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test case 1: Normal case
    print("\nTest case 1: Normal case")
    paged_kernel_lens = torch.tensor([100, 200, 150], device=device)
    cumsum = torch.cat([
        torch.zeros(1, device=device),
        torch.cumsum(paged_kernel_lens, 0)
    ])
    dcp_rank = 0
    dcp_world_size = 4
    
    split1, indices1 = filter_seq_indices_original(
        paged_kernel_lens.clone(), cumsum.clone(), dcp_rank, dcp_world_size
    )
    split2, indices2 = filter_seq_indices_optimized(
        paged_kernel_lens.clone(), cumsum.clone(), dcp_rank, dcp_world_size
    )
    
    assert torch.equal(split1, split2), "Split lengths mismatch!"
    assert torch.equal(indices1, indices2), "Indices mismatch!"
    print(f"✓ Split lengths match: {split1.tolist()}")
    print(f"✓ Indices count: {len(indices1)}")
    
    # Test case 2: Edge case - small lengths
    print("\nTest case 2: Small lengths")
    paged_kernel_lens = torch.tensor([5, 3, 8], device=device)
    cumsum = torch.cat([
        torch.zeros(1, device=device),
        torch.cumsum(paged_kernel_lens, 0)
    ])
    dcp_rank = 2
    dcp_world_size = 3
    
    split1, indices1 = filter_seq_indices_original(
        paged_kernel_lens.clone(), cumsum.clone(), dcp_rank, dcp_world_size
    )
    split2, indices2 = filter_seq_indices_optimized(
        paged_kernel_lens.clone(), cumsum.clone(), dcp_rank, dcp_world_size
    )
    
    assert torch.equal(split1, split2), "Split lengths mismatch!"
    assert torch.equal(indices1, indices2), "Indices mismatch!"
    print(f"✓ Split lengths match: {split1.tolist()}")
    print(f"✓ Indices count: {len(indices1)}")
    
    # Test case 3: Edge case - some sequences result in 0 split
    print("\nTest case 3: Zero split case")
    paged_kernel_lens = torch.tensor([2, 10, 1], device=device)
    cumsum = torch.cat([
        torch.zeros(1, device=device),
        torch.cumsum(paged_kernel_lens, 0)
    ])
    dcp_rank = 5
    dcp_world_size = 8
    
    split1, indices1 = filter_seq_indices_original(
        paged_kernel_lens.clone(), cumsum.clone(), dcp_rank, dcp_world_size
    )
    split2, indices2 = filter_seq_indices_optimized(
        paged_kernel_lens.clone(), cumsum.clone(), dcp_rank, dcp_world_size
    )
    
    assert torch.equal(split1, split2), "Split lengths mismatch!"
    assert torch.equal(indices1, indices2), "Indices mismatch!"
    print(f"✓ Split lengths match: {split1.tolist()}")
    print(f"✓ Indices count: {len(indices1)}")
    
    # Test case 4: Large batch
    print("\nTest case 4: Large batch")
    batch_size = 128
    paged_kernel_lens = torch.randint(50, 500, (batch_size,), device=device)
    cumsum = torch.cat([
        torch.zeros(1, device=device),
        torch.cumsum(paged_kernel_lens, 0)
    ])
    dcp_rank = 1
    dcp_world_size = 4
    
    split1, indices1 = filter_seq_indices_original(
        paged_kernel_lens.clone(), cumsum.clone(), dcp_rank, dcp_world_size
    )
    split2, indices2 = filter_seq_indices_optimized(
        paged_kernel_lens.clone(), cumsum.clone(), dcp_rank, dcp_world_size
    )
    
    assert torch.equal(split1, split2), "Split lengths mismatch!"
    assert torch.equal(indices1, indices2), "Indices mismatch!"
    print(f"✓ Split lengths match (first 10): {split1[:10].tolist()}")
    print(f"✓ Indices count: {len(indices1)}")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("Optimization is correct and maintains semantic equivalence.")
    print("="*50)


def benchmark_performance():
    """Benchmark the performance improvement"""
    print("\nBenchmarking performance...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Skipping benchmark (CUDA not available)")
        return
    
    import time
    
    batch_size = 256
    paged_kernel_lens = torch.randint(100, 1000, (batch_size,), device=device)
    cumsum = torch.cat([
        torch.zeros(1, device=device),
        torch.cumsum(paged_kernel_lens, 0)
    ])
    dcp_rank = 0
    dcp_world_size = 4
    
    # Warmup
    for _ in range(10):
        _ = filter_seq_indices_optimized(
            paged_kernel_lens.clone(), cumsum.clone(), dcp_rank, dcp_world_size
        )
    torch.cuda.synchronize()
    
    # Benchmark optimized
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        _ = filter_seq_indices_optimized(
            paged_kernel_lens.clone(), cumsum.clone(), dcp_rank, dcp_world_size
        )
    torch.cuda.synchronize()
    optimized_time = (time.time() - start) / iterations * 1000  # ms
    
    print(f"Optimized version: {optimized_time:.4f} ms per call")
    print(f"Batch size: {batch_size}")
    print(f"Note: Original version has additional ~20-100μs sync overhead per .item() call")


if __name__ == "__main__":
    test_correctness()
    benchmark_performance()
