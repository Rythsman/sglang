import itertools
import unittest

import numpy as np
import torch

from sglang.srt.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
    filter_kv_indices_dcp_triton,
)
from sglang.test.test_utils import CustomTestCase


class TestCreateKvIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_test(self, batch, max_batch, max_context_len):
        req_to_token = torch.arange(
            max_batch * max_context_len, dtype=torch.int32, device="cuda"
        ).reshape((max_batch, max_context_len))
        req_pool_indices = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_batch), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device="cuda",
        )
        paged_kernel_lens = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_context_len), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device="cuda",
        )

        kv_indptr = torch.zeros((batch + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # ref
        req_pool_indices_cpu = req_pool_indices.cpu().numpy()
        paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
        kv_indices_ref = torch.cat(
            [
                req_to_token[req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]]
                for i in range(batch)
            ],
            dim=0,
        ).contiguous()

        # triton
        kv_indices_triton = torch.empty(kv_indptr[-1], dtype=torch.int32, device="cuda")
        create_flashinfer_kv_indices_triton[(batch,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            None,
            kv_indices_triton,
            req_to_token.size(1),
        )

        # Check
        self.assertTrue(torch.equal(kv_indices_ref, kv_indices_triton))

    def test_create_kvindices(self):
        BATCH = [1, 37, 1786]
        MAX_BATCH = 4096
        MAX_CONTEXT_LEN = 4096
        for batch in BATCH:
            self._run_test(batch, MAX_BATCH, MAX_CONTEXT_LEN)


class TestFilterKvIndicesDCP(CustomTestCase):
    """Test filter_kv_indices_dcp_triton kernel for DCP (Data Center Parallel)."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _filter_seq_indices_reference(
        self,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_cumsum: torch.Tensor,
        dcp_rank: int,
        dcp_world_size: int,
    ):
        """Reference implementation using .item() calls (original code)."""
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

    def _run_test(self, batch, max_context_len, dcp_rank, dcp_world_size):
        """Run a single test case comparing Triton kernel with reference implementation."""
        # Generate random kv_indices (simulating the output of create_flashinfer_kv_indices_triton)
        paged_kernel_lens = torch.randint(
            1, max_context_len + 1, (batch,), dtype=torch.int32, device="cuda"
        )

        # Compute kv_indptr (cumsum)
        kv_indptr = torch.zeros((batch + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        total_kv = int(kv_indptr[-1].item())
        kv_indices = torch.randint(
            0, 100000, (total_kv,), dtype=torch.int32, device="cuda"
        )

        # Reference implementation
        filtered_lens_ref, filter_indices_ref = self._filter_seq_indices_reference(
            paged_kernel_lens, kv_indptr, dcp_rank, dcp_world_size
        )

        if filter_indices_ref.numel() == 0:
            # Skip empty case
            return

        kv_indices_ref = kv_indices[filter_indices_ref] // dcp_world_size

        # Triton implementation
        filtered_lens_triton = (
            (paged_kernel_lens - dcp_rank - 1) // dcp_world_size + 1
        ).clamp_(min=0)

        out_indptr = torch.zeros((batch + 1,), dtype=torch.int32, device="cuda")
        out_indptr[1:] = torch.cumsum(filtered_lens_triton, dim=0)

        out_total = int(out_indptr[-1].item())
        kv_indices_triton = torch.empty(out_total, dtype=torch.int32, device="cuda")

        filter_kv_indices_dcp_triton[(batch,)](
            kv_indices,
            kv_indptr,
            kv_indices_triton,
            out_indptr,
            dcp_rank,
            dcp_world_size,
        )

        # Verify results
        self.assertTrue(
            torch.equal(filtered_lens_ref, filtered_lens_triton),
            f"Filtered lengths mismatch: ref={filtered_lens_ref}, triton={filtered_lens_triton}",
        )
        self.assertTrue(
            torch.equal(kv_indices_ref.to(torch.int32), kv_indices_triton),
            f"KV indices mismatch at batch={batch}, rank={dcp_rank}, world_size={dcp_world_size}",
        )

    def test_filter_kv_indices_dcp(self):
        """Test filter_kv_indices_dcp_triton with various configurations."""
        BATCH_SIZES = [1, 8, 32, 128]
        MAX_CONTEXT_LEN = 512
        DCP_CONFIGS = [(0, 2), (1, 2), (0, 4), (2, 4), (3, 8)]

        for batch in BATCH_SIZES:
            for dcp_rank, dcp_world_size in DCP_CONFIGS:
                with self.subTest(
                    batch=batch, dcp_rank=dcp_rank, dcp_world_size=dcp_world_size
                ):
                    self._run_test(batch, MAX_CONTEXT_LEN, dcp_rank, dcp_world_size)

    def test_edge_cases(self):
        """Test edge cases like single element, all zeros, etc."""
        # Single batch item
        self._run_test(1, 100, 0, 2)

        # Short sequences (may result in 0 filtered elements for some items)
        paged_kernel_lens = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
        kv_indptr = torch.zeros((5,), dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.arange(
            int(kv_indptr[-1].item()), dtype=torch.int32, device="cuda"
        )

        dcp_rank = 0
        dcp_world_size = 4

        # Reference
        filtered_lens_ref, filter_indices_ref = self._filter_seq_indices_reference(
            paged_kernel_lens, kv_indptr, dcp_rank, dcp_world_size
        )

        # Triton
        filtered_lens_triton = (
            (paged_kernel_lens - dcp_rank - 1) // dcp_world_size + 1
        ).clamp_(min=0)
        out_indptr = torch.zeros((5,), dtype=torch.int32, device="cuda")
        out_indptr[1:] = torch.cumsum(filtered_lens_triton, dim=0)

        out_total = int(out_indptr[-1].item())
        kv_indices_triton = torch.empty(out_total, dtype=torch.int32, device="cuda")

        filter_kv_indices_dcp_triton[(4,)](
            kv_indices,
            kv_indptr,
            kv_indices_triton,
            out_indptr,
            dcp_rank,
            dcp_world_size,
        )

        kv_indices_ref = kv_indices[filter_indices_ref] // dcp_world_size
        self.assertTrue(torch.equal(kv_indices_ref.to(torch.int32), kv_indices_triton))


if __name__ == "__main__":
    unittest.main()
