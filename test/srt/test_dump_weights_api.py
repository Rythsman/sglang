"""Tests for Engine.dump_weights API.

This is a heavyweight integration-style test:
- It requires >= 4 GPUs for tp=4.
- It requires a local model directory on disk.

The test is skipped automatically unless prerequisites are met.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)


_DEFAULT_MODEL_PATH = "/mmu_mllm_hdd_2/wanghao44/models/Qwen2.5-VL-7B-Instruct"
_MODEL_PATH_ENV = "SGLANG_TEST_DUMP_WEIGHTS_MODEL_PATH"


def _get_bool_env(name: str) -> bool:
    val = (os.environ.get(name, "") or "").strip().lower()
    return val in ("1", "true", "yes", "y", "on")


def _is_in_ci() -> bool:
    try:
        from sglang.test.test_utils import is_in_ci  # pylint: disable=import-outside-toplevel

        return bool(is_in_ci())
    except Exception:
        return _get_bool_env("SGLANG_IS_IN_CI")


def _get_base_test_case():
    try:
        from sglang.test.test_utils import (  # pylint: disable=import-outside-toplevel
            CustomTestCase,
        )

        return CustomTestCase
    except Exception:
        return unittest.TestCase


class TestDumpWeightsApi(_get_base_test_case()):
    def _get_model_path(self) -> str:
        model_path = os.environ.get(_MODEL_PATH_ENV, _DEFAULT_MODEL_PATH)
        return os.path.abspath(model_path)

    def _skip_if_not_ready(self, model_path: str):
        if _is_in_ci():
            raise unittest.SkipTest("Skip heavyweight dump_weights test in CI.")
        try:
            import torch  # pylint: disable=import-outside-toplevel
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise unittest.SkipTest(f"torch is required for this test: {e}") from e

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for this test.")
        if torch.cuda.device_count() < 4:
            raise unittest.SkipTest("Need at least 4 GPUs for tp=4.")
        if not os.path.isdir(model_path):
            raise unittest.SkipTest(
                f"Local model path not found: {model_path}. "
                f"Set {_MODEL_PATH_ENV} to override."
            )

    def test_dump_weights_and_consistency(self):
        model_path = self._get_model_path()
        self._skip_if_not_ready(model_path)

        # Lazy imports to keep test collection lightweight.
        import sglang as sgl  # pylint: disable=import-outside-toplevel
        from sglang.test.dump_consistency import (  # pylint: disable=import-outside-toplevel
            DumpConsistencyTester,
        )

        out_dir1 = tempfile.mkdtemp(prefix="sglang-dump-weights-1-")
        out_dir2 = tempfile.mkdtemp(prefix="sglang-dump-weights-2-")
        self.addCleanup(lambda: shutil.rmtree(out_dir1, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(out_dir2, ignore_errors=True))

        max_tensors = int(os.environ.get("SGLANG_TEST_DUMP_WEIGHTS_MAX_TENSORS", "64"))
        tolerance = float(os.environ.get("SGLANG_TEST_DUMP_WEIGHTS_TOLERANCE", "1e-6"))

        engine = sgl.Engine(
            model_path=model_path,
            tp_size=4,
            enable_multimodal=True,
            trust_remote_code=True,
            dtype="bfloat16",
            log_level="error",
        )
        self.addCleanup(engine.shutdown)

        ret1 = engine.dump_weights(
            output_path=out_dir1,
            dump_mode="local_shard",
            layers="all",
            format="pth",
        )
        self.assertTrue(ret1.get("success"), msg=str(ret1))

        ret2 = engine.dump_weights(
            output_path=out_dir2,
            dump_mode="local_shard",
            layers="all",
            format="pth",
        )
        self.assertTrue(ret2.get("success"), msg=str(ret2))

        tester = DumpConsistencyTester(
            tolerance=tolerance,
            max_tensors_per_file=max_tensors,
            verify_tensor_values=True,
        )
        ok = tester.compare_directories(out_dir1, out_dir2)
        if not ok:
            self.fail("\n".join(tester.errors[:50]))


if __name__ == "__main__":
    unittest.main()

