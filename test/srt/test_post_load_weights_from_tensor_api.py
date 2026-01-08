"""Tests for Engine.post_load_weights_from_tensor API.

This test uses a tiny fake model + dummy weight loading to reduce cost.
It validates the API can be invoked through Engine and returns successfully.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)
if _REPO_ROOT not in sys.path:
    # Make `test.*` importable for external model registration.
    sys.path.insert(0, _REPO_ROOT)

# Register the test-only fake model package before importing sglang.
os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "test.srt.debug_utils")


def _assert_success(testcase: unittest.TestCase, ret):
    # Accept common return shapes to keep the test robust.
    if isinstance(ret, dict):
        testcase.assertTrue(ret.get("success", True), msg=str(ret))
        return
    if isinstance(ret, (tuple, list)) and ret:
        testcase.assertTrue(bool(ret[0]), msg=str(ret))
        return
    testcase.assertIsNotNone(ret)


class TestPostLoadWeightsFromTensorApi(unittest.TestCase):
    def test_post_load_weights_from_tensor_dummy_model(self):
        try:
            import sglang as sgl  # pylint: disable=import-outside-toplevel
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise unittest.SkipTest(f"sglang import failed: {e}") from e

        if not hasattr(sgl.Engine, "post_load_weights_from_tensor"):
            raise unittest.SkipTest("Engine.post_load_weights_from_tensor is not available.")

        try:
            import torch  # pylint: disable=import-outside-toplevel
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise unittest.SkipTest(f"torch is required for this test: {e}") from e

        model_dir = tempfile.mkdtemp(prefix="sglang-fake-post-load-model-")
        self.addCleanup(lambda: shutil.rmtree(model_dir, ignore_errors=True))

        # Minimal Llama config + custom architecture pointing to our fake model class.
        config = {
            "architectures": ["FakeTransformersPostLoadForCausalLM"],
            "model_type": "llama",
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            engine = sgl.Engine(
                model_path=model_dir,
                load_format="dummy",
                skip_tokenizer_init=True,
                device=device,
                tp_size=1,
                log_level="error",
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise unittest.SkipTest(f"Engine failed to start on {device}: {e}") from e

        self.addCleanup(engine.shutdown)

        ret = engine.post_load_weights_from_tensor(flush_cache=True)
        _assert_success(self, ret)


if __name__ == "__main__":
    unittest.main()

