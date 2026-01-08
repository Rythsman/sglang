"""A tiny Transformers-backend model for testing post_load_weights APIs.

This model is test-only and is registered via SGLANG_EXTERNAL_MODEL_PACKAGE.
"""

from __future__ import annotations

from sglang.srt.models.transformers import TransformersForCausalLM


class FakeTransformersPostLoadForCausalLM(TransformersForCausalLM):
    """A tiny model that records post-load hook invocations."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.post_load_call_count = 0

    def post_load_weights(self, *args, **kwargs):
        """Record that post_load_weights has been called."""
        self.post_load_call_count += 1


EntryClass = FakeTransformersPostLoadForCausalLM

