from __future__ import annotations

import dataclasses
import functools
import gzip
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import psutil
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.managers.tokenizer_manager import ReqState
    from sglang.srt.speculative.eagle_info import EagleDraftInput


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GenerationBatchResult:
    logits_output: Optional[LogitsProcessorOutput] = None
    pp_hidden_states_proxy_tensors: Optional[PPProxyTensors] = None
    next_token_ids: Optional[torch.Tensor] = None
    num_accepted_tokens: Optional[int] = None
    can_run_cuda_graph: bool = False

    # For output processing
    extend_input_len_per_req: Optional[List[int]] = None
    extend_logprob_start_len_per_req: Optional[List[int]] = None

    # For overlap scheduling
    copy_done: Optional[torch.cuda.Event] = None
    delay_sample_func: Optional[callable] = None
    future_indices: Optional[FutureIndices] = None

    # FIXME(lsyin): maybe move to a better place?
    # sync path: forward stream -> output processor
    accept_lens: Optional[torch.Tensor] = None

    # relay path: forward stream -> next step forward
    next_draft_input: Optional[EagleDraftInput] = None

    def copy_to_cpu(self, return_logprob: bool):
        """Copy tensors to CPU in overlap scheduling.
        Only the tensors which are needed for processing results are copied,
        e.g., next_token_ids, logits outputs
        """
        if return_logprob:
            if self.logits_output.next_token_logprobs is not None:
                self.logits_output.next_token_logprobs = (
                    self.logits_output.next_token_logprobs.to("cpu", non_blocking=True)
                )
            if self.logits_output.input_token_logprobs is not None:
                self.logits_output.input_token_logprobs = (
                    self.logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                )
        if self.logits_output.hidden_states is not None:
            self.logits_output.hidden_states = self.logits_output.hidden_states.to(
                "cpu", non_blocking=True
            )
        self.next_token_ids = self.next_token_ids.to("cpu", non_blocking=True)

        if self.accept_lens is not None:
            self.accept_lens = self.accept_lens.to("cpu", non_blocking=True)

        self.copy_done.record()

    @classmethod
    def from_pp_proxy(
        cls, logits_output, next_pp_outputs: PPProxyTensors, can_run_cuda_graph
    ):
        # TODO(lsyin): refactor PP and avoid using dict
        proxy_dict = next_pp_outputs.tensors
        return cls(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=next_pp_outputs["next_token_ids"],
            extend_input_len_per_req=proxy_dict.get("extend_input_len_per_req", None),
            extend_logprob_start_len_per_req=proxy_dict.get(
                "extend_logprob_start_len_per_req", None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
        )


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
) -> Optional[str]:
    """Validate and potentially truncate input length.

    Args:
        req: The request containing input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs

    Returns:
        Error message if validation fails, None if successful
    """
    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated. "
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            return error_msg

    return None


def get_logprob_dict_from_result(result: GenerationBatchResult) -> dict:

    logits_output = result.logits_output
    assert logits_output is not None

    return {
        "extend_input_len_per_req": result.extend_input_len_per_req,
        "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
        "next_token_logprobs": result.logits_output.next_token_logprobs,
        "next_token_top_logprobs_val": result.logits_output.next_token_top_logprobs_val,
        "next_token_top_logprobs_idx": result.logits_output.next_token_top_logprobs_idx,
        "next_token_token_ids_logprobs_val": result.logits_output.next_token_token_ids_logprobs_val,
        "next_token_token_ids_logprobs_idx": result.logits_output.next_token_token_ids_logprobs_idx,
        "input_token_logprobs": result.logits_output.input_token_logprobs,
        "input_top_logprobs_val": result.logits_output.input_top_logprobs_val,
        "input_top_logprobs_idx": result.logits_output.input_top_logprobs_idx,
        "input_token_ids_logprobs_val": result.logits_output.input_token_ids_logprobs_val,
        "input_token_ids_logprobs_idx": result.logits_output.input_token_ids_logprobs_idx,
    }


def get_logprob_from_pp_outputs(
    next_pp_outputs: PPProxyTensors,
) -> tuple[LogitsProcessorOutput, list[int], list[int]]:
    logits_output = LogitsProcessorOutput(
        # Do not send logits and hidden states because they are large
        next_token_logits=None,
        hidden_states=None,
        next_token_logprobs=next_pp_outputs["next_token_logprobs"],
        next_token_top_logprobs_val=next_pp_outputs["next_token_top_logprobs_val"],
        next_token_top_logprobs_idx=next_pp_outputs["next_token_top_logprobs_idx"],
        next_token_token_ids_logprobs_val=next_pp_outputs[
            "next_token_token_ids_logprobs_val"
        ],
        next_token_token_ids_logprobs_idx=next_pp_outputs[
            "next_token_token_ids_logprobs_idx"
        ],
        input_token_logprobs=next_pp_outputs["input_token_logprobs"],
        input_top_logprobs_val=next_pp_outputs["input_top_logprobs_val"],
        input_top_logprobs_idx=next_pp_outputs["input_top_logprobs_idx"],
        input_token_ids_logprobs_val=next_pp_outputs["input_token_ids_logprobs_val"],
        input_token_ids_logprobs_idx=next_pp_outputs["input_token_ids_logprobs_idx"],
    )
    extend_input_len_per_req = next_pp_outputs["extend_input_len_per_req"]
    extend_logprob_start_len_per_req = next_pp_outputs[
        "extend_logprob_start_len_per_req"
    ]

    return logits_output, extend_input_len_per_req, extend_logprob_start_len_per_req


class TracingManager:
    def __init__(self, prefix: str, output_dir: str):
        # file_name = f"{prefix}custom_profiler.chrome_trace.json"
        file_name = f"{prefix}custom_profiler.trace.json.gz"
        self.output_path = os.path.join(output_dir, file_name)
        self.events = []
        self.pid = psutil.Process().name()

        logger.info(f"Tracing events will be dumped to {self.output_path}")

    def dump_events(self):
        logger.info(f"Dumpping events to {self.output_path}")
        if self.output_path.endswith(".gz"):
            with gzip.open(self.output_path, "wt") as f:
                f.write(json.dumps(self.events, indent=4, separators=(",", ":")))
        else:
            with open(self.output_path, "w") as f:
                f.write(json.dumps(self.events, indent=4, separators=(",", ":")))
        logger.info(f"Dump events({len(self.events)}) finished")
        self.events = []

    def append(self, event: dict):
        self.events.append(event)


tracing_manager: Optional[TracingManager] = None


def init_tracing_manager(prefix: str, output_dir: str):
    global tracing_manager
    if tracing_manager is None:
        tracing_manager = TracingManager(prefix, output_dir)


def dump_tracing_events():
    global tracing_manager
    if tracing_manager is not None:
        tracing_manager.dump_events()


def trace_req(rid: int, state: ReqState, extra_info: Dict[Any, Any]):
    global tracing_manager
    if tracing_manager is None:
        return

    prompt_tokens = extra_info["meta_info"]["prompt_tokens"]
    create_time = int(state.created_time * 1000000.0)
    first_token_time = int(state.first_token_time * 1000000.0)
    finished_time = int(state.finished_time * 1000000.0)

    tracing_manager.append(
        {
            "cat": f"req_{rid}",
            "name": f"req_{rid}_prefill",
            "ts": create_time,
            "dur": first_token_time - create_time,
            "ph": "X",
            "pid": tracing_manager.pid,
            "tid": rid,
            "args": {
                "track_id": "prefill",
                "prompt_tokens": prompt_tokens,
                "throughput": prompt_tokens
                / (state.first_token_time - state.created_time),
                **(state.mm_info or {}),
            },
        }
    )

    if len(state.output_ids) <= 1:
        return

    tracing_manager.append(
        {
            "cat": f"req_{rid}",
            "name": f"req_{rid}_decode",
            "ts": first_token_time + 1,
            "dur": finished_time - first_token_time,
            "ph": "X",
            "pid": tracing_manager.pid,
            "tid": rid,
            "args": {
                "track_id": "decode",
                "output_tokens": len(state.output_ids),
                "throughput": len(state.output_ids)
                / (state.finished_time - state.first_token_time),
            },
        }
    )


class TracingProfiler:
    def __init__(
        self,
        name: str,
        forward_batch: Union[ForwardBatch, ScheduleBatch] = None,
        extra_data: dict = {},
    ):
        if forward_batch is None and extra_data.get("args", None) is not None:
            forward_batch = extra_data["args"][1]

        self.batch = forward_batch
        self.start_time: Optional[float] = None
        self.name = f"{name}_{self.batch.forward_mode.name}"
        self.extra_data = extra_data

    @property
    def batch_size(self):
        if isinstance(self.batch, ScheduleBatch):
            return self.batch.batch_size()
        return self.batch.batch_size

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global tracing_manager
        if tracing_manager is None:
            return
        end_time = time.time()
        duration = end_time - self.start_time
        tracing_manager.append(
            {
                "cat": self.name,
                "name": self.name,
                "ts": self.start_time * 1000000.0,
                "dur": duration * 1000000.0,
                "ph": "X",
                "pid": tracing_manager.pid,
                "args": {
                    "batch_size": self.batch_size,
                },
            }
        )


def trace_profile(name: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            extra_data = {
                "args": args,
                "kwargs": kwargs,
            }
            with TracingProfiler(name, extra_data=extra_data):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def extract_mm_info(
    tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]
):
    result = {
        "mm_load_time(ms)": tokenized_obj.mm_inputs["mm_load_time"] * 1000.0,
        "mm_preprocess_time(ms)": tokenized_obj.mm_inputs["mm_preprocess_time"]
        * 1000.0,
        "mm_process_time(ms)": tokenized_obj.mm_inputs["mm_process_time"] * 1000.0,
        "mm_total_time(ms)": tokenized_obj.mm_inputs["mm_total_time"] * 1000.0,
    }

    for item in tokenized_obj.mm_inputs["mm_items"]:
        result[item.modality.name] = {
            "offsets": item.offsets,
        }
    return result
