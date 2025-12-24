from __future__ import annotations

import dataclasses
import gzip
import json
import logging
import os
import threading
import time
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import psutil
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput


logger = logging.getLogger(__name__)

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None

_BYTES_TO_GIB = 1.0 / (1024.0 * 1024.0 * 1024.0)


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


class TraceManager:
    def __init__(
        self,
        prefix: str,
        output_dir: str,
        tp_rank: int,
        pp_rank: int,
        enable_nvml_sampler: bool = True,
    ):
        # file_name = f"{prefix}custom_profiler.chrome_trace.json"
        main_file_name = f"{prefix}custom_profiler.trace.json.gz"
        nvml_file_name = f"{prefix}custom_profiler.nvml.trace.json.gz"
        self.output_path = os.path.join(output_dir, main_file_name)
        self.nvml_output_path = os.path.join(output_dir, nvml_file_name)
        # Main trace events.
        #
        # NOTE: The serving stack assumes stop_profile() is only called after all
        # in-flight requests have finished and no more trace events will be appended.
        # With that contract, we can keep a simple list without locking.
        self.events: List[dict] = []
        # NVML events are produced by a single background thread. We only read and dump them
        # after joining that thread, so no additional locking is needed.
        self._nvml_events: List[dict] = []
        self.pid = os.getpid()
        self.process_name = psutil.Process().name()
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.enable_nvml_sampler = enable_nvml_sampler
        self._nvml_sampler_stop: Optional[threading.Event] = None
        self._nvml_sampler_thread: Optional[threading.Thread] = None

        logger.info(f"Tracing events will be dumped to {self.output_path}")
        self._append_metadata_to_main()
        if self.enable_nvml_sampler:
            self._maybe_start_nvml_sampler()

    def _metadata_events(self) -> List[dict]:
        """Build Chrome trace metadata events.

        NOTE: Chrome trace metadata expects numeric pid/tid. We keep tid as string
        in other events for readability, but use tid=0 in metadata.
        """
        return [
            {
                "ph": "M",
                "name": "process_name",
                "pid": self.pid,
                "tid": 0,
                "args": {"name": self.process_name},
            },
            {
                "ph": "M",
                "name": "process_labels",
                "pid": self.pid,
                "tid": 0,
                "args": {"tp_rank": self.tp_rank, "pp_rank": self.pp_rank},
            },
        ]

    def _append_metadata_to_main(self) -> None:
        for e in self._metadata_events():
            self.append(e)

    def _append_metadata_to_nvml(self) -> None:
        for e in self._metadata_events():
            self.append_nvml(e)

    @staticmethod
    def _get_nvml_sampling_interval_s() -> float:
        """Return NVML sampling interval in seconds.

        Set env var SGLANG_NVML_SAMPLING_INTERVAL_MS:
        - <= 0: disable sampling
        - > 0: sampling interval in milliseconds
        """
        raw = os.getenv("SGLANG_NVML_SAMPLING_INTERVAL_MS", "500").strip()
        try:
            interval_ms = float(raw)
        except ValueError:
            interval_ms = 500.0
        return interval_ms / 1000.0

    def _maybe_start_nvml_sampler(self):
        """Start a low-frequency NVML sampler in a background thread.

        The sampler is only enabled when:
        - custom trace manager is initialized
        - torch.cuda is available
        - NVML is available
        - SGLANG_NVML_SAMPLING_INTERVAL_MS > 0
        """
        interval_s = self._get_nvml_sampling_interval_s()
        if interval_s <= 0:
            return
        if not torch.cuda.is_available():
            return
        if not _ensure_nvml_inited():
            return
        if (
            self._nvml_sampler_thread is not None
            and self._nvml_sampler_thread.is_alive()
        ):
            return

        # Write metadata once the sampler is confirmed to start.
        self._append_metadata_to_nvml()
        stop = threading.Event()
        thread = threading.Thread(
            target=self._nvml_sampler_loop,
            args=(stop, interval_s),
            daemon=True,
            name=f"sglang-nvml-sampler-tp{self.tp_rank}-pp{self.pp_rank}",
        )
        self._nvml_sampler_stop = stop
        self._nvml_sampler_thread = thread
        thread.start()

    def stop_nvml_sampler(self):
        stop = self._nvml_sampler_stop
        thread = self._nvml_sampler_thread
        if stop is None or thread is None:
            return
        stop.set()
        thread.join(timeout=1.0)
        if thread.is_alive():
            # Keep handles so callers can retry stop later. Avoid dumping NVML events
            # while the sampler is still running.
            return
        self._nvml_sampler_stop = None
        self._nvml_sampler_thread = None

    def _nvml_sampler_loop(self, stop: threading.Event, interval_s: float):
        tid = f"NVML TP{self.tp_rank} PP{self.pp_rank}"
        while not stop.is_set():
            self._trace_nvml_memory_counter(tid=tid)
            stop.wait(interval_s)

    def append_nvml(self, event: dict) -> None:
        self._nvml_events.append(event)

    def _trace_nvml_memory_counter(self, tid: str, device_index: Optional[int] = None):
        """Append NVML used/free/total memory as Chrome counter events."""
        if not torch.cuda.is_available():
            return
        if not _ensure_nvml_inited():
            return

        if device_index is None:
            try:
                device_index = torch.cuda.current_device()
            except Exception:
                return
        try:
            handle = _get_nvml_handle(device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception:
            return

        self.append_nvml(
            {
                "cat": "NVML Memory",
                "name": "NVML Memory",
                "ts": time.time() * 1000000.0,
                "ph": "C",
                "pid": self.pid,
                "tid": tid,
                "args": {
                    "used_bytes": mem.used * _BYTES_TO_GIB,
                    "free_bytes": mem.free * _BYTES_TO_GIB,
                    "total_bytes": mem.total * _BYTES_TO_GIB,
                },
            }
        )

    def dump_events(self):
        """Dump collected (non-NVML) Chrome trace events to disk.

        This is used by the scheduler process. Profiling output failures should
        never crash the serving loop (e.g. user moved/deleted the output dir).
        """
        logger.info(f"Dumpping events to {self.output_path}")
        events = self.events
        self.events = []

        self._dump_events_to_path(events=events, output_path=self.output_path)
        logger.info(f"Dump events({len(events)}) finished")

    def dump_nvml_events(self) -> None:
        """Dump collected NVML Chrome trace events to disk.

        NVML events are only appended by the sampler thread. Call stop_nvml_sampler()
        before dumping to avoid races.
        """
        if not self.enable_nvml_sampler:
            return
        if self._nvml_sampler_thread is not None and self._nvml_sampler_thread.is_alive():
            return
        if not self._nvml_events:
            return
        logger.info(f"Dumpping NVML events to {self.nvml_output_path}")
        self._dump_events_to_path(events=self._nvml_events, output_path=self.nvml_output_path)
        logger.info(f"Dump NVML events({len(self._nvml_events)}) finished")

    def append(self, event: dict):
        self.events.append(event)

    def _dump_events_to_path(self, events: List[dict], output_path: str) -> None:
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            # Ensure output directory exists. This makes repeated profiling
            # robust even if the user moves/deletes the profile directory.
            os.makedirs(parent_dir, exist_ok=True)

        payload = json.dumps(events, indent=4, separators=(",", ":"))

        def _write_atomically() -> None:
            # Atomic replace avoids partial/corrupted files if the process exits
            # or errors out mid-write.
            tmp_path = f"{output_path}.tmp.{os.getpid()}.{time.time_ns()}"
            try:
                if output_path.endswith(".gz"):
                    with gzip.open(tmp_path, "wt") as f:
                        f.write(payload)
                else:
                    with open(tmp_path, "w") as f:
                        f.write(payload)
                os.replace(tmp_path, output_path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

        try:
            _write_atomically()
        except FileNotFoundError:
            # The directory may have been moved/deleted between makedirs and open.
            if parent_dir:
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                    _write_atomically()
                except Exception:
                    logger.warning(
                        "Failed to dump profiling events (directory missing). "
                        "Profiling output is skipped for stability.",
                        exc_info=True,
                    )
                    if output_path == self.output_path:
                        self.events = events + self.events
                    return
            else:
                logger.warning(
                    "Failed to dump profiling events (invalid output path). "
                    "Profiling output is skipped for stability.",
                    exc_info=True,
                )
                if output_path == self.output_path:
                    self.events = events + self.events
                return
        except Exception:
            logger.warning(
                "Failed to dump profiling events. Profiling output is skipped for stability.",
                exc_info=True,
            )
            if output_path == self.output_path:
                self.events = events + self.events
            return


trace_manager: Optional[TraceManager] = None


def init_trace_manager(
    prefix: str,
    output_dir: str,
    tp_rank: int = 0,
    pp_rank: int = 0,
    force: bool = False,
    enable_nvml_sampler: Optional[bool] = None,
):
    global trace_manager
    if enable_nvml_sampler is None:
        enable_nvml_sampler = (
            trace_manager.enable_nvml_sampler if trace_manager is not None else True
        )

    new_output_path = os.path.join(output_dir, f"{prefix}custom_profiler.trace.json.gz")
    needs_reinit = (
        force
        or trace_manager is None
        or trace_manager.output_path != new_output_path
        or trace_manager.tp_rank != tp_rank
        or trace_manager.pp_rank != pp_rank
    )
    if not needs_reinit:
        return

    if trace_manager is not None:
        old = trace_manager
        # Prevent any late trace appends during dumping/re-init.
        trace_manager = None
        old.stop_nvml_sampler()
        old.dump_events()
        old.dump_nvml_events()
    trace_manager = TraceManager(
        prefix,
        output_dir,
        tp_rank,
        pp_rank,
        enable_nvml_sampler=enable_nvml_sampler,
    )


def get_trace_manager() -> Optional[TraceManager]:
    global trace_manager
    return trace_manager


def dump_trace_events():
    global trace_manager
    if trace_manager is not None:
        tm = trace_manager
        # Prevent any late trace appends during dumping.
        trace_manager = None
        tm.stop_nvml_sampler()
        tm.dump_events()
        tm.dump_nvml_events()


class ReqTraceStatus(IntEnum):
    PRE_SCHEDULER = auto()
    MM_PROCESS = auto()
    PRE_SCHEDULER_COMM = auto()
    SCHEDULER_BROADCAST = auto()
    SCHEDULER_WAITING = auto()
    SCHEDULER_PREFILL = auto()
    SCHEDULER_DECODE = auto()
    POST_SCHEDULER = auto()


def trace_req_begin(
    rid: int,
    status: ReqTraceStatus,
    time_s: Optional[float] = None,
    extra_info: Optional[Dict[Any, Any]] = None,
):
    global trace_manager
    if (
        trace_manager is None
        or trace_manager.tp_rank != 0
        or trace_manager.pp_rank != 0
    ):
        return

    ts_s = time.time() if time_s is None else time_s
    trace_manager.append(
        {
            "cat": f"req_{rid}",
            "name": status.name,
            "ts": ts_s * 1000000.0,
            "ph": "B",
            # Keep request-level events in a dedicated pseudo-process to avoid
            # leaking request details into the scheduler/model runner timelines.
            "pid": "ReqDetail",
            "tid": rid,
            "args": extra_info or {},
        }
    )


def trace_req_end(
    rid: int,
    status: ReqTraceStatus,
    time_s: Optional[float] = None,
    extra_info: Optional[Dict[Any, Any]] = None,
):
    global trace_manager
    if (
        trace_manager is None
        or trace_manager.tp_rank != 0
        or trace_manager.pp_rank != 0
    ):
        return

    ts_s = time.time() if time_s is None else time_s
    trace_manager.append(
        {
            "cat": f"req_{rid}",
            "name": status.name,
            "ts": ts_s * 1000000.0,
            "ph": "E",
            # Keep request-level events in a dedicated pseudo-process to avoid
            # leaking request details into the scheduler/model runner timelines.
            "pid": "ReqDetail",
            "tid": rid,
            "args": extra_info or {},
        }
    )


class BatchTraceStatus(IntEnum):
    ENCODER = auto()
    PREFILL = auto()
    DECODE = auto()


def trace_batch_begin(
    status: Union[ForwardMode, BatchTraceStatus],
    extra_info: Optional[Dict[Any, Any]] = None,
    tid: str = "Default",
):
    global trace_manager
    if trace_manager is None:
        return

    trace_manager.append(
        {
            "cat": status.name,
            "name": status.name,
            "ts": time.time() * 1000000.0,
            "ph": "B",
            "pid": trace_manager.pid,
            "tid": tid,
            "args": extra_info or {},
        }
    )


def trace_batch_end(
    status: Union[ForwardMode, BatchTraceStatus],
    extra_info: Optional[Dict[Any, Any]] = None,
    tid: str = "Default",
):
    global trace_manager
    if trace_manager is None:
        return

    trace_manager.append(
        {
            "cat": status.name,
            "name": status.name,
            "ts": time.time() * 1000000.0,
            "ph": "E",
            "pid": trace_manager.pid,
            "tid": tid,
            "args": extra_info or {},
        }
    )


def trace_usage(name: str, args: Dict[Any, Any], tid: str = "Usage"):
    global trace_manager
    if trace_manager is None:
        return

    trace_manager.append(
        {
            "cat": name,
            "name": name,
            "ts": time.time() * 1000000.0,
            "ph": "C",
            "pid": trace_manager.pid,
            "tid": tid,
            "args": args,
        }
    )


_nvml_inited = False
_nvml_handles: Dict[int, Any] = {}


def _ensure_nvml_inited() -> bool:
    global _nvml_inited
    if pynvml is None:
        return False
    if _nvml_inited:
        return True
    try:
        pynvml.nvmlInit()
    except Exception:
        return False
    _nvml_inited = True
    return True


def _get_nvml_handle(device_index: int):
    if device_index in _nvml_handles:
        return _nvml_handles[device_index]
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    _nvml_handles[device_index] = handle
    return handle


def extract_mm_info(
    mm_inputs: Dict[str, Any],
):
    result = {}
    if mm_inputs is None:
        return result

    if "mm_load_time" in mm_inputs:
        result["mm_load_time(ms)"] = mm_inputs["mm_load_time"] * 1000.0
    if "mm_preprocess_time" in mm_inputs:
        result["mm_preprocess_time(ms)"] = mm_inputs["mm_preprocess_time"] * 1000.0
    if "mm_process_time" in mm_inputs:
        result["mm_process_time(ms)"] = mm_inputs["mm_process_time"] * 1000.0
    if "mm_total_time" in mm_inputs:
        result["mm_total_time(ms)"] = mm_inputs["mm_total_time"] * 1000.0

    for item in mm_inputs.get("mm_items", []):
        result[item.modality.name] = {"offsets": item.offsets}

    return result
