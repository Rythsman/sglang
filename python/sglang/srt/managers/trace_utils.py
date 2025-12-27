from __future__ import annotations

import functools
import gzip
import json
import logging
import os
import threading
import time
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import psutil
import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)


try:
    import pynvml
except Exception:
    pynvml = None

# Constants
SAMPLE_INTERVAL_MS = int(os.environ.get("SGLANG_GPU_STATS_SAMPLE_INTERVAL_MS", 500))
SAMPLE_INTERVAL_SEC = SAMPLE_INTERVAL_MS / 1000.0
MICROSECONDS_PER_SECOND = 1_000_000.0
BYTES_TO_GIBIBYTE = 1024**3

_global_stats: Dict[int, Dict[str, Any]] = {}


class GPUStatsMonitor:
    """GPU status monitor that periodically samples GPU metrics

    Args:
        gpu_id: GPU device ID to monitor
    """

    def __init__(self, gpu_id: int, pid) -> None:
        if pynvml is None:
            self._available = False
            return

        self._pid = pid
        self._available = True
        self._events: List[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._gpu_id = gpu_id
        self._stop_event = threading.Event()

        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self._available = False

    def start(self) -> None:
        """Start the monitoring thread"""
        if not self._available:
            return

        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._ready_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=5.0)

    def stop(self) -> None:
        """Stop the monitoring thread"""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        # Wait outside the lock to avoid deadlock
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._ready_event is not None:
            self._ready_event.clear()

    def _sample_loop(self) -> None:
        """Background sampling loop with precise timing"""
        self._ready_event.set()
        next_tick = time.perf_counter()
        while self._running and not self._stop_event.is_set():
            next_tick += SAMPLE_INTERVAL_SEC

            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
                # torch related
                torch_allocated_gb = float(torch.cuda.memory_allocated(self._gpu_id))
                torch_reserved_gb = float(torch.cuda.memory_reserved(self._gpu_id))
                payload = {
                    "memory_total_gb": round(mem_info.total / BYTES_TO_GIBIBYTE, 2),
                    "memory_free_gb": round(mem_info.free / BYTES_TO_GIBIBYTE, 2),
                    "memory_used_gb": round(mem_info.used / BYTES_TO_GIBIBYTE, 2),
                    "torch_allocated_gb": round(
                        torch_allocated_gb / BYTES_TO_GIBIBYTE, 2
                    ),
                    "torch_reserved_gb": round(
                        torch_reserved_gb / BYTES_TO_GIBIBYTE, 2
                    ),
                    "gpu_util_p": util.gpu,
                    "memory_util_p": util.memory,
                    "temperature_c": temp,
                }

                event = _create_trace_event(
                    cat="gpu_stats",
                    name="gpu_stats",
                    ph="C",
                    pid=self._pid,
                    tid="gpu_stats",
                    ts=time.time(),
                    args=payload,
                )
                self._events.append(event)

            except Exception as e:
                logger.error(f"Error sampling GPU stats: {e}")

            # Precise interval control
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

    def take_events(self) -> list[dict]:
        """Retrieve and clear collected events"""
        if not self._available or self._running:
            return []

        events = self._events
        self._events = []
        return events

    def __enter__(self) -> "GPUStatsMonitor":
        self.start()
        return self

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Any
    ) -> None:
        self.stop()

    def __del__(self) -> None:
        """Safely cleanup NVML resources"""
        if not self._available or pynvml is None:
            return

        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


class TraceManager:
    """Chrome Trace event manager for performance profiling

    Args:
        prefix: Output filename prefix
        output_dir: Directory to save trace files
        tp_rank: Tensor parallel rank
        pp_rank: Pipeline parallel rank
        gpu_id: Target GPU device ID (optional for tokenizer manager)
    """

    def __init__(
        self,
        prefix: str,
        output_dir: str,
        tp_rank: int,
        pp_rank: int,
        gpu_id: Optional[int] = None,
    ) -> None:
        file_name = f"{prefix}custom_profiler.chrome_trace.json.gz"
        self._output_path = os.path.join(output_dir, file_name)
        self._events: list[dict] = []
        self.pid = f"{prefix}{psutil.Process().pid}"
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.gpu_id = gpu_id
        self._start_time = None
        self._enable = False

        self._gpu_monitor: Optional[GPUStatsMonitor] = None
        if gpu_id is not None:
            self._gpu_monitor = GPUStatsMonitor(gpu_id, self.pid)

        logger.info(f"Tracing events will be dumped to {self._output_path}")

    def start_profile(self) -> None:
        """Start performance profiling"""
        if self._gpu_monitor:
            self._gpu_monitor.start()
        self._start_time = time.time()
        self._enable = True

    def stop_profile(self) -> None:
        """Stop profiling and save results"""
        # Collect GPU monitoring data
        if self._gpu_monitor:
            self._gpu_monitor.stop()
            gpu_events = self._gpu_monitor.take_events()
            self._events.extend(gpu_events)

        # Add global stats event (only on rank 0)
        global _global_stats
        is_main_rank = self.tp_rank == 0 and self.pp_rank == 0
        if is_main_rank and self.gpu_id is not None:
            duration = (time.time() - self._start_time) * MICROSECONDS_PER_SECOND
            self._events.append(
                {
                    "cat": "global_stats",
                    "name": "global_stats",
                    "ts": self._start_time * MICROSECONDS_PER_SECOND,
                    "dur": duration,
                    "ph": "X",
                    "pid": "global_stats",
                    "tid": "global_stats",
                    "args": _global_stats,
                }
            )

        # Persist events to file
        self.dump_events()
        self._enable = False

    @property
    def enable(self) -> bool:
        """Check if tracing is enabled"""
        return self._enable

    def dump_events(self) -> None:
        """Write events to JSON file (compressed if .gz extension)"""
        if not self._events or not self._enable:
            return

        logger.info(f"Dumping {len(self._events)} events to {self._output_path}")

        try:
            json_str = json.dumps(self._events, indent=4, separators=(",", ":"))

            if self._output_path.endswith(".gz"):
                with gzip.open(self._output_path, "wt", encoding="utf-8") as f:
                    f.write(json_str)
            else:
                with open(self._output_path, "w", encoding="utf-8") as f:
                    f.write(json_str)

            logger.info("Successfully dumped events")
        except Exception as e:
            logger.error(f"Failed to dump events: {e}")
        finally:
            self._events = []

    def append(self, event: dict) -> None:
        """Append event to trace buffer"""
        self._events.append(event)


# Global TraceManager instance
_trace_manager: Optional[TraceManager] = None


def init_trace_manager(
    prefix: str,
    output_dir: str,
    tp_rank: int = 0,
    pp_rank: int = 0,
    gpu_id: Optional[int] = None,
) -> None:
    """Start performance profiling"""
    global _trace_manager
    if _trace_manager is None:
        _trace_manager = TraceManager(prefix, output_dir, tp_rank, pp_rank, gpu_id)


def start_trace_profile() -> None:
    """Start performance profiling"""
    global _trace_manager
    if _trace_manager is None:
        return
    _trace_manager.start_profile()


def stop_trace_profile() -> None:
    """Stop profiling and save results"""
    global _trace_manager
    if _trace_manager is None:
        return
    _trace_manager.stop_profile()


def get_trace_manager() -> Optional[TraceManager]:
    """Get the global TraceManager instance"""
    return _trace_manager


def trace_global_stats(stats: Dict[str, Any]) -> None:
    """Update global statistics

    Args:
        stats: Dictionary of stats to merge
    """
    _global_stats.update(stats)


class ReqTraceStatus(IntEnum):
    """Request tracing status enumeration"""

    PRE_SCHEDULER = auto()
    MM_PROCESS = auto()
    PRE_SCHEDULER_COMM = auto()
    SCHEDULER_BROADCAST = auto()
    SCHEDULER_WAITING = auto()
    SCHEDULER_PREFILL = auto()
    SCHEDULER_DECODE = auto()
    POST_SCHEDULER = auto()


def _should_trace_req() -> bool:
    """Check if request tracing is enabled on rank 0"""
    return (
        _trace_manager is not None
        and _trace_manager.enable
        and _trace_manager.tp_rank == 0
        and _trace_manager.pp_rank == 0
    )


def _create_trace_event(
    cat: str,
    name: str,
    ph: str,
    pid: Union[str, int],
    tid: str,
    ts: Optional[float] = None,
    dur: Optional[float] = None,
    args: Optional[Dict[Any, Any]] = None,
) -> dict:
    """Create a standardized trace event

    Args:
        cat: Event category
        name: Event name
        ph: Event phase (B/E/X/C)
        pid: Process ID
        tid: Thread ID
        ts: Timestamp in seconds (current time if None)
        dur: Duration in seconds (None for instant events)
        args: Event arguments dictionary
    """
    event = {
        "cat": cat,
        "name": name,
        "ph": ph,
        "pid": pid,
        "tid": tid,
        "args": args if args is not None else {},
    }
    event["ts"] = (time.time() if ts is None else ts) * MICROSECONDS_PER_SECOND
    if ph == "X":
        event["dur"] = dur * MICROSECONDS_PER_SECOND if dur is not None else None
    return event


def trace_req(
    rid: str,
    status: ReqTraceStatus,
    ph: str,
    ts: Optional[float] = None,
    extra_info: Optional[Dict[Any, Any]] = None,
) -> None:
    """Trace request event (internal)

    Args:
        rid: Request ID
        status: Request status
        ph: Event phase ("B" or "E")
        time_s: Timestamp in seconds
        extra_info: Additional metadata
    """
    if not _should_trace_req():
        return

    event = _create_trace_event(
        cat=f"req_{rid}",
        name=status.name,
        ph=ph,
        pid="ReqDetail",
        tid=rid,
        ts=ts,
        args=extra_info,
    )
    _trace_manager.append(event)


def trace_req_begin(
    rid: str,
    status: ReqTraceStatus,
    ts: Optional[float] = None,
    extra_info: Optional[Dict[Any, Any]] = None,
) -> None:
    """Mark the beginning of a request event

    Args:
        rid: Request ID
        status: Request status
        time_s: Timestamp in seconds (uses current time if None)
        extra_info: Additional metadata dictionary
    """
    trace_req(rid, status, "B", ts, extra_info)


def trace_req_end(
    rid: str,
    status: ReqTraceStatus,
    ts: Optional[float] = None,
    extra_info: Optional[Dict[Any, Any]] = None,
) -> None:
    """Mark the end of a request event

    Args:
        rid: Request ID
        status: Request status
        time_s: Timestamp in seconds (uses current time if None)
        extra_info: Additional metadata dictionary
    """
    trace_req(rid, status, "E", ts, extra_info)


class BatchTraceStatus(IntEnum):
    """Batch processing tracing status enumeration"""

    ENCODER = auto()
    PREFILL = auto()
    DECODE = auto()


def trace_batch(
    status: Union[ForwardMode, BatchTraceStatus, str],
    ph: str,
    extra_info: Optional[Dict[Any, Any]] = None,
    tid: str = "Default",
) -> None:
    """Trace batch event (internal)

    Args:
        status: Batch processing status
        ph: Event phase ("B" or "E")
        extra_info: Additional metadata
        tid: Thread ID
    """
    if _trace_manager is None or not _trace_manager.enable:
        return

    status_str = status if isinstance(status, str) else status.name
    event = _create_trace_event(
        cat=status_str,
        name=status_str,
        ph=ph,
        pid=_trace_manager.pid,
        tid=tid,
        ts=None,
        args=extra_info,
    )
    _trace_manager.append(event)


def trace_batch_begin(
    status: Union[ForwardMode, BatchTraceStatus, str],
    extra_info: Optional[Dict[Any, Any]] = None,
    tid: str = "Default",
) -> None:
    """Mark the beginning of a batch event

    Args:
        status: Batch processing status
        extra_info: Additional metadata dictionary
        tid: Thread ID for visualization
    """
    trace_batch(status, "B", extra_info, tid)


def trace_batch_end(
    status: Union[ForwardMode, BatchTraceStatus, str],
    extra_info: Optional[Dict[Any, Any]] = None,
    tid: str = "Default",
) -> None:
    """Mark the end of a batch event

    Args:
        status: Batch processing status
        extra_info: Additional metadata dictionary
        tid: Thread ID for visualization
    """
    trace_batch(status, "E", extra_info, tid)


def trace_execution_time(name: Optional[str] = None) -> Callable:
    """Decorator to trace function execution time

    Args:
        name: Function name for tracing
    """

    def decorator(func: Callable) -> Callable:
        event_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not envs.SGLANG_ENABLE_TRACE_EXECUTION_TIME.get():
                return func(*args, **kwargs)

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            global _trace_manager
            if _trace_manager is None or not _trace_manager.enable:
                return result

            event = _create_trace_event(
                cat=event_name,
                name=event_name,
                ph="X",
                ts=start_time,
                dur=end_time - start_time,
                pid=_trace_manager.pid,
                tid=event_name,
            )
            _trace_manager.append(event)
            return result

        return wrapper

    return decorator


def trace_usage(name: str, args: Dict[Any, Any], tid: str = "Usage") -> None:
    """Record instantaneous usage counter

    Args:
        name: Counter name
        args: Counter values dictionary
        tid: Thread ID for visualization
    """
    if not _should_trace_req():
        return

    event = _create_trace_event(
        cat=name,
        name=name,
        ph="C",
        pid=_trace_manager.pid,
        tid=tid,
        args=args,
    )
    _trace_manager.append(event)


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

    for item in mm_inputs["mm_items"]:
        result[item.modality.name] = {"offsets": item.offsets}

    return result
