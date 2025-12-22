import asyncio
import logging
import multiprocessing as mp
import pickle
import sys
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

_MP_MM_PROCESSOR: Any = None


def _mp_init_mm_processor(mm_processor_or_factory: Any) -> None:
    """Initialize a global mm_processor in each child process."""
    global _MP_MM_PROCESSOR
    _MP_MM_PROCESSOR = (
        mm_processor_or_factory()
        if callable(mm_processor_or_factory)
        else mm_processor_or_factory
    )


def _mp_process_mm_data(
    *,
    image_data: Optional[List[Union[str, bytes]]],
    audio_data: Optional[List[Union[str, bytes]]],
    input_text_or_ids: Union[str, List[int], None],
    request_obj: Any,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Run process_mm_data in a child process."""
    proc = _MP_MM_PROCESSOR
    if proc is None:
        raise RuntimeError("Multiprocess worker is not initialized.")
    sync_fn = getattr(proc, "process_mm_data", None)
    if not callable(sync_fn):
        raise RuntimeError("mm_processor does not expose callable 'process_mm_data'.")
    return sync_fn(
        image_data=image_data,
        audio_data=audio_data,
        input_text=input_text_or_ids,
        request_obj=request_obj,
        **kwargs,
    )


def _default_mp_start_method() -> str:
    # Linux default is "fork". For CUDA workloads, "spawn" is usually safer.
    return "fork" if sys.platform.startswith("linux") else "spawn"


class AsyncMMDataProcessor:
    """
    Async wrapper for a multimodal processor.

    Behavior:
      - If the underlying processor exposes `process_mm_data_async`, call/await it directly.
      - Otherwise, fall back to running a synchronous `process_mm_data` in an executor
        (thread pool by default; process pool optional for benchmarking).
      - Optionally guard per-call concurrency via an asyncio.Semaphore.
      - Optionally enforce per-call timeout via asyncio.wait_for.
    """

    def __init__(
        self,
        mm_processor: Any,
        *,
        max_concurrent_calls: Optional[int] = None,
        timeout_s: Optional[float] = None,
        executor_backend: str = "thread",
        mm_processor_factory: Optional[Callable[[], Any]] = None,
        mp_start_method: Optional[str] = None,
    ) -> None:
        """
        Args:
            mm_processor: An object exposing either
                - async def process_mm_data_async(...): -> Dict[str, Any]
              or
                - def process_mm_data(...): -> Dict[str, Any]
            max_concurrent_calls: Optional concurrency cap for per-call execution.
            timeout_s: Optional timeout (seconds) for each `process()` call.
            executor_backend: Fallback executor backend for sync path.
                Supported values: "thread" (default), "process".
            mm_processor_factory: Optional zero-arg callable to build a fresh
                mm_processor inside each child process when executor_backend="process".
                The callable itself must be picklable (avoid lambdas / nested defs).
            mp_start_method: multiprocessing start method for process backend
                (e.g., "fork", "spawn", "forkserver"). Defaults to platform choice.
        """
        self.mm_processor = mm_processor
        self.timeout_s = timeout_s
        self.executor_backend = executor_backend
        self.mm_processor_factory = mm_processor_factory
        self.mp_start_method = mp_start_method or _default_mp_start_method()

        # Concurrency guard (None -> unlimited)
        self.semaphore = (
            asyncio.Semaphore(max_concurrent_calls) if max_concurrent_calls else None
        )

        # Detect async path; if missing, prepare a fallback executor for sync path
        self._proc_async = getattr(mm_processor, "process_mm_data_async", None)
        self.is_async = asyncio.iscoroutinefunction(self._proc_async)
        self.fallback_exec: Optional[Executor] = None
        if not self.is_async:
            if self.executor_backend == "thread":
                self.fallback_exec = ThreadPoolExecutor(max_workers=max_concurrent_calls)
            elif self.executor_backend == "process":
                init_arg = self.mm_processor_factory or self.mm_processor
                try:
                    # Fail fast when init args are not picklable.
                    pickle.dumps(init_arg)
                except Exception as e:
                    raise ValueError(
                        "executor_backend='process' requires mm_processor_factory (preferred) "
                        "or mm_processor to be picklable. Also ensure request_obj/kwargs are "
                        "picklable per call."
                    ) from e
                ctx = mp.get_context(self.mp_start_method)
                self.fallback_exec = ProcessPoolExecutor(
                    max_workers=max_concurrent_calls,
                    mp_context=ctx,
                    initializer=_mp_init_mm_processor,
                    initargs=(init_arg,),
                )
            else:
                raise ValueError(
                    f"Unsupported executor_backend={self.executor_backend!r}. "
                    "Expected 'thread' or 'process'."
                )

    async def process(
        self,
        *,
        image_data: Optional[List[Union[str, bytes]]] = None,
        audio_data: Optional[List[Union[str, bytes]]] = None,
        input_text_or_ids: Union[str, List[int], None] = None,
        request_obj: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Public entrypoint: process a single multimodal request without blocking the event loop.
        """

        async def _invoke() -> Dict[str, Any]:
            if self.is_async:
                # Native async implementation
                return await self._proc_async(
                    image_data=image_data,
                    audio_data=audio_data,
                    input_text=input_text_or_ids,
                    request_obj=request_obj,
                    **kwargs,
                )

            # Synchronous fallback
            sync_fn = getattr(self.mm_processor, "process_mm_data", None)
            if not callable(sync_fn):
                raise RuntimeError(
                    "mm_processor has neither 'process_mm_data_async' nor 'process_mm_data'."
                )
            loop = asyncio.get_running_loop()
            if self.executor_backend == "process":
                fn = partial(
                    _mp_process_mm_data,
                    image_data=image_data,
                    audio_data=audio_data,
                    input_text_or_ids=input_text_or_ids,
                    request_obj=request_obj,
                    kwargs=kwargs,
                )
            else:
                fn = partial(
                    sync_fn,
                    image_data=image_data,
                    audio_data=audio_data,
                    input_text=input_text_or_ids,
                    request_obj=request_obj,
                    **kwargs,
                )
            return await loop.run_in_executor(self.fallback_exec, fn)

        # Apply optional concurrency guard
        if self.semaphore is not None:
            async with self.semaphore:
                if self.timeout_s is not None:
                    return await asyncio.wait_for(_invoke(), timeout=self.timeout_s)
                return await _invoke()

        # No concurrency guard
        if self.timeout_s is not None:
            return await asyncio.wait_for(_invoke(), timeout=self.timeout_s)
        return await _invoke()

    def shutdown(self) -> None:
        """Gracefully shutdown resources owned by this wrapper."""
        try:
            if self.fallback_exec:
                self.fallback_exec.shutdown(wait=False)
        except Exception:
            logger.exception(
                "Error while shutting down fallback executor in AsyncMMDataProcessor"
            )

    def __del__(self):
        # Best-effort shutdown
        try:
            self.shutdown()
        except Exception:
            pass
