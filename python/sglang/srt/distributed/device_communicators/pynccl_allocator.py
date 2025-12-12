import json
import os
import tempfile
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.utils.cpp_extension
from packaging import version
from torch.cuda.memory import CUDAPluggableAllocator

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.server_args import get_global_server_args

after_2_8_0 = version.parse(torch.__version__) >= version.parse("2.8.0")

nccl_allocator_source = """

#include <cuda_runtime.h>
#include <stdlib.h>

extern "C" {

// copy from https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclRemoteError             =  6,
               ncclInProgress              =  7,
               ncclNumResults              =  8 } ncclResult_t;
typedef struct ncclComm* ncclComm_t;
typedef struct ncclWindow_vidmem* ncclWindow_t;
ncclResult_t  ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags);
#define NCCL_WIN_COLL_SYMMETRIC 0x01

ncclResult_t  ncclMemAlloc(void** ptr, size_t size);
ncclResult_t  ncclMemFree(void *ptr);

void* nccl_alloc_plug(size_t size, int device, void* stream) {
  void* ptr;
  ncclResult_t err = ncclMemAlloc(&ptr, size);

  const char *str_val = getenv("SGLANG_TMP_NCCL_COMM_VALUE");
  if (str_val == NULL || str_val[0] == '\\0') {
    // Fallback: return the allocation without symmetric registration.
    // This avoids undefined behavior in strtoull and prevents hard hangs when
    // the env var is not set (e.g. mis-nested contexts).
    return ptr;
  }
  char *endptr;
  void* int_val = (void *)strtoull(str_val, &endptr, 0);
  if (endptr == str_val) {
    // Invalid value; skip registration.
    return ptr;
  }

  ncclComm_t comm = (ncclComm_t)(int_val);
  ncclWindow_t win;
  ncclResult_t err2 = ncclCommWindowRegister(comm, ptr, size, &win, NCCL_WIN_COLL_SYMMETRIC);

  return ptr;
}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
  ncclResult_t err = ncclMemFree(ptr);
}

}
"""

_allocator = None
_mem_pool = None
_graph_pool_id = None
_active_symmetric_memory_context = None


def is_symmetric_memory_enabled():
    return get_global_server_args().enable_symm_mem


def _should_enable_symmetric_memory_for_group(
    group_coordinator: GroupCoordinator, disabled: bool
) -> bool:
    """Return whether symmetric memory should be enabled for this group.

    Symmetric memory window registration in NCCL behaves like a collective on the
    communicator. When multiple communicators (e.g. TP and DCP) both attempt to
    register windows opportunistically via a general-purpose allocator, ordering
    differences can lead to NCCL failures.

    In practice, DCP is the most sensitive path (decode all-gather). To improve
    robustness in multi-group configurations, we default to enabling symmetric
    memory only for DCP when both TP and DCP are enabled.
    """
    if disabled or not is_symmetric_memory_enabled() or group_coordinator.world_size == 1:
        return False

    # Optional override: comma-separated group prefixes, e.g. "dcp,tp".
    allowlist = os.environ.get("SGLANG_SYMM_MEM_GROUPS")
    if allowlist is not None:
        allow = {s.strip() for s in allowlist.split(",") if s.strip()}
        # Empty allowlist means "disable everywhere".
        if not allow:
            return False
        group_prefix = group_coordinator.group_name.split(":")[0]
        return group_prefix in allow

    # Default heuristic: if both TP>1 and DCP>1, enable only for DCP.
    try:
        tp_size = int(getattr(get_global_server_args(), "tp_size", 1) or 1)
    except Exception:
        tp_size = 1
    try:
        dcp_size = int(os.getenv("SGLANG_DCP", "1") or "1")
    except Exception:
        dcp_size = 1
    if tp_size > 1 and dcp_size > 1:
        return group_coordinator.group_name.split(":")[0] == "dcp"

    return True


def set_graph_pool_id(graph_pool_id):
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def disable_symmetric_memory_context():
    if _active_symmetric_memory_context is None:
        return None
    saved_context = _active_symmetric_memory_context
    saved_context.__exit__(None, None, None)
    return saved_context


def restore_symmetric_memory_context(saved_context):
    if saved_context is not None:
        saved_context.__enter__()


def get_nccl_mem_pool():
    global _allocator, _mem_pool
    if _mem_pool is None:
        out_dir = tempfile.gettempdir()
        nccl_allocator_libname = "nccl_allocator"
        torch.utils.cpp_extension.load_inline(
            name=nccl_allocator_libname,
            cpp_sources=nccl_allocator_source,
            with_cuda=True,
            extra_ldflags=["-lnccl"],
            verbose=True,
            is_python_module=False,
            build_directory=out_dir,
        )
        _allocator = CUDAPluggableAllocator(
            f"{out_dir}/{nccl_allocator_libname}.so",
            "nccl_alloc_plug",
            "nccl_free_plug",
        ).allocator()
        _mem_pool = torch.cuda.MemPool(_allocator)
    return _mem_pool


# region agent log
def _agent_log(hypothesis_id: str, location: str, message: str, data: dict):
    try:
        payload = {
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(
            r"/home/wanghao44/code/sglang-dcp-2/deploy_tools/dco_dbg.log",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


# endregion


class SymmetricMemoryContext:
    """
    Context manager for using symmetric memory with pynccl.

    To Utilize the symmetric memory feature in NCCL, the buffers need to be allocated
    by `ncclMemAlloc` and registered by `ncclCommWindowRegister`. Due to this, we introduce
    this context manager. All tensors created under this context will be correctly
    allocated and registered with a custom allocator.
    """

    def __init__(
        self,
        group_coordinator: GroupCoordinator,
    ):
        self.group_coordinator = group_coordinator
        self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
        # NOTE: Determine capture mode at __enter__ time to reflect the actual
        # current stream context (graph capture switches streams).
        self.is_graph_capture = False
        self.exited = False
        self._prev_nccl_comm_env: Optional[str] = None
        self._had_prev_nccl_comm_env: bool = False

    def __enter__(self):
        assert (
            self.group_coordinator.pynccl_comm is not None
        ), f"Symmetric memory requires pynccl to be enabled in group '{self.group_coordinator.group_name}'"

        self.is_graph_capture = torch.cuda.is_current_stream_capturing()

        if self.is_graph_capture:
            assert (
                _graph_pool_id is not None
            ), "graph_pool_id is not set under graph capture"
            cur_device = torch.cuda.current_device()
            # Pause graph memory pool to use symmetric memory with cuda graph
            if after_2_8_0:
                torch._C._cuda_endAllocateToPool(cur_device, _graph_pool_id)
            else:
                torch._C._cuda_endAllocateCurrentStreamToPool(
                    cur_device, _graph_pool_id
                )

        if self.exited:
            # mempool ctx (@contextlib.contextmanager) is not re-entrant
            self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
            self.exited = False
        self._mem_pool_ctx.__enter__()

        # Set the env var to pass this argument to the C functions.
        self._prev_nccl_comm_env = os.environ.get("SGLANG_TMP_NCCL_COMM_VALUE")
        self._had_prev_nccl_comm_env = "SGLANG_TMP_NCCL_COMM_VALUE" in os.environ
        os.environ["SGLANG_TMP_NCCL_COMM_VALUE"] = str(
            self.group_coordinator.pynccl_comm.comm.value
        )

        global _active_symmetric_memory_context
        _active_symmetric_memory_context = self

        # region agent log
        _agent_log(
            hypothesis_id="H1",
            location="pynccl_allocator.py:SymmetricMemoryContext.__enter__",
            message="enter symmetric memory context",
            data={
                "group": getattr(self.group_coordinator, "unique_name", "unknown"),
                "rank": getattr(self.group_coordinator, "rank_in_group", -1),
                "world": getattr(self.group_coordinator, "world_size", -1),
                "is_graph_capture": self.is_graph_capture,
                "graph_pool_id": _graph_pool_id,
                "pynccl_present": self.group_coordinator.pynccl_comm is not None,
                "pynccl_disabled": getattr(
                    self.group_coordinator.pynccl_comm, "disabled", None
                ),
                "env_set_value": os.environ.get("SGLANG_TMP_NCCL_COMM_VALUE"),
            },
        )
        # endregion

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mem_pool_ctx.__exit__(exc_type, exc_val, exc_tb)

        if self.is_graph_capture:
            cur_device = torch.cuda.current_device()
            if after_2_8_0:
                torch._C._cuda_beginAllocateCurrentThreadToPool(
                    cur_device, _graph_pool_id
                )
            else:
                torch._C._cuda_beginAllocateToPool(cur_device, _graph_pool_id)

        # Restore env var to support nested/overlapping symmetric memory contexts.
        if self._had_prev_nccl_comm_env:
            assert self._prev_nccl_comm_env is not None
            os.environ["SGLANG_TMP_NCCL_COMM_VALUE"] = self._prev_nccl_comm_env
        else:
            os.environ.pop("SGLANG_TMP_NCCL_COMM_VALUE", None)

        global _active_symmetric_memory_context
        _active_symmetric_memory_context = None

        self.exited = True

        # region agent log
        _agent_log(
            hypothesis_id="H1",
            location="pynccl_allocator.py:SymmetricMemoryContext.__exit__",
            message="exit symmetric memory context",
            data={
                "group": getattr(self.group_coordinator, "unique_name", "unknown"),
                "rank": getattr(self.group_coordinator, "rank_in_group", -1),
                "is_graph_capture": self.is_graph_capture,
                "graph_pool_id": _graph_pool_id,
                "exc_type": str(exc_type) if exc_type else None,
            },
        )
        # endregion


def use_symmetric_memory(group_coordinator: GroupCoordinator, disabled: bool = False):
    # Avoid cross-group nesting. Registration is collective per communicator, and
    # nested/overlapping usage across different communicators is fragile.
    active = _active_symmetric_memory_context
    if (
        active is not None
        and getattr(active, "group_coordinator", None) is not None
        and active.group_coordinator.unique_name != group_coordinator.unique_name
    ):
        return nullcontext()

    enabled = _should_enable_symmetric_memory_for_group(group_coordinator, disabled)
    return SymmetricMemoryContext(group_coordinator) if enabled else nullcontext()
