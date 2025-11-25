# start_profile 后启用 cuda_graph 的代码运行逻辑分析

## 概述

当调用 `start_profile` 启动性能分析后，如果系统启用了 `cuda_graph`，整体的代码执行流程会与常规执行有所不同。本文档详细分析这一流程。

## 关键组件

### 1. Profiler 启动 (`start_profile`)

**位置**: `python/sglang/srt/managers/scheduler_profiler_mixin.py`

```python
def start_profile(self, stage: Optional[ForwardMode] = None):
    # 创建 torch.profiler.profile 对象
    self.torch_profiler = torch.profiler.profile(
        activities=torchprof_activities,
        with_stack=with_stack,
        record_shapes=record_shapes,
    )
    self.torch_profiler.start()  # 启动 profiler
    self.profile_in_progress = True
```

**作用**: 
- 启动 PyTorch 的性能分析器
- 记录 CPU 和 CUDA 活动
- 设置 `profile_in_progress` 标志

### 2. CUDA Graph 初始化

**位置**: `python/sglang/srt/model_executor/cuda_graph_runner.py`

在 `ModelRunner.init_device_graphs()` 中，会创建 `CudaGraphRunner` 实例：

```python
def init_device_graphs(self):
    # 创建 CudaGraphRunner
    self.graph_runner = CudaGraphRunner(self)
```

**CUDA Graph 捕获过程** (`CudaGraphRunner.capture()`):

1. **准备阶段**:
   - 冻结垃圾回收 (`freeze_gc`)
   - 进入 graph capture 上下文 (`graph_capture()`)
   - 创建 CUDA stream

2. **对每个 batch size 进行捕获**:
   ```python
   for bs in capture_range:
       with patch_model(...) as forward:
           graph, output_buffers = self.capture_one_batch_size(bs, forward)
           self.graphs[bs] = graph
           self.output_buffers[bs] = output_buffers
   ```

3. **单个 batch size 的捕获** (`capture_one_batch_size`):
   - 准备输入张量（input_ids, positions, seq_lens 等）
   - 创建 `ForwardBatch` 对象
   - **预热运行** (`run_once()` 执行 2 次)
   - **捕获 CUDA graph**:
     ```python
     with self.device_module.graph(graph, pool=pool, stream=stream):
         out = run_once_fn()  # 捕获这个执行序列
     ```

## 执行流程

### 正常执行路径（无 profiling）

```
Scheduler.run_batch()
  └─> ModelWorker.forward_batch_generation()
      └─> ModelRunner.forward()
          └─> ModelRunner._forward_raw()
              ├─> 检查 can_run_graph
              ├─> 如果可以: CudaGraphRunner.replay()
              │   ├─> replay_prepare()  # 准备输入数据
              │   └─> graph.replay()     # 直接 replay CUDA graph
              └─> 如果不行: 执行常规 forward (forward_decode/forward_extend)
```

### 启用 profiling 后的执行路径

```
Scheduler.run_batch()
  └─> _profile_batch_predicate(batch)  # 检查是否需要启动/停止 profiling
      └─> start_profile()  # 如果满足条件，启动 profiler
  └─> ModelWorker.forward_batch_generation()
      └─> ModelRunner.forward()
          └─> ModelRunner._forward_raw()
              ├─> 检查 can_run_graph
              ├─> 如果可以: CudaGraphRunner.replay()
              │   ├─> replay_prepare()  # 准备输入数据
              │   └─> graph.replay()   # ⚠️ 关键：直接 replay CUDA graph
              └─> 如果不行: 执行常规 forward
```

## 关键代码路径分析

### 1. 判断是否使用 CUDA Graph

**位置**: `python/sglang/srt/model_executor/model_runner.py:2011-2028`

```python
def _forward_raw(self, forward_batch, ...):
    # 检查是否可以运行 CUDA graph
    mode_check = (
        forward_batch.forward_mode.is_cuda_graph
        if self.device == "cuda"
        else forward_batch.forward_mode.is_cpu_graph
    )
    can_run_graph = bool(
        mode_check()
        and self.graph_runner
        and self.graph_runner.can_run(forward_batch)
    )
    
    if can_run_graph:
        # 使用 CUDA graph replay
        ret = self.graph_runner.replay(
            forward_batch,
            skip_attn_backend_init=skip_attn_backend_init,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return ret, can_run_graph
    else:
        # 执行常规 forward
        ...
```

### 2. CUDA Graph Replay

**位置**: `python/sglang/srt/model_executor/cuda_graph_runner.py:794-822`

```python
def replay(self, forward_batch, ...):
    # 准备输入数据
    if not skip_attn_backend_init:
        self.replay_prepare(forward_batch, pp_proxy_tensors)
    
    # ⚠️ 关键：直接 replay 预捕获的 CUDA graph
    # 这里不会执行 Python 的 forward 函数，而是直接执行 GPU kernel 序列
    self.graphs[self.bs].replay()
    
    # 从预分配的 output_buffers 中提取结果
    output = self.output_buffers[self.bs]
    return LogitsProcessorOutput(...)
```

**`replay_prepare` 的作用**:
- 将 `forward_batch` 的数据复制到预分配的 CUDA graph 输入缓冲区
- 处理 batch size padding（如果需要）
- 初始化 attention backend 的 metadata

### 3. Profiling 与 CUDA Graph 的交互

**重要发现**: 

当使用 CUDA graph replay 时，**profiler 仍然可以记录 GPU 活动**，因为：

1. **CUDA Graph Replay 仍然执行 GPU kernels**: 
   - `graph.replay()` 会执行预捕获的 GPU kernel 序列
   - PyTorch profiler 可以记录这些 GPU kernel 的执行

2. **CPU 侧的 Python 代码减少**:
   - CUDA graph replay 跳过了 Python 的 forward 函数执行
   - 只有 `replay_prepare()` 和 `graph.replay()` 的调用
   - 这会导致 CPU 侧的 profiling 信息较少

3. **Profiler 记录的内容**:
   - GPU kernel 执行时间（完整记录）
   - CPU 侧的操作（主要是数据准备和 graph replay 调用）
   - 内存操作（如果启用了 MEM profiling）

## CUDA Graph 的优势

1. **减少 CPU-GPU 同步**: 
   - 预编译的 kernel 序列，减少启动开销
   - 减少 Python 解释器开销

2. **性能提升**:
   - 跳过 Python forward 函数调用
   - GPU kernel 执行更高效

3. **Profiling 的影响**:
   - GPU 侧 profiling 信息完整
   - CPU 侧信息较少（因为跳过了 forward 函数）

## 注意事项

1. **CUDA Graph 的限制**:
   - 只能捕获固定形状的操作
   - 需要满足 `can_run()` 的条件（batch size, encoder_lens 等）

2. **Profiling 时的行为**:
   - 如果 batch 满足 CUDA graph 条件，会使用 graph replay
   - 如果不满足，会回退到常规 forward（此时 profiling 会记录完整的 Python 调用栈）

3. **内存管理**:
   - CUDA graph 使用预分配的内存池
   - `replay_prepare()` 将数据复制到这些预分配的缓冲区

## 总结

启用 `cuda_graph` 后，`start_profile` 的执行流程：

1. **Profiler 启动**: 在 `_profile_batch_predicate` 中根据条件启动
2. **执行路径选择**: `ModelRunner._forward_raw()` 检查是否可以使用 CUDA graph
3. **CUDA Graph Replay**: 如果可用，直接 replay 预捕获的 graph，跳过 Python forward
4. **Profiling 记录**: GPU kernel 执行被完整记录，但 CPU 侧信息较少

这种设计在保持高性能的同时，仍然允许对 GPU 执行进行详细的性能分析。
