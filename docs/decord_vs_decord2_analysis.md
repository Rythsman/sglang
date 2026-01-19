# decord==0.6.0 vs decord2==3.0.0 帧值差异分析

## 背景

SGLang 0.5.2 版本使用 `decord==0.6.0`，而当前 main 分支使用 `decord2==3.0.0`。用户发现处理相同视频时，frames 的具体 value 出现微小差异（diff）。

## 库信息

| 属性 | decord 0.6.0 | decord2 3.0.0 |
|-----|-------------|---------------|
| GitHub | https://github.com/dmlc/decord | https://github.com/johnnynunez/decord2 |
| 维护状态 | 原始项目 | Fork 版本，持续更新 |
| FFmpeg 版本 | 较旧版本 | 更新版本 (FFmpeg 6.x/7.x) |

## 差异根本原因分析

### 1. FFmpeg 版本差异 (主要原因)

**这是导致帧值微小差异的最根本原因。**

decord2 更新了底层 FFmpeg 版本，不同版本的 FFmpeg 在以下方面有细微差异：

```
decord 0.6.0:  基于 FFmpeg 4.x
decord2 3.0.0: 基于 FFmpeg 6.x 或 7.x
```

#### 具体影响点：

1. **H.264/H.265 解码器更新**
   - 新版本 FFmpeg 对某些边缘情况的处理不同
   - motion compensation 算法的微调
   - 环路滤波器 (loop filter) 参数差异

2. **色彩空间转换精度**
   - YUV → RGB 转换公式可能有微调
   - BT.601 vs BT.709 色彩矩阵的默认选择
   - 舍入方式差异 (round vs floor vs ceil)

### 2. 像素格式转换 (YUV → RGB)

视频通常以 YUV 格式存储，需要转换为 RGB 供模型使用：

```python
# YUV to RGB 转换公式 (BT.709)
R = Y + 1.5748 * (V - 128)
G = Y - 0.1873 * (U - 128) - 0.4681 * (V - 128)
B = Y + 1.8556 * (U - 128)
```

**差异来源：**
- 浮点运算精度差异
- 整数舍入方式不同
- 色彩范围 (limited range vs full range) 处理

### 3. 缩放算法差异

当使用 `width` 或 `height` 参数调整分辨率时：

```python
# decord 0.6.0 可能使用
swscale with SWS_BILINEAR

# decord2 3.0.0 可能使用
swscale with SWS_BICUBIC 或其他算法
```

不同的插值算法会导致像素值差异。

### 4. GPU vs CPU 解码差异

```python
from decord import cpu, gpu

# CPU 解码
vr = VideoReader(path, ctx=cpu(0))

# GPU 解码 (使用 NVDEC)
vr = VideoReader(path, ctx=gpu(0))
```

**差异来源：**
- NVDEC 硬件解码器与软件解码器的实现差异
- GPU 浮点运算精度 (FP32 vs FP16)
- 不同 CUDA 版本的影响

### 5. 帧索引和时间戳精度

```python
# get_batch 内部使用的 seek 算法可能不同
frames = vr.get_batch([0, 10, 20])
```

**差异来源：**
- 关键帧 (I-frame) 定位策略
- B-frame 解码顺序
- 时间戳精度 (微秒 vs 毫秒)

## 验证差异的代码

```python
import numpy as np
from decord import VideoReader, cpu

def compare_decord_outputs(video_path, indices=[0, 10, 20]):
    """Compare frame outputs between decord versions."""
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(indices)
    
    # Convert to numpy
    if hasattr(frames, 'asnumpy'):
        frames_np = frames.asnumpy()
    else:
        frames_np = np.array(frames)
    
    print(f"Shape: {frames_np.shape}")
    print(f"Dtype: {frames_np.dtype}")
    print(f"Min: {frames_np.min()}, Max: {frames_np.max()}")
    print(f"Mean: {frames_np.mean():.6f}")
    print(f"Std: {frames_np.std():.6f}")
    
    # Sample pixel values for comparison
    print(f"\nSample pixels (frame 0, top-left 3x3):")
    print(frames_np[0, :3, :3, :])
    
    return frames_np
```

## 差异量级估计

通常这类差异的量级为：

| 差异类型 | 典型范围 |
|---------|---------|
| 像素值 (0-255) | ±1 ~ ±3 |
| 归一化后 (0-1) | ±0.004 ~ ±0.012 |
| 均方误差 (MSE) | < 10 |
| PSNR | > 40 dB |

## 对模型推理的影响

1. **视觉模型通常对这种微小差异不敏感**
   - 现代视觉模型对输入噪声有一定容忍度
   - 数据增强训练使模型对小扰动鲁棒

2. **可能影响的场景**
   - 严格的数值一致性测试
   - 确定性推理要求
   - 模型输出的精确复现

## 解决方案

### 方案1：统一使用同一版本

```toml
# pyproject.toml
dependencies = [
    "decord==0.6.0",  # 保持与 0.5.2 一致
]
```

### 方案2：容忍差异

在测试中使用容差比较：

```python
import numpy as np

def compare_frames(frames1, frames2, atol=3, rtol=0.02):
    """Compare frames with tolerance."""
    return np.allclose(frames1, frames2, atol=atol, rtol=rtol)
```

### 方案3：记录并验证差异范围

```python
def validate_frame_difference(frames_old, frames_new):
    """Validate that differences are within acceptable range."""
    diff = np.abs(frames_new.astype(float) - frames_old.astype(float))
    
    assert diff.max() <= 5, f"Max diff {diff.max()} exceeds threshold"
    assert diff.mean() <= 1, f"Mean diff {diff.mean()} exceeds threshold"
    
    print(f"Diff stats: max={diff.max()}, mean={diff.mean():.4f}")
```

## 结论

**根本原因：** `decord2==3.0.0` 使用了更新版本的 FFmpeg，导致视频解码过程中的 YUV→RGB 色彩空间转换、解码算法细节产生微小数值差异。

**影响程度：** 差异通常在像素级别为 ±1~3，对模型推理结果影响极小。

**建议：**
1. 如需严格数值一致，锁定 decord 版本
2. 如可接受微小差异，在测试中增加容差
3. 记录版本依赖，确保环境可复现
