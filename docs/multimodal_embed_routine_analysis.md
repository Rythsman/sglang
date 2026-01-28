# SGLang 多模态嵌入处理流程深度解析

## 一、核心函数调用链概览

```
general_mm_embed_routine()                 # 入口函数
    │
    ├── embed_tokens = language_model.get_input_embeddings()
    │
    ├── [条件判断] forward_mode 是否为 extend/prefill 且包含多模态输入
    │       │
    │       └── embed_mm_inputs()              # 多模态嵌入主函数
    │               │
    │               ├── 遍历每种模态 (IMAGE, VIDEO, AUDIO)
    │               │       │
    │               │       └── get_embedding_and_mask()
    │               │               │
    │               │               ├── _get_precomputed_embedding()  # 尝试获取预计算嵌入
    │               │               │
    │               │               └── _get_chunked_prefill_embedding()  # 分块预填充嵌入
    │               │                       │
    │               │                       ├── embedding_cache.get()  # 尝试从缓存获取
    │               │                       │
    │               │                       ├── data_embedding_func()  # 调用 ViT 计算嵌入
    │               │                       │
    │               │                       ├── embedding_cache.set()  # 缓存嵌入
    │               │                       │
    │               │                       └── get_embedding_chunk()  # 提取当前 chunk 的嵌入
    │               │
    │               ├── input_embedding(input_ids)  # 获取文本 token 嵌入
    │               │
    │               └── input_embeds[indices] = embedding  # 替换多模态位置的嵌入
    │
    └── language_model(input_embeds=input_embeds)  # 语言模型前向传播
```

## 二、核心数据结构详解

### 2.1 ForwardBatch 中的多模态相关字段

```python
@dataclass
class ForwardBatch:
    # 基础信息
    forward_mode: ForwardMode        # 前向模式: EXTEND, DECODE, MIXED 等
    batch_size: int                  # 批次大小
    input_ids: torch.Tensor          # 输入 token IDs, shape: (total_tokens,)
    
    # extend/prefill 相关
    extend_prefix_lens: torch.Tensor      # 每个请求的前缀长度 (已缓存部分)
    extend_seq_lens: torch.Tensor         # 每个请求当前 chunk 的长度
    extend_prefix_lens_cpu: List[int]     # CPU 版本
    extend_seq_lens_cpu: List[int]        # CPU 版本
    
    # 多模态输入
    mm_inputs: List[MultimodalInputs]  # 每个请求的多模态输入列表
```

### 2.2 MultimodalInputs 结构

```python
@dataclass
class MultimodalInputs:
    mm_items: List[MultimodalDataItem]  # 多模态数据项列表
    
    # Token ID 信息
    im_token_id: Optional[int]      # 图像占位符 token ID
    im_start_id: Optional[int]      # 图像开始 token ID
    im_end_id: Optional[int]        # 图像结束 token ID
    video_token_id: Optional[int]   # 视频占位符 token ID
    audio_token_id: Optional[int]   # 音频占位符 token ID
    
    # Qwen-VL 特有
    mrope_positions: Optional[torch.Tensor]  # 多维旋转位置编码
```

### 2.3 MultimodalDataItem 结构

```python
@dataclass
class MultimodalDataItem:
    modality: Modality              # 模态类型: IMAGE, VIDEO, AUDIO
    hash: int                       # 数据哈希值 (用于缓存)
    pad_value: int                  # 占位符值 = hash % 2^30
    offsets: List[Tuple[int, int]]  # 在 input_ids 中的位置范围列表
    
    feature: torch.Tensor           # 原始特征 (如 pixel_values)
    precomputed_embeddings: torch.Tensor  # 预计算的嵌入
    
    model_specific_data: dict       # 模型特定数据 (如 image_grid_thw)
```

## 三、函数详细解析

### 3.1 general_mm_embed_routine - 入口函数

**位置**: `python/sglang/srt/managers/mm_utils.py:1045-1136`

```python
def general_mm_embed_routine(
    input_ids: torch.Tensor,           # 输入 token IDs
    forward_batch: ForwardBatch,       # 批次信息
    language_model: nn.Module,         # 语言模型
    multimodal_model: nn.Module,       # 多模态编码器 (ViT等)
    data_embedding_funcs: Dict[Modality, DataEmbeddingFunc],  # 嵌入函数映射
    placeholder_tokens: Dict[Modality, List[int]],  # 占位符 tokens
    use_deepstack: Dict[Modality, bool],  # 是否使用 deepstack
    **kwargs,
) -> torch.Tensor:
```

**核心逻辑**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     general_mm_embed_routine                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. 获取文本嵌入层: embed_tokens = language_model.get_input_embeddings() │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. 判断是否需要处理多模态:                                          │
│     - forward_mode 不是 DECODE                                       │
│     - forward_mode 不是 TARGET_VERIFY                                │
│     - forward_batch.contains_mm_inputs() 为 True                     │
└─────────────────────────────────────────────────────────────────────┘
                    │                           │
              [需要处理]                    [不需要处理]
                    │                           │
                    ▼                           ▼
┌─────────────────────────────┐   ┌─────────────────────────────────┐
│  3a. 调用 embed_mm_inputs   │   │  3b. 直接调用 embed_tokens       │
│      获取混合嵌入            │   │      input_embeds = embed_tokens │
│                             │   │                    (input_ids)   │
└─────────────────────────────┘   └─────────────────────────────────┘
                    │                           │
                    └───────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. 调用语言模型: hidden_states = language_model(input_embeds=...)    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 embed_mm_inputs - 多模态嵌入主函数

**位置**: `python/sglang/srt/managers/mm_utils.py:909-1043`

```python
def embed_mm_inputs(
    mm_inputs_list: List[MultimodalInputs],  # 多模态输入列表
    extend_prefix_lens: List[int],           # 每个请求的前缀长度
    extend_seq_lens: List[int],              # 每个请求的序列长度
    input_ids: torch.Tensor,                 # 输入 token IDs
    input_embedding: nn.Embedding,           # 文本嵌入层
    multimodal_model: nn.Module,             # 多模态模型
    data_embedding_func_mapping: Dict[Modality, DataEmbeddingFunc],
    placeholder_tokens: dict[Modality, List[int]],
    use_deepstack: Dict[Modality, bool],
) -> Tuple[torch.Tensor, dict]:
```

**核心逻辑图**:

```
输入数据示例 (2个请求, 第一个有1张图, 第二个有2张图):
─────────────────────────────────────────────────────────────────

mm_inputs_list = [
    MultimodalInputs(
        mm_items=[
            MultimodalDataItem(modality=IMAGE, offsets=[(10, 585)], ...)
        ]
    ),
    MultimodalInputs(
        mm_items=[
            MultimodalDataItem(modality=IMAGE, offsets=[(5, 580), (590, 1165)], ...)
        ]
    )
]

extend_prefix_lens = [0, 100]      # 第一个请求从头开始, 第二个已有100个缓存token
extend_seq_lens = [600, 1100]     # 当前chunk长度

input_ids = [tok1, tok2, ..., img_pad, img_pad, ..., tok_n, ...]
            │                  │                        │
            └──────────────────┴────────────────────────┘
                        Total: 1700 tokens

处理流程:
─────────────────────────────────────────────────────────────────

Step 1: 展平所有多模态项
        item_flatten_list = [img_item_1, img_item_2, img_item_3]

Step 2: 按模态分组并计算嵌入
        ┌─────────────────────────────────────────┐
        │  modality = IMAGE                       │
        │  items = [img_item_1, img_item_2, img_item_3] │
        │                                         │
        │  计算 items_size = [0, 1, 3]           │
        │  表示: 请求1有1个图, 请求2有2个图       │
        │                                         │
        │  调用 get_embedding_and_mask()         │
        └─────────────────────────────────────────┘

Step 3: 获取文本嵌入并替换多模态位置
        ┌─────────────────────────────────────────┐
        │  input_embeds = embed_tokens(input_ids) │
        │  Shape: (1700, hidden_size)             │
        │                                         │
        │  对于每个 (embedding, mask):           │
        │    indices = torch.where(mask)[0]      │
        │    input_embeds[indices] = embedding    │
        └─────────────────────────────────────────┘
```

### 3.3 get_embedding_and_mask - 获取嵌入和掩码

**位置**: `python/sglang/srt/managers/mm_utils.py:855-906`

```python
def get_embedding_and_mask(
    data_embedding_func: DataEmbeddingFunc,   # 嵌入计算函数 (如 ViT)
    embedding_items: List[MultimodalDataItem], # 待嵌入的数据项
    placeholder_tensor: torch.Tensor,          # 占位符值张量
    input_ids: torch.Tensor,                   # 输入 token IDs
    items_size: List[int],                     # 每个请求的累计项数
    prefix_length: List[int],                  # 前缀长度
    extend_length: List[int],                  # 扩展长度
    items_offset_list: List[List[Tuple[int, int]]],  # 偏移列表
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

**流程解析**:

```
                    get_embedding_and_mask
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌──────────────────────┐      ┌──────────────────────┐
│ _get_precomputed     │      │ precomputed          │
│ _embedding()         │      │ embedding 不存在     │
│                      │      │                      │
│ 检查是否有预计算嵌入   │      │         │            │
└──────────┬───────────┘      └─────────┬────────────┘
           │                            │
     [存在预计算]                  [需要计算]
           │                            │
           ▼                            ▼
┌──────────────────────┐      ┌──────────────────────┐
│ 直接返回预计算嵌入    │      │ _get_chunked_prefill │
│                      │      │ _embedding()         │
│                      │      │                      │
│                      │      │ 调用 ViT 计算嵌入    │
└──────────────────────┘      └──────────────────────┘
           │                            │
           └───────────┬────────────────┘
                       ▼
           ┌──────────────────────┐
           │ _get_multimodal_mask │
           │                      │
           │ 创建布尔掩码,标识     │
           │ input_ids 中哪些位置  │
           │ 是多模态占位符        │
           │                      │
           │ mask = isin(input_ids, │
           │        placeholder_tensor) │
           └──────────────────────┘
                       │
                       ▼
           ┌──────────────────────┐
           │ _adjust_embedding    │
           │ _length()            │
           │                      │
           │ 调整嵌入长度以匹配   │
           │ 掩码中的位置数       │
           └──────────────────────┘
```

### 3.4 _get_chunked_prefill_embedding - 分块预填充嵌入

**位置**: `python/sglang/srt/managers/mm_utils.py:548-612`

这是最复杂的函数,处理分块预填充场景下的多模态嵌入计算。

```python
def _get_chunked_prefill_embedding(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
```

**分块预填充示例**:

```
假设有一个很长的请求,包含一张大图片:
─────────────────────────────────────────────────────────────────

原始序列: [text_tokens...][image_tokens x 576][more_text...]
总长度: 2000 tokens, 其中图片占位符位于 [500, 1075]

chunked_prefill_size = 512 (每次最多处理512个token)

Chunk 1: [0, 512)
  - prefix_len = 0
  - extend_len = 512
  - 图片区间 [500, 1075] 与 [0, 512) 有交集 [500, 512)
  - 需要计算图片嵌入,但只提取 [500, 512) 对应的部分

Chunk 2: [512, 1024)
  - prefix_len = 512
  - extend_len = 512
  - 图片区间 [500, 1075] 与 [512, 1024) 有交集 [512, 1024)
  - 从缓存获取嵌入,提取 [512, 1024) 对应的部分

Chunk 3: [1024, 1536)
  - prefix_len = 1024
  - extend_len = 512
  - 图片区间 [500, 1075] 与 [1024, 1536) 有交集 [1024, 1075]
  - 从缓存获取嵌入,提取 [1024, 1075] 对应的部分

Chunk 4: [1536, 2000)
  - prefix_len = 1536
  - extend_len = 464
  - 图片区间 [500, 1075] 与 [1536, 2000) 无交集
  - 不需要图片嵌入

缓存机制:
─────────────────────────────────────────────────────────────────

embedding_cache (MultiModalStaticCache):
  - key: hash(所有图片item的hash组合)
  - value: EmbeddingResult(embedding=完整的图片嵌入)
  
  Chunk 1 计算嵌入后存入缓存
  Chunk 2, 3 从缓存获取,避免重复计算 ViT
```

### 3.5 get_embedding_chunk - 提取嵌入分片

**位置**: `python/sglang/srt/managers/mm_utils.py:380-422`

```python
def get_embedding_chunk(
    embedding: torch.Tensor,           # 完整嵌入
    extend_prefix_len: int,            # 当前chunk的前缀长度
    extend_seq_len: int,               # 当前chunk的长度
    items_offset: List[Tuple[int, int]], # 多模态区间列表
) -> Tuple[torch.Tensor, int, int]:
```

**详细计算示例**:

```
假设:
  embedding.shape = (576, 4096)  # 576个图片token, 4096维嵌入
  extend_prefix_len = 512        # 当前chunk从位置512开始
  extend_seq_len = 512           # 当前chunk长度512
  items_offset = [(500, 1075)]   # 图片在位置500-1075

计算过程:
─────────────────────────────────────────────────────────────────

extend_start_index = 512 (chunk开始位置)
extend_end_index = 512 + 512 - 1 = 1023 (chunk结束位置)

对于 items_offset 中的每个 (start=500, end=1075):

  start_index 计算:
    extend_start_index=512 在区间 [500, 1075] 内
    start_index += 512 - 500 = 12
    
  end_index 计算:
    extend_end_index=1023 在区间 [500, 1075] 内
    end_index += 1023 - 500 + 1 = 524

结果:
  embedding_chunk = embedding[12:524]  # shape: (512, 4096)
  
  这正好对应当前chunk中需要的图片嵌入部分
```

## 四、完整数据流动示意图

```
                           用户请求
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TokenizerManager                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 输入:                                                    │    │
│  │   text = "<image>描述这张图片"                           │    │
│  │   image_data = [PIL.Image]                               │    │
│  │                                                          │    │
│  │ 处理:                                                    │    │
│  │   1. mm_processor.process_mm_data_async()                │    │
│  │   2. 生成 input_ids = [im_start, pad, pad, ..., im_end, text...] │
│  │   3. 生成 mm_items = [MultimodalDataItem(...)]           │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Scheduler                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 创建 Req 对象:                                          │    │
│  │   req.origin_input_ids = input_ids                       │    │
│  │   req.multimodal_inputs = MultimodalInputs(mm_items)     │    │
│  │                                                          │    │
│  │ 调度:                                                    │    │
│  │   - 匹配 RadixCache 前缀                                 │    │
│  │   - 确定 extend_prefix_len, extend_seq_len              │    │
│  │   - 创建 ScheduleBatch                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TpModelWorker                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 创建 ModelWorkerBatch:                                  │    │
│  │   batch.input_ids = torch.tensor([...])                  │    │
│  │   batch.multimodal_inputs = [MultimodalInputs(...)]      │    │
│  │   batch.extend_prefix_lens = [...]                       │    │
│  │   batch.extend_seq_lens = [...]                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ModelRunner                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 创建 ForwardBatch:                                       │    │
│  │   forward_batch.input_ids = tensor([1, 2, PAD, PAD,...]) │    │
│  │   forward_batch.mm_inputs = [...]                        │    │
│  │   forward_batch.extend_prefix_lens_cpu = [0]             │    │
│  │   forward_batch.extend_seq_lens_cpu = [600]              │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model.forward()                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 调用 general_mm_embed_routine():                         │    │
│  │                                                          │    │
│  │   Step 1: embed_tokens = model.get_input_embeddings()    │    │
│  │                                                          │    │
│  │   Step 2: embed_mm_inputs()                              │    │
│  │     │                                                    │    │
│  │     ├─► 遍历 IMAGE 模态                                 │    │
│  │     │     │                                              │    │
│  │     │     └─► get_embedding_and_mask()                  │    │
│  │     │           │                                        │    │
│  │     │           ├─► _get_chunked_prefill_embedding()    │    │
│  │     │           │     │                                  │    │
│  │     │           │     ├─► cache.get() → miss            │    │
│  │     │           │     │                                  │    │
│  │     │           │     ├─► ViT(pixel_values)             │    │
│  │     │           │     │   → image_embeds (576, 4096)    │    │
│  │     │           │     │                                  │    │
│  │     │           │     ├─► cache.set()                   │    │
│  │     │           │     │                                  │    │
│  │     │           │     └─► get_embedding_chunk()         │    │
│  │     │           │         → chunk_embeds (576, 4096)    │    │
│  │     │           │                                        │    │
│  │     │           └─► _get_multimodal_mask()              │    │
│  │     │               → mask (600,) bool                  │    │
│  │     │                                                    │    │
│  │     ├─► input_embeds = embed_tokens(input_ids)          │    │
│  │     │   → (600, 4096)                                   │    │
│  │     │                                                    │    │
│  │     └─► input_embeds[mask] = image_embeds               │    │
│  │         → 替换576个位置                                  │    │
│  │                                                          │    │
│  │   Step 3: language_model(input_embeds=input_embeds)      │    │
│  │     → hidden_states                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 五、可运行单元测试示例

### 5.1 测试 get_embedding_chunk 函数

```python
# test_embedding_chunk.py
import torch
import unittest
import sys
sys.path.insert(0, '/workspace/python')

from sglang.srt.managers.mm_utils import get_embedding_chunk


class TestGetEmbeddingChunk(unittest.TestCase):
    """测试 get_embedding_chunk 函数"""
    
    def test_full_overlap(self):
        """
        测试场景: chunk 完全在图片区间内
        
        图片区间: [100, 675] (576个token)
        当前chunk: [200, 500] (extend_prefix_len=200, extend_seq_len=300)
        期望提取: embedding[100:400] (300个嵌入)
        """
        embedding = torch.randn(576, 4096)  # 576个图片token
        extend_prefix_len = 200
        extend_seq_len = 300
        items_offset = [(100, 675)]  # 图片在位置100-675
        
        chunk, start_idx, end_idx = get_embedding_chunk(
            embedding, extend_prefix_len, extend_seq_len, items_offset
        )
        
        # chunk 开始位置 200 在区间内, start_index = 200 - 100 = 100
        # chunk 结束位置 499 在区间内, end_index = 499 - 100 + 1 = 400
        self.assertEqual(start_idx, 100)
        self.assertEqual(end_idx, 400)
        self.assertEqual(chunk.shape[0], 300)  # 提取300个嵌入
        
    def test_partial_overlap_start(self):
        """
        测试场景: chunk 部分与图片区间重叠 (chunk 开始在图片前)
        
        图片区间: [100, 675]
        当前chunk: [0, 200] (extend_prefix_len=0, extend_seq_len=200)
        期望提取: embedding[0:100] (100个嵌入, 对应位置100-199)
        """
        embedding = torch.randn(576, 4096)
        extend_prefix_len = 0
        extend_seq_len = 200
        items_offset = [(100, 675)]
        
        chunk, start_idx, end_idx = get_embedding_chunk(
            embedding, extend_prefix_len, extend_seq_len, items_offset
        )
        
        # chunk 开始位置 0 在区间 [100, 675] 之前, start_index = 0
        # chunk 结束位置 199 在区间内, end_index = 199 - 100 + 1 = 100
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 100)
        self.assertEqual(chunk.shape[0], 100)
        
    def test_partial_overlap_end(self):
        """
        测试场景: chunk 部分与图片区间重叠 (chunk 结束在图片后)
        
        图片区间: [100, 675]
        当前chunk: [600, 800] (extend_prefix_len=600, extend_seq_len=200)
        期望提取: embedding[500:576] (76个嵌入, 对应位置600-675)
        """
        embedding = torch.randn(576, 4096)
        extend_prefix_len = 600
        extend_seq_len = 200
        items_offset = [(100, 675)]
        
        chunk, start_idx, end_idx = get_embedding_chunk(
            embedding, extend_prefix_len, extend_seq_len, items_offset
        )
        
        # chunk 开始位置 600 在区间内, start_index = 600 - 100 = 500
        # chunk 结束位置 799 在区间外, end_index = 675 - 100 + 1 = 576
        self.assertEqual(start_idx, 500)
        self.assertEqual(end_idx, 576)
        self.assertEqual(chunk.shape[0], 76)
        
    def test_no_overlap(self):
        """
        测试场景: chunk 与图片区间无重叠
        
        图片区间: [100, 675]
        当前chunk: [700, 900] (extend_prefix_len=700, extend_seq_len=200)
        期望提取: 空张量
        """
        embedding = torch.randn(576, 4096)
        extend_prefix_len = 700
        extend_seq_len = 200
        items_offset = [(100, 675)]
        
        chunk, start_idx, end_idx = get_embedding_chunk(
            embedding, extend_prefix_len, extend_seq_len, items_offset
        )
        
        # chunk 完全在图片区间之后
        # start_index = 576 (累加完整区间)
        # end_index = 576
        self.assertEqual(chunk.shape[0], 0)
        
    def test_multiple_images(self):
        """
        测试场景: 多张图片
        
        图片区间: [(50, 149), (200, 299)] (两张图片各100个token)
        当前chunk: [100, 250] (extend_prefix_len=100, extend_seq_len=150)
        期望: 提取第一张图的后50个 + 第二张图的前50个 = 100个嵌入
        """
        embedding = torch.randn(200, 4096)  # 两张图共200个token
        extend_prefix_len = 100
        extend_seq_len = 150
        items_offset = [(50, 149), (200, 299)]
        
        chunk, start_idx, end_idx = get_embedding_chunk(
            embedding, extend_prefix_len, extend_seq_len, items_offset
        )
        
        # 第一张图: [50, 149], chunk[100, 249]
        #   start_index += 100 - 50 = 50 (因为chunk开始在图片区间内)
        #   end_index += 149 - 50 + 1 = 100 (因为chunk结束在图片区间外)
        # 第二张图: [200, 299], chunk[100, 249]
        #   start_index 不变 (chunk开始在这个区间前)
        #   end_index += 249 - 200 + 1 = 150
        self.assertEqual(start_idx, 50)
        self.assertEqual(end_idx, 150)
        self.assertEqual(chunk.shape[0], 100)


if __name__ == '__main__':
    unittest.main()
```

### 5.2 测试 _get_multimodal_mask 函数

```python
# test_multimodal_mask.py
import torch
import unittest
import sys
sys.path.insert(0, '/workspace/python')

from sglang.srt.managers.mm_utils import _get_multimodal_mask


class TestGetMultimodalMask(unittest.TestCase):
    """测试 _get_multimodal_mask 函数"""
    
    def test_simple_mask(self):
        """
        测试场景: 简单的占位符掩码
        
        input_ids: [1, 2, 999, 999, 999, 3, 4]
        placeholder: [999]
        期望mask: [F, F, T, T, T, F, F]
        """
        input_ids = torch.tensor([1, 2, 999, 999, 999, 3, 4])
        placeholder_tensor = torch.tensor([999])
        
        mask = _get_multimodal_mask(input_ids, placeholder_tensor)
        
        expected = torch.tensor([False, False, True, True, True, False, False])
        self.assertTrue(torch.equal(mask.squeeze(-1), expected))
        
    def test_multiple_placeholders(self):
        """
        测试场景: 多个不同的占位符值
        
        input_ids: [1, 888, 2, 999, 3]
        placeholder: [888, 999]
        期望mask: [F, T, F, T, F]
        """
        input_ids = torch.tensor([1, 888, 2, 999, 3])
        placeholder_tensor = torch.tensor([888, 999])
        
        mask = _get_multimodal_mask(input_ids, placeholder_tensor)
        
        expected = torch.tensor([False, True, False, True, False])
        self.assertTrue(torch.equal(mask.squeeze(-1), expected))
        
    def test_no_placeholders(self):
        """
        测试场景: 没有占位符
        """
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        placeholder_tensor = torch.tensor([999])
        
        mask = _get_multimodal_mask(input_ids, placeholder_tensor)
        
        self.assertEqual(mask.sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
```

### 5.3 测试完整的 embed_mm_inputs 流程 (Mock)

```python
# test_embed_mm_inputs.py
import torch
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.insert(0, '/workspace/python')

from sglang.srt.managers.schedule_batch import (
    Modality, 
    MultimodalDataItem, 
    MultimodalInputs
)
from sglang.srt.managers.mm_utils import (
    embed_mm_inputs,
    init_mm_embedding_cache,
)


class TestEmbedMmInputs(unittest.TestCase):
    """测试 embed_mm_inputs 函数"""
    
    @classmethod
    def setUpClass(cls):
        # 初始化多模态嵌入缓存
        init_mm_embedding_cache(max_size=1024 * 1024 * 100)  # 100MB
    
    def create_mock_mm_item(self, num_tokens=576, hidden_size=4096):
        """创建模拟的多模态数据项"""
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=12345,
            pad_value=12345 % (1 << 30),
            offsets=[(10, 10 + num_tokens - 1)],
            feature=torch.randn(1, 3, 224, 224),  # 模拟 pixel_values
        )
        return item
    
    def test_basic_embedding_replacement(self):
        """
        测试基本的嵌入替换流程
        
        场景: 
        - 1个请求, 1张图片
        - input_ids 中位置 10-585 是图片占位符
        - 期望: 这些位置被替换为图片嵌入
        """
        hidden_size = 128  # 使用较小的维度便于测试
        num_image_tokens = 50
        total_tokens = 100
        
        # 创建测试数据
        pad_value = 99999
        input_ids = torch.zeros(total_tokens, dtype=torch.long)
        input_ids[10:60] = pad_value  # 图片占位符
        
        # 创建多模态输入
        mm_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=pad_value,
            pad_value=pad_value,
            offsets=[(10, 59)],
            feature=torch.randn(1, 3, 224, 224),
        )
        mm_inputs = MultimodalInputs(
            mm_items=[mm_item],
            im_token_id=pad_value,
        )
        
        # 创建模拟的嵌入层
        embed_layer = torch.nn.Embedding(100000, hidden_size)
        
        # 创建模拟的多模态模型
        multimodal_model = Mock()
        image_embedding = torch.randn(num_image_tokens, hidden_size)
        multimodal_model.get_image_feature = Mock(return_value=image_embedding)
        
        # 调用函数
        result, other_info = embed_mm_inputs(
            mm_inputs_list=[mm_inputs],
            extend_prefix_lens=[0],
            extend_seq_lens=[total_tokens],
            input_ids=input_ids,
            input_embedding=embed_layer,
            multimodal_model=multimodal_model,
            data_embedding_func_mapping={Modality.IMAGE: multimodal_model.get_image_feature},
        )
        
        # 验证
        self.assertEqual(result.shape, (total_tokens, hidden_size))
        
        # 验证图片位置的嵌入已被替换
        # (由于 mock, 我们无法直接验证值,但可以验证形状和调用)
        multimodal_model.get_image_feature.assert_called_once()


class TestEmbeddingChunkIntegration(unittest.TestCase):
    """
    集成测试: 模拟分块预填充场景
    """
    
    def test_chunked_prefill_scenario(self):
        """
        测试分块预填充场景
        
        场景:
        - 总序列长度: 1000 tokens
        - 图片位置: [200, 775] (576 tokens)
        - Chunk 1: [0, 500)
        - Chunk 2: [500, 1000)
        
        期望:
        - Chunk 1 需要图片嵌入 [200, 500) 部分
        - Chunk 2 需要图片嵌入 [500, 775] 部分
        """
        from sglang.srt.managers.mm_utils import get_embedding_chunk
        
        embedding = torch.randn(576, 256)  # 完整图片嵌入
        image_start, image_end = 200, 775
        
        # Chunk 1
        chunk1, _, _ = get_embedding_chunk(
            embedding=embedding,
            extend_prefix_len=0,
            extend_seq_len=500,
            items_offset=[(image_start, image_end)]
        )
        
        # Chunk 2
        chunk2, _, _ = get_embedding_chunk(
            embedding=embedding,
            extend_prefix_len=500,
            extend_seq_len=500,
            items_offset=[(image_start, image_end)]
        )
        
        # 验证
        # Chunk 1: [0, 500) 与 [200, 775] 交集是 [200, 500), 共300个token
        self.assertEqual(chunk1.shape[0], 300)
        
        # Chunk 2: [500, 1000) 与 [200, 775] 交集是 [500, 775], 共276个token
        self.assertEqual(chunk2.shape[0], 276)
        
        # 验证总和等于完整图片
        self.assertEqual(chunk1.shape[0] + chunk2.shape[0], 576)


if __name__ == '__main__':
    unittest.main()
```

### 5.4 端到端测试脚本

```python
# test_e2e_mm_embed.py
"""
端到端测试: 模拟完整的多模态嵌入流程

这个测试展示了从 input_ids + MultimodalInputs 到最终嵌入的完整流程
"""
import torch
import unittest
import sys
sys.path.insert(0, '/workspace/python')


def simulate_mm_embed_routine():
    """
    模拟 general_mm_embed_routine 的核心逻辑
    不依赖实际模型,便于理解流程
    """
    print("=" * 70)
    print("模拟多模态嵌入流程")
    print("=" * 70)
    
    # 配置
    vocab_size = 32000
    hidden_size = 256
    total_tokens = 100
    num_image_tokens = 50
    image_placeholder = 151655  # 典型的图片占位符 ID
    
    # Step 1: 构造 input_ids
    # 格式: [text_tokens][image_placeholder * N][more_text_tokens]
    print("\n[Step 1] 构造 input_ids")
    input_ids = torch.randint(0, vocab_size, (total_tokens,))
    image_start, image_end = 20, 20 + num_image_tokens - 1
    input_ids[image_start:image_end+1] = image_placeholder
    print(f"  - Total tokens: {total_tokens}")
    print(f"  - Image position: [{image_start}, {image_end}]")
    print(f"  - Image tokens: {num_image_tokens}")
    print(f"  - input_ids sample: {input_ids[:25].tolist()}...")
    
    # Step 2: 创建文本嵌入
    print("\n[Step 2] 创建文本嵌入")
    embed_layer = torch.nn.Embedding(vocab_size + 10000, hidden_size)  # 额外空间给占位符
    text_embeds = embed_layer(input_ids.clamp(max=vocab_size-1))
    print(f"  - text_embeds shape: {text_embeds.shape}")
    
    # Step 3: 创建多模态掩码
    print("\n[Step 3] 创建多模态掩码")
    mm_mask = (input_ids == image_placeholder)
    mm_positions = torch.where(mm_mask)[0]
    print(f"  - Mask sum: {mm_mask.sum().item()}")
    print(f"  - MM positions: [{mm_positions[0].item()}, ..., {mm_positions[-1].item()}]")
    
    # Step 4: 模拟 ViT 计算图片嵌入
    print("\n[Step 4] 模拟 ViT 计算图片嵌入")
    # 实际中这里会调用: image_embeds = vit_model(pixel_values)
    image_embeds = torch.randn(num_image_tokens, hidden_size)
    print(f"  - image_embeds shape: {image_embeds.shape}")
    
    # Step 5: 替换嵌入
    print("\n[Step 5] 替换多模态位置的嵌入")
    final_embeds = text_embeds.clone()
    final_embeds[mm_mask] = image_embeds
    print(f"  - final_embeds shape: {final_embeds.shape}")
    
    # Step 6: 验证
    print("\n[Step 6] 验证替换结果")
    # 检查替换前后的差异
    text_positions = ~mm_mask
    text_diff = (final_embeds[text_positions] - text_embeds[text_positions]).abs().sum()
    mm_diff = (final_embeds[mm_mask] - image_embeds).abs().sum()
    print(f"  - Text positions unchanged: {text_diff.item() < 1e-6}")
    print(f"  - MM positions correctly replaced: {mm_diff.item() < 1e-6}")
    
    print("\n" + "=" * 70)
    print("流程完成!")
    print("=" * 70)
    
    return final_embeds


def simulate_chunked_prefill():
    """
    模拟分块预填充场景
    """
    print("\n" + "=" * 70)
    print("模拟分块预填充场景")
    print("=" * 70)
    
    # 配置
    hidden_size = 256
    total_seq_len = 1000
    num_image_tokens = 576
    chunk_size = 300
    image_start, image_end = 200, 200 + num_image_tokens - 1  # [200, 775]
    
    print(f"\n配置:")
    print(f"  - Total sequence: {total_seq_len} tokens")
    print(f"  - Image position: [{image_start}, {image_end}] ({num_image_tokens} tokens)")
    print(f"  - Chunk size: {chunk_size}")
    
    # 模拟完整的图片嵌入 (在实际场景中由 ViT 计算)
    full_image_embeds = torch.randn(num_image_tokens, hidden_size)
    
    # 模拟缓存
    cache = {}
    cache_key = "image_hash_12345"
    
    # 分块处理
    num_chunks = (total_seq_len + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        prefix_len = chunk_idx * chunk_size
        seq_len = min(chunk_size, total_seq_len - prefix_len)
        chunk_start = prefix_len
        chunk_end = prefix_len + seq_len - 1
        
        print(f"\n[Chunk {chunk_idx + 1}] Range: [{chunk_start}, {chunk_end}]")
        
        # 计算与图片区间的交集
        overlap_start = max(chunk_start, image_start)
        overlap_end = min(chunk_end, image_end)
        
        if overlap_start <= overlap_end:
            # 有交集,需要图片嵌入
            
            # 检查缓存
            if cache_key not in cache:
                print(f"  - Cache MISS, computing image embedding...")
                cache[cache_key] = full_image_embeds
            else:
                print(f"  - Cache HIT!")
            
            # 提取当前 chunk 需要的部分
            embed_start = overlap_start - image_start
            embed_end = overlap_end - image_start + 1
            chunk_image_embeds = cache[cache_key][embed_start:embed_end]
            
            print(f"  - Image overlap: [{overlap_start}, {overlap_end}]")
            print(f"  - Extracted embeddings: [{embed_start}, {embed_end}) = {chunk_image_embeds.shape[0]} tokens")
        else:
            print(f"  - No image overlap in this chunk")
    
    print("\n" + "=" * 70)
    print("分块预填充完成!")
    print("=" * 70)


if __name__ == '__main__':
    # 运行模拟
    simulate_mm_embed_routine()
    simulate_chunked_prefill()
```

## 六、关键点总结

### 6.1 占位符机制

```
input_ids 中的图片位置被填充为 pad_value = hash(image) % 2^30

这样做的好处:
1. RadixCache 可以通过 input_ids 匹配前缀,包括图片部分
2. 相同的图片会有相同的 pad_value,支持缓存复用
3. pad_value 足够大,不会与正常 token ID 冲突
```

### 6.2 缓存策略

```
MultiModalStaticCache 使用 LRU 策略:
- key: hash(所有图片item的hash组合)
- value: 完整的图片嵌入

缓存命中场景:
1. 同一请求的分块预填充 (Chunk 2,3,... 复用 Chunk 1 的计算)
2. 不同请求使用相同图片
3. 多轮对话中复用历史图片
```

### 6.3 分块预填充的嵌入提取

```
get_embedding_chunk 的核心逻辑:

对于每个图片区间 (start, end):
  - 如果 chunk_start 在区间内: start_index += chunk_start - start
  - 如果 chunk_start 在区间后: start_index += 区间长度
  - 如果 chunk_end 在区间内: end_index += chunk_end - start + 1
  - 如果 chunk_end 在区间后: end_index += 区间长度

最终: embedding_chunk = embedding[start_index:end_index]
```

## 七、调试技巧

### 7.1 打印关键中间变量

```python
# 在 embed_mm_inputs 中添加调试输出
print(f"mm_inputs_list length: {len(mm_inputs_list)}")
for i, mm_input in enumerate(mm_inputs_list):
    print(f"  Request {i}:")
    print(f"    mm_items: {len(mm_input.mm_items)}")
    for j, item in enumerate(mm_input.mm_items):
        print(f"      Item {j}: modality={item.modality}, offsets={item.offsets}")
```

### 7.2 验证嵌入替换

```python
# 在替换后检查
mm_positions = torch.where(mask)[0]
print(f"Replaced {len(mm_positions)} positions")
print(f"First replaced position: {mm_positions[0].item()}")
print(f"Last replaced position: {mm_positions[-1].item()}")
```
