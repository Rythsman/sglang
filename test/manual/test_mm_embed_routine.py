"""
SGLang 多模态嵌入处理流程单元测试

这些测试帮助理解 general_mm_embed_routine 及其相关函数的工作原理。
可以直接运行: python test/manual/test_mm_embed_routine.py

Author: SGLang Team
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

import torch
import unittest
from typing import List, Tuple, Optional


class TestGetEmbeddingChunk(unittest.TestCase):
    """
    测试 get_embedding_chunk 函数
    
    该函数从完整的多模态嵌入中提取当前 chunk 需要的部分。
    这是分块预填充 (chunked prefill) 的核心功能。
    """
    
    def setUp(self):
        """导入被测函数"""
        from sglang.srt.managers.mm_utils import get_embedding_chunk
        self.get_embedding_chunk = get_embedding_chunk
    
    def test_case_1_full_overlap(self):
        """
        测试用例1: Chunk 完全在图片区间内
        
        场景图示:
        0         100       200       300       400       500       600       700
        |---------|---------|---------|---------|---------|---------|---------|
                  [==========图片区间 [100, 675]====================]
                            [===Chunk [200, 500)===]
                            
        期望: 提取 embedding[100:400] (共300个嵌入)
        """
        print("\n" + "=" * 60)
        print("测试用例1: Chunk 完全在图片区间内")
        print("=" * 60)
        
        # 创建测试数据
        hidden_size = 256
        num_image_tokens = 576
        embedding = torch.randn(num_image_tokens, hidden_size)
        
        extend_prefix_len = 200  # chunk 从位置 200 开始
        extend_seq_len = 300     # chunk 长度 300
        items_offset = [(100, 675)]  # 图片在位置 100-675
        
        print(f"输入参数:")
        print(f"  - embedding.shape: {embedding.shape}")
        print(f"  - extend_prefix_len: {extend_prefix_len}")
        print(f"  - extend_seq_len: {extend_seq_len}")
        print(f"  - items_offset: {items_offset}")
        
        # 调用函数
        chunk, start_idx, end_idx = self.get_embedding_chunk(
            embedding, extend_prefix_len, extend_seq_len, items_offset
        )
        
        print(f"\n输出结果:")
        print(f"  - chunk.shape: {chunk.shape}")
        print(f"  - start_idx: {start_idx}")
        print(f"  - end_idx: {end_idx}")
        
        # 手动计算期望值
        # chunk_start = 200, chunk_end = 499 (200 + 300 - 1)
        # 图片区间 [100, 675]
        # start_index = 200 - 100 = 100 (chunk_start 在区间内)
        # end_index = 499 - 100 + 1 = 400 (chunk_end 在区间内)
        expected_start = 100
        expected_end = 400
        expected_num = 300
        
        print(f"\n期望结果:")
        print(f"  - expected start_idx: {expected_start}")
        print(f"  - expected end_idx: {expected_end}")
        print(f"  - expected chunk size: {expected_num}")
        
        # 断言
        self.assertEqual(start_idx, expected_start)
        self.assertEqual(end_idx, expected_end)
        self.assertEqual(chunk.shape[0], expected_num)
        print("\n✓ 测试通过!")
    
    def test_case_2_partial_overlap_start(self):
        """
        测试用例2: Chunk 与图片区间部分重叠 (chunk 开始在图片前)
        
        场景图示:
        0         100       200       300
        |---------|---------|---------|
                  [===图片 [100, 299]===]
        [===Chunk [0, 150)===]
        
        期望: 提取 embedding[0:50] (共50个嵌入, 对应位置100-149)
        """
        print("\n" + "=" * 60)
        print("测试用例2: Chunk 与图片区间部分重叠 (chunk在前)")
        print("=" * 60)
        
        embedding = torch.randn(200, 256)
        extend_prefix_len = 0
        extend_seq_len = 150
        items_offset = [(100, 299)]
        
        print(f"输入参数:")
        print(f"  - extend_prefix_len: {extend_prefix_len}")
        print(f"  - extend_seq_len: {extend_seq_len}")
        print(f"  - items_offset: {items_offset}")
        
        chunk, start_idx, end_idx = self.get_embedding_chunk(
            embedding, extend_prefix_len, extend_seq_len, items_offset
        )
        
        print(f"\n输出结果:")
        print(f"  - chunk.shape: {chunk.shape}")
        print(f"  - start_idx: {start_idx}")
        print(f"  - end_idx: {end_idx}")
        
        # chunk_start = 0, chunk_end = 149
        # 图片区间 [100, 299]
        # start_index = 0 (chunk_start 在区间前)
        # end_index = 149 - 100 + 1 = 50 (chunk_end 在区间内)
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 50)
        self.assertEqual(chunk.shape[0], 50)
        print("\n✓ 测试通过!")
    
    def test_case_3_partial_overlap_end(self):
        """
        测试用例3: Chunk 与图片区间部分重叠 (chunk 结束在图片后)
        
        场景图示:
        0         100       200       300       400
        |---------|---------|---------|---------|
                  [===图片 [100, 299]===]
                            [===Chunk [200, 400)===]
        
        期望: 提取 embedding[100:200] (共100个嵌入, 对应位置200-299)
        """
        print("\n" + "=" * 60)
        print("测试用例3: Chunk 与图片区间部分重叠 (chunk在后)")
        print("=" * 60)
        
        embedding = torch.randn(200, 256)
        extend_prefix_len = 200
        extend_seq_len = 200
        items_offset = [(100, 299)]
        
        print(f"输入参数:")
        print(f"  - extend_prefix_len: {extend_prefix_len}")
        print(f"  - extend_seq_len: {extend_seq_len}")
        print(f"  - items_offset: {items_offset}")
        
        chunk, start_idx, end_idx = self.get_embedding_chunk(
            embedding, extend_prefix_len, extend_seq_len, items_offset
        )
        
        print(f"\n输出结果:")
        print(f"  - chunk.shape: {chunk.shape}")
        print(f"  - start_idx: {start_idx}")
        print(f"  - end_idx: {end_idx}")
        
        # chunk_start = 200, chunk_end = 399
        # 图片区间 [100, 299]
        # start_index = 200 - 100 = 100 (chunk_start 在区间内)
        # end_index = 299 - 100 + 1 = 200 (chunk_end 在区间外, 使用区间结束)
        self.assertEqual(start_idx, 100)
        self.assertEqual(end_idx, 200)
        self.assertEqual(chunk.shape[0], 100)
        print("\n✓ 测试通过!")
    
    def test_case_4_no_overlap(self):
        """
        测试用例4: Chunk 与图片区间无重叠
        
        场景图示:
        0         100       200       300       400       500
        |---------|---------|---------|---------|---------|
                  [===图片 [100, 299]===]
                                        [===Chunk [400, 500)===]
        
        期望: 返回空张量
        """
        print("\n" + "=" * 60)
        print("测试用例4: Chunk 与图片区间无重叠")
        print("=" * 60)
        
        embedding = torch.randn(200, 256)
        extend_prefix_len = 400
        extend_seq_len = 100
        items_offset = [(100, 299)]
        
        print(f"输入参数:")
        print(f"  - extend_prefix_len: {extend_prefix_len}")
        print(f"  - extend_seq_len: {extend_seq_len}")
        print(f"  - items_offset: {items_offset}")
        
        chunk, start_idx, end_idx = self.get_embedding_chunk(
            embedding, extend_prefix_len, extend_seq_len, items_offset
        )
        
        print(f"\n输出结果:")
        print(f"  - chunk.shape: {chunk.shape}")
        print(f"  - start_idx: {start_idx}")
        print(f"  - end_idx: {end_idx}")
        
        # chunk 完全在图片区间之后
        # start_index 和 end_index 都等于完整区间长度
        self.assertEqual(chunk.shape[0], 0)
        print("\n✓ 测试通过!")
    
    def test_case_5_multiple_images(self):
        """
        测试用例5: 多张图片
        
        场景图示:
        0         50        100       150       200       250       300
        |---------|---------|---------|---------|---------|---------|
        [==图片1 [0, 99]==]
                            [==图片2 [150, 249]==]
                  [=======Chunk [50, 200)=======]
        
        期望: 提取图片1的后50个 + 图片2的前50个 = 100个嵌入
        """
        print("\n" + "=" * 60)
        print("测试用例5: 多张图片")
        print("=" * 60)
        
        # 两张图片各100个token
        embedding = torch.randn(200, 256)
        extend_prefix_len = 50
        extend_seq_len = 150
        items_offset = [(0, 99), (150, 249)]
        
        print(f"输入参数:")
        print(f"  - embedding.shape: {embedding.shape} (两张图共200个token)")
        print(f"  - extend_prefix_len: {extend_prefix_len}")
        print(f"  - extend_seq_len: {extend_seq_len}")
        print(f"  - items_offset: {items_offset}")
        
        chunk, start_idx, end_idx = self.get_embedding_chunk(
            embedding, extend_prefix_len, extend_seq_len, items_offset
        )
        
        print(f"\n输出结果:")
        print(f"  - chunk.shape: {chunk.shape}")
        print(f"  - start_idx: {start_idx}")
        print(f"  - end_idx: {end_idx}")
        
        # chunk_start = 50, chunk_end = 199
        # 
        # 图片1 [0, 99]:
        #   start_index = 50 - 0 = 50 (chunk_start 在区间内)
        #   end_index = 99 - 0 + 1 = 100 (chunk_end 在区间外)
        #
        # 图片2 [150, 249]:
        #   start_index += 0 (chunk_start 在这个区间前)
        #   end_index += 199 - 150 + 1 = 50 (chunk_end 在区间内)
        #
        # 最终: start_index = 50, end_index = 150
        # 提取: embedding[50:150] = 100个token
        
        self.assertEqual(start_idx, 50)
        self.assertEqual(end_idx, 150)
        self.assertEqual(chunk.shape[0], 100)
        print("\n✓ 测试通过!")


class TestMultimodalMask(unittest.TestCase):
    """
    测试 _get_multimodal_mask 函数
    
    该函数创建一个布尔掩码,标识 input_ids 中哪些位置是多模态占位符。
    """
    
    def setUp(self):
        from sglang.srt.managers.mm_utils import _get_multimodal_mask
        self.get_mask = _get_multimodal_mask
    
    def test_simple_mask(self):
        """
        测试简单的占位符掩码
        
        input_ids: [1, 2, 999, 999, 999, 3, 4]
        placeholder: [999]
        期望mask: [F, F, T, T, T, F, F]
        """
        print("\n" + "=" * 60)
        print("测试简单的占位符掩码")
        print("=" * 60)
        
        input_ids = torch.tensor([1, 2, 999, 999, 999, 3, 4])
        placeholder_tensor = torch.tensor([999])
        
        print(f"input_ids: {input_ids.tolist()}")
        print(f"placeholder: {placeholder_tensor.tolist()}")
        
        mask = self.get_mask(input_ids, placeholder_tensor)
        
        print(f"mask: {mask.squeeze(-1).tolist()}")
        
        expected = torch.tensor([False, False, True, True, True, False, False])
        self.assertTrue(torch.equal(mask.squeeze(-1), expected))
        print("\n✓ 测试通过!")
    
    def test_multiple_placeholders(self):
        """
        测试多个不同的占位符值
        
        input_ids: [1, 888, 2, 999, 3]
        placeholder: [888, 999]
        期望mask: [F, T, F, T, F]
        """
        print("\n" + "=" * 60)
        print("测试多个不同的占位符值")
        print("=" * 60)
        
        input_ids = torch.tensor([1, 888, 2, 999, 3])
        placeholder_tensor = torch.tensor([888, 999])
        
        print(f"input_ids: {input_ids.tolist()}")
        print(f"placeholders: {placeholder_tensor.tolist()}")
        
        mask = self.get_mask(input_ids, placeholder_tensor)
        
        print(f"mask: {mask.squeeze(-1).tolist()}")
        
        expected = torch.tensor([False, True, False, True, False])
        self.assertTrue(torch.equal(mask.squeeze(-1), expected))
        print("\n✓ 测试通过!")


class TestEmbeddingReplacement(unittest.TestCase):
    """
    测试嵌入替换逻辑
    
    这是 embed_mm_inputs 的核心操作:
    1. 获取文本嵌入
    2. 创建多模态掩码
    3. 用多模态嵌入替换掩码位置
    """
    
    def test_embedding_replacement(self):
        """
        测试嵌入替换的完整流程
        """
        print("\n" + "=" * 60)
        print("测试嵌入替换流程")
        print("=" * 60)
        
        # 配置
        vocab_size = 1000
        hidden_size = 64
        total_tokens = 20
        num_mm_tokens = 5
        mm_placeholder = 999
        
        # Step 1: 创建 input_ids
        input_ids = torch.randint(0, vocab_size, (total_tokens,))
        mm_start, mm_end = 8, 12
        input_ids[mm_start:mm_end+1] = mm_placeholder
        
        print(f"Step 1: 创建 input_ids")
        print(f"  - Total tokens: {total_tokens}")
        print(f"  - MM positions: [{mm_start}, {mm_end}]")
        print(f"  - input_ids: {input_ids.tolist()}")
        
        # Step 2: 创建文本嵌入
        embed_layer = torch.nn.Embedding(vocab_size + 1000, hidden_size)
        clamped_ids = input_ids.clamp(max=vocab_size-1)
        text_embeds = embed_layer(clamped_ids)
        
        print(f"\nStep 2: 创建文本嵌入")
        print(f"  - text_embeds.shape: {text_embeds.shape}")
        
        # Step 3: 创建掩码
        mm_mask = (input_ids == mm_placeholder)
        
        print(f"\nStep 3: 创建掩码")
        print(f"  - mask: {mm_mask.tolist()}")
        print(f"  - mask sum: {mm_mask.sum().item()}")
        
        # Step 4: 创建多模态嵌入 (模拟 ViT 输出)
        mm_embeds = torch.ones(num_mm_tokens, hidden_size) * 999  # 用特殊值便于验证
        
        print(f"\nStep 4: 创建多模态嵌入")
        print(f"  - mm_embeds.shape: {mm_embeds.shape}")
        print(f"  - mm_embeds 使用特殊值 999 便于验证")
        
        # Step 5: 替换
        final_embeds = text_embeds.clone()
        final_embeds[mm_mask] = mm_embeds
        
        print(f"\nStep 5: 执行替换")
        print(f"  - final_embeds.shape: {final_embeds.shape}")
        
        # Step 6: 验证
        print(f"\nStep 6: 验证结果")
        
        # 检查多模态位置是否正确替换
        replaced_values = final_embeds[mm_mask]
        is_replaced = torch.allclose(replaced_values, mm_embeds)
        print(f"  - 多模态位置正确替换: {is_replaced}")
        
        # 检查文本位置是否保持不变
        text_mask = ~mm_mask
        text_unchanged = torch.allclose(final_embeds[text_mask], text_embeds[text_mask])
        print(f"  - 文本位置保持不变: {text_unchanged}")
        
        self.assertTrue(is_replaced)
        self.assertTrue(text_unchanged)
        print("\n✓ 测试通过!")


class TestChunkedPrefillSimulation(unittest.TestCase):
    """
    模拟分块预填充场景
    
    这个测试展示了当序列很长需要分块处理时,
    如何正确提取每个 chunk 需要的多模态嵌入。
    """
    
    def setUp(self):
        from sglang.srt.managers.mm_utils import get_embedding_chunk
        self.get_embedding_chunk = get_embedding_chunk
    
    def test_chunked_prefill_simulation(self):
        """
        模拟分块预填充场景
        
        配置:
        - 总序列长度: 1000 tokens
        - 图片位置: [200, 775] (576 tokens)
        - Chunk size: 300
        
        Chunks:
        - Chunk 0: [0, 300) - 与图片交集 [200, 300) = 100 tokens
        - Chunk 1: [300, 600) - 与图片交集 [300, 600) = 300 tokens
        - Chunk 2: [600, 900) - 与图片交集 [600, 775] = 176 tokens
        - Chunk 3: [900, 1000) - 无交集
        
        期望: 所有 chunk 的嵌入总和 = 576
        """
        print("\n" + "=" * 60)
        print("模拟分块预填充场景")
        print("=" * 60)
        
        # 配置
        hidden_size = 64
        total_seq_len = 1000
        num_image_tokens = 576
        chunk_size = 300
        image_start, image_end = 200, 775  # [200, 775]
        
        print(f"配置:")
        print(f"  - Total sequence: {total_seq_len} tokens")
        print(f"  - Image range: [{image_start}, {image_end}] ({num_image_tokens} tokens)")
        print(f"  - Chunk size: {chunk_size}")
        
        # 创建完整的图片嵌入
        full_embedding = torch.randn(num_image_tokens, hidden_size)
        items_offset = [(image_start, image_end)]
        
        # 分块处理
        chunks = []
        num_chunks = (total_seq_len + chunk_size - 1) // chunk_size
        
        print(f"\n分块处理:")
        for i in range(num_chunks):
            prefix_len = i * chunk_size
            seq_len = min(chunk_size, total_seq_len - prefix_len)
            
            chunk, start_idx, end_idx = self.get_embedding_chunk(
                full_embedding, prefix_len, seq_len, items_offset
            )
            
            chunk_start = prefix_len
            chunk_end = prefix_len + seq_len - 1
            
            print(f"  Chunk {i}: [{chunk_start}, {chunk_end}] -> 提取 {chunk.shape[0]} 个嵌入")
            
            if chunk.shape[0] > 0:
                chunks.append(chunk)
        
        # 验证
        total_extracted = sum(c.shape[0] for c in chunks)
        print(f"\n验证:")
        print(f"  - 总共提取: {total_extracted} 个嵌入")
        print(f"  - 期望: {num_image_tokens} 个嵌入")
        
        self.assertEqual(total_extracted, num_image_tokens)
        print("\n✓ 测试通过!")


class TestFullPipelineSimulation(unittest.TestCase):
    """
    完整流程模拟
    
    模拟 general_mm_embed_routine 的完整处理流程,
    不依赖实际模型,便于理解整体逻辑。
    """
    
    def test_full_pipeline(self):
        """
        模拟完整的多模态嵌入处理流程
        """
        print("\n" + "=" * 70)
        print("完整流程模拟")
        print("=" * 70)
        
        # 配置
        vocab_size = 32000
        hidden_size = 128
        total_tokens = 100
        num_image_tokens = 50
        image_placeholder = 151655
        
        print(f"\n配置:")
        print(f"  - vocab_size: {vocab_size}")
        print(f"  - hidden_size: {hidden_size}")
        print(f"  - total_tokens: {total_tokens}")
        print(f"  - num_image_tokens: {num_image_tokens}")
        print(f"  - image_placeholder: {image_placeholder}")
        
        # Step 1: 构造 input_ids
        print(f"\n[Step 1] 构造 input_ids")
        input_ids = torch.randint(0, vocab_size, (total_tokens,))
        image_start, image_end = 20, 20 + num_image_tokens - 1
        input_ids[image_start:image_end+1] = image_placeholder
        print(f"  - Image position: [{image_start}, {image_end}]")
        
        # Step 2: 检查 forward_mode
        print(f"\n[Step 2] 检查 forward_mode")
        forward_mode = "EXTEND"  # 模拟
        is_decode = forward_mode == "DECODE"
        has_mm_inputs = True
        should_process_mm = not is_decode and has_mm_inputs
        print(f"  - forward_mode: {forward_mode}")
        print(f"  - should_process_mm: {should_process_mm}")
        
        # Step 3: 获取文本嵌入层
        print(f"\n[Step 3] 获取文本嵌入层")
        embed_layer = torch.nn.Embedding(vocab_size + 10000, hidden_size)
        print(f"  - embed_layer: Embedding({vocab_size + 10000}, {hidden_size})")
        
        if should_process_mm:
            # Step 4: 准备多模态输入
            print(f"\n[Step 4] 准备多模态输入")
            mm_inputs_list = [{"offsets": [(image_start, image_end)]}]
            extend_prefix_lens = [0]
            extend_seq_lens = [total_tokens]
            print(f"  - mm_inputs_list: {len(mm_inputs_list)} 个请求")
            print(f"  - extend_prefix_lens: {extend_prefix_lens}")
            print(f"  - extend_seq_lens: {extend_seq_lens}")
            
            # Step 5: 调用 embed_mm_inputs (模拟)
            print(f"\n[Step 5] 调用 embed_mm_inputs (模拟)")
            
            # 5a. 获取文本嵌入
            clamped_ids = input_ids.clamp(max=vocab_size-1)
            text_embeds = embed_layer(clamped_ids)
            print(f"  5a. text_embeds.shape: {text_embeds.shape}")
            
            # 5b. 创建掩码
            mm_mask = (input_ids == image_placeholder)
            print(f"  5b. mm_mask.sum(): {mm_mask.sum().item()}")
            
            # 5c. 计算多模态嵌入 (模拟 ViT)
            image_embeds = torch.randn(num_image_tokens, hidden_size)
            print(f"  5c. image_embeds.shape: {image_embeds.shape} (模拟 ViT 输出)")
            
            # 5d. 替换
            input_embeds = text_embeds.clone()
            input_embeds[mm_mask] = image_embeds
            print(f"  5d. 替换完成, input_embeds.shape: {input_embeds.shape}")
        else:
            # 纯文本
            print(f"\n[Step 4] 纯文本处理")
            input_embeds = embed_layer(input_ids.clamp(max=vocab_size-1))
        
        # Step 6: 语言模型前向传播 (模拟)
        print(f"\n[Step 6] 语言模型前向传播 (模拟)")
        # hidden_states = language_model(input_embeds=input_embeds)
        hidden_states = input_embeds  # 简化模拟
        print(f"  - hidden_states.shape: {hidden_states.shape}")
        
        # 验证
        print(f"\n[验证]")
        self.assertEqual(hidden_states.shape, (total_tokens, hidden_size))
        print(f"  - Shape 正确: {hidden_states.shape == (total_tokens, hidden_size)}")
        
        print("\n" + "=" * 70)
        print("✓ 完整流程模拟成功!")
        print("=" * 70)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "SGLang 多模态嵌入流程测试" + " " * 20 + "#")
    print("#" * 70)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestGetEmbeddingChunk))
    suite.addTests(loader.loadTestsFromTestCase(TestMultimodalMask))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingReplacement))
    suite.addTests(loader.loadTestsFromTestCase(TestChunkedPrefillSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestFullPipelineSimulation))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"  - 运行测试: {result.testsRun}")
    print(f"  - 成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  - 失败: {len(result.failures)}")
    print(f"  - 错误: {len(result.errors)}")
    
    return result


if __name__ == '__main__':
    run_all_tests()
