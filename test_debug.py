import sys
import os
import unittest
import torch
import shutil
import tempfile
import logging
from dataclasses import dataclass

# 确保项目根目录在 sys.path 中
sys.path.append(os.getcwd())

# 显式导入所有 methods 模块，以触发装饰器注册
import methods.pruning.random
import methods.pruning.l2
import methods.quantization.fp16
import methods.quantization.int8_sq
import methods.retraining.finetuning

from core.engine import SearchEngine
from core.compressor import Compressor
from core.evaluator import Evaluator
from methods.registry import get_method

# Mock 一个符合 CausalLM 接口的模型
class MockLlamaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 词表大小 100，维度 32
        self.embed_tokens = torch.nn.Embedding(100, 32)
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(32, 32) for _ in range(2)
        ])
        self.norm = torch.nn.LayerNorm(32)
        self.lm_head = torch.nn.Linear(32, 100)
        
        # Mock config
        self.config = type('Config', (), {'use_cache': True})()

    def forward(self, input_ids, labels=None, **kwargs):
        # input_ids: (Batch, SeqLen)
        if input_ids.dim() != 2:
            # 这里的检查可以帮助我们复现 "Unpack error" 或维度问题
            # 如果传入的是 (B, 1, L)，Embedding 会报错吗？会变成 (B, 1, L, D)
            pass
            
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # 简单模拟 loss，必须 requires_grad
            loss = logits.mean() 
            
        return type('Output', (), {'loss': loss, 'logits': logits})()
        
    def gradient_checkpointing_enable(self):
        pass

class TestAutoLLMCompressor(unittest.TestCase):
    
    def setUp(self):
        # 创建临时目录用于保存结果
        self.test_dir = tempfile.mkdtemp()
        self.model = MockLlamaModel()
        self.evaluator = None 
        
        # 配置日志输出到控制台
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    def tearDown(self):
        # 清理临时目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_finetuning_method(self):
        """
        测试 Finetuning 方法，重点测试 DataLoader 和 input_ids 维度问题
        """
        print("\n=== Testing Finetuning Method ===")
        finetuner = get_method("finetuning")({})
        
        # 构造模拟数据集: List of (1, seq_len) tensors
        # 模拟真实情况：tokenizer 输出通常是 (1, L)
        seq_len = 10
        dataset = [torch.randint(0, 100, (1, seq_len)) for _ in range(20)]
        
        # 调用 apply
        # 期望：finetuning 内部能正确处理 (1, L) 数据，将其 squeeze 为 (L,)
        # 或者是 DataLoader collate 后变成 (B, 1, L)，然后在 forward 前被 squeeze
        try:
            finetuner.apply(self.model, dataset=dataset, epochs=1.0, learning_rate=1e-3)
            print("Finetuning executed successfully.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Finetuning failed: {e}")

    def test_bayesian_search_with_finetuning(self):
        """
        测试贝叶斯搜索能否跑通包含 Finetuning 的混合管线
        """
        print("\n=== Testing Bayesian Search with Hybrid Pipeline ===")
        
        # Mock Evaluator
        class MockEvaluator:
            def __init__(self):
                # 构造 dataset 用于 finetuning
                self.dataset = [torch.randint(0, 100, (1, 10)) for _ in range(10)]
            
            def evaluate_perplexity(self, model):
                return 10.0 # 固定 PPL

        mock_evaluator = MockEvaluator()
        engine = SearchEngine(search_strategy="bayesian", evaluator=mock_evaluator)
        
        # 强制启用 retrain
        constraints = {
            "n_trials": 2, # 跑两次
            "enable_retrain": True
        }
        
        try:
            best_config = engine.search(self.model, constraints)
            print(f"Search successful. Best config: {best_config}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Search failed: {e}")

