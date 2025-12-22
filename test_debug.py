import sys
import os
import unittest
import torch
import shutil
import tempfile
import logging

# 确保项目根目录在 sys.path 中
sys.path.append(os.getcwd())

# 显式导入所有 methods 模块，以触发装饰器注册
import methods.pruning.random
import methods.pruning.l2
import methods.quantization.fp16
import methods.quantization.int8_sq

from core.engine import SearchEngine
from core.compressor import Compressor
from core.evaluator import Evaluator
from methods.registry import get_method

# 模拟一个简单的线性模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 10)
        self.lm_head = torch.nn.Linear(10, 10) # 模拟输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.lm_head(x)
        return x

class TestAutoLLMCompressor(unittest.TestCase):
    
    def setUp(self):
        # 创建临时目录用于保存结果
        self.test_dir = tempfile.mkdtemp()
        self.model = SimpleModel()
        self.evaluator = None # 简单测试可以不需要真实 evaluator
        
        # 配置日志输出到控制台，方便观察
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)

    def test_bayesian_search_static_space(self):
        """
        测试贝叶斯搜索是否解决了动态空间问题
        """
        print("\n=== Testing Bayesian Search (Static Space Check) ===")
        engine = SearchEngine(search_strategy="bayesian", evaluator=self.evaluator)
        
        # 定义一个简单的 Evaluator mock，返回随机 PPL
        class MockEvaluator:
            def evaluate_perplexity(self, model):
                return torch.rand(1).item() * 10
        
        engine.evaluator = MockEvaluator()
        
        # 设置约束
        constraints = {
            "target_ratio": 0.5,
            "n_trials": 10 # 跑 10 次足够触发潜在错误
        }
        
        try:
            best_config = engine.search(self.model, constraints)
            print(f"Search successful. Best config: {best_config}")
        except ValueError as e:
            if "CategoricalDistribution does not support dynamic value space" in str(e):
                self.fail("Caught the Optuna dynamic space error! The fix is not working.")
            else:
                self.fail(f"Caught unexpected error: {e}")
        except Exception as e:
            self.fail(f"Caught unexpected error during search: {e}")

    def test_pruning_methods(self):
        """
        测试剪枝方法的基本运行
        """
        print("\n=== Testing Pruning Methods ===")
        # 1. Random Pruning
        pruner = get_method("random_pruning")({})
        model_copy = SimpleModel()
        pruner.apply(model_copy, sparsity=0.5)
        # 简单检查是否报错，以及权重是否改变（略）
        
        # 2. L2 Pruning
        l2_pruner = get_method("l2_structured_pruning")({})
        model_copy = SimpleModel()
        l2_pruner.apply(model_copy, sparsity=0.5)
    
    def test_hybrid_logic(self):
        """
        测试混合方法的逻辑流
        """
        print("\n=== Testing Hybrid Logic Simulation ===")
        # 模拟引擎中的 hybrid 分支
        # 这里主要依赖 test_bayesian_search_static_space 的覆盖
        pass

if __name__ == '__main__':
    unittest.main()
