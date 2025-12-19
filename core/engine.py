from typing import Dict, Any, List
import torch.nn as nn
import itertools
import copy
from methods.registry import COMPRESSION_REGISTRY, get_method
from core.compressor import Compressor
from core.evaluator import Evaluator

class SearchEngine:
    """
    搜索引擎核心：负责在搜索空间中寻找最佳压缩配置。
    """
    def __init__(self, search_strategy: str = "grid", evaluator: Evaluator = None):
        self.search_strategy = search_strategy
        self.available_methods = list(COMPRESSION_REGISTRY.keys())
        self.evaluator = evaluator

    def search(self, model: nn.Module, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据约束条件搜索最佳压缩方案。
        constraints: 包含 'target_ratio' (float)
        """
        print(f"Starting search with strategy: {self.search_strategy}")
        target_ratio = constraints.get("target_ratio", 1.0)
        print(f"Target Compression Ratio: <= {target_ratio}")
        
        best_score = float('inf') # PPL 越低越好
        best_config = None
        
        # 遍历所有注册的压缩方法
        for method_name in self.available_methods:
            method_class = get_method(method_name)
            temp_instance = method_class({}) 
            info = temp_instance.get_info()
            
            search_space = info.get("search_space", {})
            print(f"Searching method: {method_name}, Space: {search_space}")
            
            # 生成网格搜索的所有参数组合
            param_names = list(search_space.keys())
            param_values = list(search_space.values())
            
            # itertools.product 生成笛卡尔积
            for combination in itertools.product(*param_values):
                current_params = dict(zip(param_names, combination))
                
                # 构造当前配置
                config = {
                    "method": method_name,
                    "params": current_params
                }
                
                # 1. [关键步骤] 预估压缩比
                estimated_ratio = self._estimate_compression_ratio(method_name, current_params)
                
                # 2. [筛选] 检查是否满足用户目标
                if estimated_ratio > target_ratio:
                    print(f"  [Skip] {config['method']} {config['params']} (Ratio {estimated_ratio:.2f} > {target_ratio})")
                    continue
                
                # 3. [评估] 只有满足条件的才跑 PPL
                score = self._evaluate_candidate(model, config)
                print(f"  [Eval] {config['method']} {config['params']} (Ratio {estimated_ratio:.2f}) -> PPL: {score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_config = config
        
        print(f"Search completed. Best Score: {best_score:.4f}")
        return best_config

    def _estimate_compression_ratio(self, method_name: str, params: Dict[str, Any]) -> float:
        """
        估算给定配置相对于当前 FP16 模型的压缩比。
        """
        name = method_name.lower()
        
        # 1. Pruning 类
        if "pruning" in name or "prune" in name:
            # 剪枝: Ratio = 1 - sparsity
            sparsity = params.get("sparsity", 0.0)
            return 1.0 - sparsity
            
        # 2. Quantization 类
        if "int8" in name:
            # INT8 (8bit) / FP16 (16bit) = 0.5
            return 0.5
        if "int4" in name:
            # INT4 (4bit) / FP16 (16bit) = 0.25
            return 0.25
        if "fp16" in name:
            # FP16 / FP16 = 1.0
            return 1.0
            
        # 默认不压缩
        return 1.0

    def _evaluate_candidate(self, model: nn.Module, config: Dict[str, Any]) -> float:
        """
        评估单个候选配置。
        """
        # 深拷贝模型以防修改原模型
        model_copy = copy.deepcopy(model)
        
        compressor = Compressor()
        compressed_model = compressor.run(model_copy, config)
        
        if self.evaluator:
            return self.evaluator.evaluate_perplexity(compressed_model)
        else:
            return 0.0
