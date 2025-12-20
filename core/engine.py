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
        
        # 记录所有尝试过的组合及其结果，用于生成报告
        search_history = []
        
        # 1. 单方法搜索
        for method_name in self.available_methods:
            method_class = get_method(method_name)
            temp_instance = method_class({}) 
            info = temp_instance.get_info()
            
            search_space = info.get("search_space", {})
            print(f"Searching method: {method_name}, Space: {search_space}")
            
            param_names = list(search_space.keys())
            param_values = list(search_space.values())
            
            for combination in itertools.product(*param_values):
                current_params = dict(zip(param_names, combination))
                config = {
                    "method": method_name,
                    "params": current_params
                }
                
                estimated_ratio = self._estimate_compression_ratio(method_name, current_params)
                
                if estimated_ratio > target_ratio:
                    print(f"  [Skip] {config['method']} {config['params']} (Ratio {estimated_ratio:.2f} > {target_ratio})")
                    continue
                
                score = self._evaluate_candidate(model, config)
                print(f"  [Eval] {config['method']} {config['params']} (Ratio {estimated_ratio:.2f}) -> PPL: {score:.4f}")
                
                # 记录结果
                search_history.append({
                    "config": config,
                    "ratio": estimated_ratio,
                    "ppl": score,
                    "type": "single"
                })
                
                if score < best_score:
                    best_score = score
                    best_config = config

        # 2. 混合方法搜索 (Pipeline Search)
        # 组合策略：尝试 [量化 -> 剪枝] 和 [剪枝 -> 量化] 的所有可能组合
        # 限制：仅组合两种不同类型的方法，避免搜索空间爆炸
        print("\n--- Starting Pipeline Search (Hybrid Methods) ---")
        
        methods_info = {}
        for m in self.available_methods:
            cls = get_method(m)
            inst = cls({})
            methods_info[m] = inst.get_info()
            
        quant_methods = [m for m in self.available_methods if methods_info[m].get("type") == "quantization"]
        prune_methods = [m for m in self.available_methods if methods_info[m].get("type") == "pruning"]
        
        # 生成所有可能的 pipeline 组合
        # 顺序 1: 量化 -> 剪枝
        pipelines_to_search = []
        for q in quant_methods:
            for p in prune_methods:
                pipelines_to_search.append([q, p]) # Quant -> Prune
                pipelines_to_search.append([p, q]) # Prune -> Quant
        
        for p_methods in pipelines_to_search:
            # 构建该 pipeline 的参数空间笛卡尔积
            # 例如: Quant(alpha) x Prune(sparsity)
            
            # 获取每个方法的参数空间
            spaces = [methods_info[m].get("search_space", {}) for m in p_methods]
            param_names_list = [list(s.keys()) for s in spaces]
            param_values_list = [list(s.values()) for s in spaces]
            
            # 生成每一步的参数组合
            # step1_combos: [params1_a, params1_b, ...]
            # step2_combos: [params2_a, params2_b, ...]
            step_combos_list = []
            for i in range(len(p_methods)):
                if not param_values_list[i]: # 无参数方法 (如 fp16)
                    step_combos_list.append([{}])
                else:
                    # 生成该步骤的所有参数组合
                    combos = []
                    for val_combo in itertools.product(*param_values_list[i]):
                        combos.append(dict(zip(param_names_list[i], val_combo)))
                    step_combos_list.append(combos)
            
            # 生成整个 pipeline 的参数组合 (Step 1 x Step 2)
            for pipeline_params in itertools.product(*step_combos_list):
                # pipeline_params 是一个元组，包含每一步的参数字典 ({'alpha':...}, {'sparsity':...})
                
                # 构造 pipeline 配置
                current_pipeline = []
                total_ratio = 1.0
                
                for i, method_name in enumerate(p_methods):
                    params = pipeline_params[i]
                    current_pipeline.append({
                        "method": method_name,
                        "params": params
                    })
                    total_ratio *= self._estimate_compression_ratio(method_name, params)
                
                pipeline_config = {"pipeline": current_pipeline}
                pipeline_name = "+".join(p_methods)
                pipeline_param_str = str(pipeline_params)
                
                # 筛选
                if total_ratio > target_ratio:
                    print(f"  [Skip] Pipeline {pipeline_name} {pipeline_param_str} (Ratio {total_ratio:.2f} > {target_ratio})")
                    continue
                
                # 评估
                score = self._evaluate_candidate(model, pipeline_config)
                print(f"  [Eval] Pipeline {pipeline_name} {pipeline_param_str} (Ratio {total_ratio:.2f}) -> PPL: {score:.4f}")
                
                # 记录结果
                search_history.append({
                    "config": pipeline_config,
                    "ratio": total_ratio,
                    "ppl": score,
                    "type": "pipeline"
                })
                
                if score < best_score:
                    best_score = score
                    best_config = pipeline_config
        
        print(f"Search completed. Best Score: {best_score:.4f}")
        # 将历史记录附加到 best_config 中返回，以便上层保存到报告
        if best_config:
            best_config["search_history"] = search_history
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
        # 深拷贝模型会引发 OOM，改用 state_dict 备份恢复策略
        # 1. 将当前状态备份到 CPU 以节省显存
        original_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        
        compressor = Compressor()
        try:
            # 2. 就地执行压缩（注意：compressor 需要支持就地修改）
            compressed_model = compressor.run(model, config)
            
            if self.evaluator:
                return self.evaluator.evaluate_perplexity(compressed_model)
            else:
                return 0.0
        except Exception as e:
            print(f"Evaluation failed for {config}: {e}")
            return float('inf')
        finally:
            # 3. 恢复原始权重，确保不影响后续搜索
            model.load_state_dict(original_state)
