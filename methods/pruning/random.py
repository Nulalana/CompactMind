import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from methods.base import BaseCompressionMethod
from methods.registry import register_method

@register_method("random_pruning")
class RandomPruning(BaseCompressionMethod):
    def apply(self, model, **kwargs):
        sparsity = kwargs.get("sparsity", 0.5)
        print(f"Applying Random Pruning with sparsity: {sparsity}")
        
        # 遍历所有 Linear 层进行随机剪枝
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 使用 PyTorch 内置的 random_unstructured 剪枝
                prune.random_unstructured(module, name="weight", amount=sparsity)
                # 永久固化剪枝（移除 mask，直接修改 weight）
                prune.remove(module, "weight")
                
        return model

    def get_info(self):
        return {
            "type": "pruning",
            "search_space": {
                "sparsity": [0.1, 0.3, 0.5, 0.7, 0.9] # 搜索引擎会尝试这些值
            }
        }
