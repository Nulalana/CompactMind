import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from methods.base import BaseCompressionMethod
from methods.registry import register_method

@register_method("l2_structured_pruning")
class L2StructuredPruning(BaseCompressionMethod):
    def apply(self, model, **kwargs):
        sparsity = kwargs.get("sparsity", 0.5)
        print(f"Applying L2 Structured Pruning with sparsity: {sparsity}")
        
        # 敏感层关键词列表，避免剪枝这些层
        sensitive_keywords = ["lm_head", "embed_tokens", "wte", "wpe"]
        
        for name, module in model.named_modules():
            # 检查是否为敏感层
            if any(k in name for k in sensitive_keywords):
                # print(f"  Skipping sensitive layer: {name}")
                continue

            if isinstance(module, nn.Linear):
                # 结构化剪枝：按 L2 范数剪掉整行 (dim=0) 或 整列 (dim=1)
                # 这里我们剪掉输出维度 (dim=0)，相当于剪掉神经元
                try:
                    prune.ln_structured(
                        module, 
                        name="weight", 
                        amount=sparsity, 
                        n=2,       # L2 norm
                        dim=0      # Prune output channels
                    )
                    prune.remove(module, "weight")
                except Exception as e:
                    # 有些层的维度可能不支持，跳过
                    print(f"Skipping layer {name}: {e}")
                    
        return model

    def get_info(self):
        return {
            "type": "pruning",
            "search_space": {
                "sparsity": [0.25, 0.5, 0.75]
            }
        }
