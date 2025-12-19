from methods.base import BaseCompressionMethod
from methods.registry import register_method
import torch
import torch.nn as nn

@register_method("int8_sq")
class INT8SQQuantization(BaseCompressionMethod):
    def apply(self, model, **kwargs):
        # 获取 alpha 参数，默认为同事推荐的 0.85
        alpha = kwargs.get("alpha", 0.85)
        print(f"Applying Simulated INT8-SQ Quantization with alpha={alpha}...")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._quantize_layer(module, alpha)
                
        return model

    def _quantize_layer(self, layer, alpha):
        """
        对单层进行模拟量化 (Fake Quantization)
        W_quant = round(clip(W / scale) * scale
        """
        w = layer.weight.data
        
        # 1. 计算最大绝对值
        max_val = w.abs().max()
        
        # 2. 根据 alpha 计算截断阈值 (模拟 SmoothQuant 中对离群点的处理)
        # 如果 alpha=0.85，意味着我们只关注 85% 的幅值范围，舍弃极端的 15%
        # 这有助于提高量化分辨率
        threshold = max_val * alpha
        
        # 3. 计算缩放因子 (Scale)
        # INT8 范围是 [-127, 127]
        scale = threshold / 127.0
        
        # 4. 模拟量化过程：FP -> INT8 -> FP
        # clamp: 截断到 [-threshold, threshold]
        # div(scale).round(): 量化到整数
        # mul(scale): 反量化回浮点
        w_quant = (w.clamp(-threshold, threshold) / scale).round() * scale
        
        # 5. 直接修改权重（原地更新）
        layer.weight.data = w_quant

    def get_info(self):
        return {
            "type": "quantization",
            "search_space": {
                # 搜索引擎可以尝试不同的 alpha 值
                "alpha": [0.85, 0.90, 0.95, 0.99] 
            }
        }
