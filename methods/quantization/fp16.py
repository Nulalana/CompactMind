from methods.base import BaseCompressionMethod
from methods.registry import register_method
import torch

@register_method("fp16")
class FP16Quantization(BaseCompressionMethod):
    def apply(self, model, **kwargs):
        print("Applying FP16 Quantization (converting model to half precision)...")
        # 直接调用 PyTorch 的 half() 方法
        # 注意：这会真正改变模型精度和显存占用
        model.half()
        return model

    def get_info(self):
        return {
            "type": "quantization",
            "search_space": {} # FP16 没有可搜索参数，是一个确定性操作
        }
