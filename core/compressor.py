import torch.nn as nn
from typing import Dict, Any
from methods.registry import get_method

class Compressor:
    """
    压缩执行器：根据 SearchEngine 返回的配置执行具体操作。
    """
    def run(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        method_name = config.get("method")
        params = config.get("params", {})
        
        print(f"Initializing compression method: {method_name}")
        method_class = get_method(method_name)
        compressor_instance = method_class(params)
        
        print("Executing compression...")
        compressed_model = compressor_instance.apply(model)
        
        return compressed_model
