import torch.nn as nn
from typing import Dict, Any
from methods.registry import get_method

class Compressor:
    """
    压缩执行器：根据 SearchEngine 返回的配置执行具体操作。
    """
    def run(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        if "pipeline" in config:
            # 混合模式：按顺序执行多个压缩步骤
            for step in config["pipeline"]:
                mname = step.get("method")
                params = step.get("params", {})
                
                print(f"Initializing compression method: {mname}")
                mclass = get_method(mname)
                instance = mclass(params)
                
                print("Executing compression step...")
                model = instance.apply(model, **params)
            return model
        else:
            # 单一模式：兼容旧配置
            method_name = config.get("method")
            params = config.get("params", {})
            
            print(f"Initializing compression method: {method_name}")
            method_class = get_method(method_name)
            compressor_instance = method_class(params)
            
            print("Executing compression...")
            compressed_model = compressor_instance.apply(model, **params)
            
            return compressed_model
