import inspect
from typing import Dict, Type, Any

# 全局注册表，存储所有可用的压缩方法
COMPRESSION_REGISTRY: Dict[str, Type['BaseCompressionMethod']] = {}

def register_method(name: str):
    """
    装饰器：用于将新的压缩方法注册到系统中。
    使用方法:
    @register_method("my_quantization")
    class MyQuantization(BaseCompressionMethod):
        ...
    """
    def _register(cls):
        if name in COMPRESSION_REGISTRY:
            raise ValueError(f"Method {name} already registered!")
        COMPRESSION_REGISTRY[name] = cls
        return cls
    return _register

def get_method(name: str) -> Type['BaseCompressionMethod']:
    if name not in COMPRESSION_REGISTRY:
        raise ValueError(f"Method {name} not found in registry. Available: {list(COMPRESSION_REGISTRY.keys())}")
    return COMPRESSION_REGISTRY[name]
