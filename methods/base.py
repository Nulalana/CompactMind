from abc import ABC, abstractmethod
from typing import Any, Dict
import torch.nn as nn

class BaseCompressionMethod(ABC):
    """
    所有压缩方法的基类。
    用户自定义的压缩方法必须继承此类并实现 apply 方法。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        执行压缩逻辑。
        :param model: 待压缩的 PyTorch 模型
        :return: 压缩后的模型
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        返回该方法的元数据（如名称、类型、参数空间等），用于搜索算法。
        """
        pass
