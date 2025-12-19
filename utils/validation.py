"""
输入验证工具模块
用于验证配置文件、参数等输入的有效性
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json


class ValidationError(Exception):
    """验证错误异常"""
    pass


def validate_compression_config(config: Dict[str, Any]) -> None:
    """
    验证压缩配置的有效性
    
    Args:
        config: 压缩配置字典
        
    Raises:
        ValidationError: 如果配置无效
    """
    if not isinstance(config, dict):
        raise ValidationError(f"配置必须是字典类型，得到 {type(config)}")
    
    # 验证量化配置
    if 'quantization' in config:
        quant_config = config['quantization']
        if not isinstance(quant_config, dict):
            raise ValidationError("quantization 配置必须是字典类型")
        
        if 'bits' in quant_config:
            bits = quant_config['bits']
            if not isinstance(bits, dict):
                raise ValidationError("quantization.bits 必须是字典类型")
            
            valid_layers = {'embedding', 'attention', 'linear', 'norm'}
            valid_bits = {2, 4, 6, 8, 16, 32}
            
            for layer, bit_value in bits.items():
                if layer not in valid_layers:
                    raise ValidationError(f"无效的层类型: {layer}，有效值: {valid_layers}")
                if not isinstance(bit_value, int) or bit_value not in valid_bits:
                    raise ValidationError(f"无效的比特数: {bit_value}，有效值: {valid_bits}")
    
    # 验证剪枝配置
    if 'pruning' in config:
        prune_config = config['pruning']
        if not isinstance(prune_config, dict):
            raise ValidationError("pruning 配置必须是字典类型")
        
        if 'ratio' in prune_config:
            ratios = prune_config['ratio']
            if not isinstance(ratios, dict):
                raise ValidationError("pruning.ratio 必须是字典类型")
            
            for layer, ratio_value in ratios.items():
                if not isinstance(ratio_value, (int, float)):
                    raise ValidationError(f"剪枝比例必须是数字类型: {layer}")
                if not (0 <= ratio_value < 1):
                    raise ValidationError(f"剪枝比例必须在 [0, 1) 范围内: {layer} = {ratio_value}")
    
    # 验证注意力优化配置
    if 'attention_optimization' in config:
        attn_config = config['attention_optimization']
        if not isinstance(attn_config, dict):
            raise ValidationError("attention_optimization 配置必须是字典类型")
        
        if 'kv_cache_quantization' in attn_config:
            kv_bits = attn_config['kv_cache_quantization']
            if not isinstance(kv_bits, int) or kv_bits not in {4, 8, 16, 32}:
                raise ValidationError(f"无效的KV缓存量化比特数: {kv_bits}")


def validate_json_file(file_path: Union[str, Path], schema: Optional[Dict] = None) -> Dict[str, Any]:
    """
    验证并加载JSON文件
    
    Args:
        file_path: JSON文件路径
        schema: 可选的JSON schema（未来可扩展）
        
    Returns:
        解析后的JSON字典
        
    Raises:
        ValidationError: 如果文件不存在、格式错误或内容无效
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValidationError(f"文件不存在: {file_path}")
    
    if not file_path.is_file():
        raise ValidationError(f"路径不是文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"JSON格式错误: {file_path}, 错误: {e}")
    except Exception as e:
        raise ValidationError(f"读取文件失败: {file_path}, 错误: {e}")
    
    if schema is not None:
        # 未来可以实现JSON schema验证
        pass
    
    return data


def validate_positive_number(value: Union[int, float], name: str = "value") -> None:
    """
    验证数值是否为正数
    
    Args:
        value: 要验证的数值
        name: 参数名称（用于错误消息）
        
    Raises:
        ValidationError: 如果数值不是正数
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} 必须是数字类型，得到 {type(value)}")
    
    if value <= 0:
        raise ValidationError(f"{name} 必须是正数，得到 {value}")


def validate_range(value: Union[int, float], min_val: float, max_val: float, 
                   name: str = "value") -> None:
    """
    验证数值是否在指定范围内
    
    Args:
        value: 要验证的数值
        min_val: 最小值
        max_val: 最大值
        name: 参数名称（用于错误消息）
        
    Raises:
        ValidationError: 如果数值不在范围内
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} 必须是数字类型，得到 {type(value)}")
    
    if not (min_val <= value <= max_val):
        raise ValidationError(f"{name} 必须在 [{min_val}, {max_val}] 范围内，得到 {value}")

