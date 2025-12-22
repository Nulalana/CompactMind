# AutoLLM-Compressor (CompactMind)

一个可插拔、可扩展的完全本地化自动化大模型压缩框架。核心设计为“注册中心 + 搜索引擎 + 执行器 + 评估器”，支持 Llama-2 等大模型在本地环境下的自动压缩与评估。

## ✨ 核心特性

*   **完全本地化**: 支持加载本地模型（如 Llama-2）和本地数据集（WikiText-2），无需联网，安全稳定。
*   **混合方法搜索 (Hybrid Search)**: 自动搜索单一方法及组合方法（如“量化+剪枝”），寻找帕累托最优解。
*   **插件化架构**: 压缩算法（剪枝、量化）通过装饰器注册，零侵入扩展。
*   **自动搜索**: 内置 Grid Search 自动寻找最优压缩参数组合（Sparsity, Bits 等）。
*   **真实评估**: 基于 PPL (Perplexity) 的闭环评估，拒绝随机数据糊弄。
*   **可视化报告**: 自动生成压缩比 vs PPL 的散点图，直观展示不同方法的性能前沿。

## 📂 项目结构详解

```text
AutoLLM-Compressor/
├── core/                       # [核心引擎]
│   ├── compressor.py           # 压缩执行器：支持单方法和 Hybrid 组合执行
│   ├── engine.py               # 搜索引擎：实现 Grid Search 及混合方法搜索
│   ├── evaluator.py            # 评估器：计算模型的 PPL (困惑度)
│   └── __init__.py
│
├── methods/                    # [算法仓库]
│   ├── pruning/                # 剪枝算法
│   │   ├── random.py           # 随机剪枝 (Baseline)
│   │   ├── l2.py               # L2 结构化剪枝 (Structured Pruning)
│   │   └── __init__.py
│   ├── quantization/           # 量化算法
│   │   ├── fp16.py             # FP16 半精度量化
│   │   ├── int8_sq.py          # INT8-SQ 模拟量化 (SmoothQuant 思想)
│   │   └── __init__.py
│   ├── base.py                 # 算法基类，定义标准接口
│   ├── registry.py             # 注册中心，管理所有可用算法
│   └── __init__.py
│
├── utils/                      # [工具类]
│   ├── data_loader.py          # 数据加载器
│   └── __init__.py
│
├── scripts/                    # [辅助脚本]
│   ├── download_model.py       # 下载模型
│   ├── download_from_modelscope.py # 从 ModelScope 下载
│   └── download_data.py        # 下载数据集
│
├── results/                    # [结果输出]
│   └── run_YYYYMMDD/           # 独立运行目录
│       ├── report.json         # 详细实验报告
│       ├── search_history.csv  # 搜索过程数据
│       └── search_space_analysis.png # 搜索空间可视化图
│
├── main.py                     # [主入口] 项目启动文件
└── README.md                   # 项目文档
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install torch transformers datasets modelscope matplotlib
```

### 2. 准备模型与数据
*   **下载数据**:
    ```bash
    python scripts/download_data.py
    ```
*   **下载模型**:
    ```bash
    # 推荐：使用魔塔社区下载
    python scripts/download_from_modelscope.py
    ```

### 3. 运行项目
推荐使用 GPU 运行，并指定本地模型与数据路径：

```bash
# 示例：使用 GPU 运行，压缩比目标 0.8
python main.py --gpu \
  --model_path /path/to/Llama-2-7b-hf \
  --data_path /path/to/wikitext2/test.txt \
  --target_ratio 0.8
```

程序会自动：
1.  加载模型与数据。
2.  **单方法搜索**: 遍历所有单一方法（如 INT8, L2剪枝）及其参数。
3.  **混合方法搜索**: 遍历所有两步组合（如“INT8 + L2剪枝”）。
4.  **评估与择优**: 筛选出满足压缩比（如 0.8）且 PPL 最低的配置。
5.  输出最佳配置，并生成 `search_space_analysis.png` 图表。

### 4. 命令行参数详解

```bash
python main.py [options]
```

*   `--model_path`: 本地模型路径 (必须包含 config.json 等文件)
*   `--data_path`: 外部数据集路径 (如 wikitext2 的 test.txt)
*   `--target_ratio`: 目标压缩比 (0.0-1.0)，默认 0.5。程序会跳过压缩率不足的方案。
*   `--gpu`: 强制使用 GPU (推荐)。
*   `--cpu`: 强制使用 CPU (极慢，仅供调试)。
*   `--data_samples`: 校准样本数量 (默认 10)，显存紧张时可调小。
*   `--strategy`: 搜索策略，默认为 `grid` (网格搜索)。

## 📊 结果分析

运行结束后，检查 `results/run_xxx/` 目录：
*   **search_space_analysis.png**: 散点图。横轴为压缩比，纵轴为 PPL。
    *   🔵 蓝色点：单一方法
    *   🔺 红色点：混合方法 (Hybrid)
    *   你可以直观看到混合方法是否突破了单一方法的性能边界（帕累托前沿）。
*   **search_history.csv**: 所有尝试过的配置数据，可用于 Excel 分析。
*   **report.json**: 完整的机器可读报告。

## 🛠️ 如何导入新的压缩方法？

本项目采用**注册机制**，添加新算法只需简单三步：

### 第一步：新建文件
在 `methods/pruning/` 或 `methods/quantization/` 下新建一个 Python 文件（例如 `my_pruning.py`）。

### 第二步：编写算法类
继承 `BaseCompressionMethod` 并使用 `@register_method` 装饰器。

```python
# methods/pruning/my_pruning.py

from methods.base import BaseCompressionMethod
from methods.registry import register_method
import torch

# 1. 使用装饰器注册唯一名称
@register_method("my_custom_pruning")
class MyCustomPruning(BaseCompressionMethod):
    
    # 2. 实现核心压缩逻辑
    def apply(self, model, **kwargs):
        threshold = kwargs.get("threshold", 0.1)
        print(f"Applying My Pruning with threshold {threshold}...")
        # ... 这里写你的剪枝代码 ...
        return model
        
    # 3. 定义搜索空间 (供自动搜索使用)
    def get_info(self):
        return {
            "type": "pruning", 
            "search_space": {
                "threshold": [0.1, 0.3, 0.5] # 搜索引擎会尝试这些值
            }
        }
```

### 第三步：激活
在 `methods/pruning/__init__.py` 中添加一行导入：

```python
# methods/pruning/__init__.py
from .random import RandomPruning
from .my_pruning import MyCustomPruning  # <--- 新增这行
```

**完成！** 下次运行 `python main.py` 时，系统会自动将你的新算法加入单方法搜索和混合方法搜索中。

---
维护者：AutoLLM-Compressor 项目组
