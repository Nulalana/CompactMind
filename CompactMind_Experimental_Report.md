# 大模型自动压缩系统 (CompactMind) 实验报告

## 1. 摘要 (Abstract)

随着大语言模型 (LLMs) 参数规模的不断增长，其部署和推理成本已成为实际应用的主要瓶颈。虽然现有的剪枝 (Pruning) 和量化 (Quantization) 技术能够有效降低模型体积，但针对特定模型手动寻找最优的压缩参数组合（如剪枝稀疏度、量化位宽、平滑因子等）是一个耗时且依赖经验的过程。

本项目提出了 **CompactMind**，一个基于贝叶斯优化的自动化大模型压缩框架。该系统能够自适应地在搜索空间中探索最优的压缩策略，支持单一方法及“剪枝-再训练-量化”的混合压缩管线 (Hybrid Pipeline)。实验结果表明，CompactMind 能够自动发现接近最优的压缩配置，并在 LLaMA-2-7B 模型上验证了 SmoothQuant (W8A8) 的有效性，同时揭示了混合压缩中误差累积的挑战。

---

## 2. 引言 (Introduction)

### 2.1 背景
大模型（如 LLaMA-2-7B）虽然在各项任务上表现优异，但其巨大的显存占用和计算需求限制了在边缘设备或消费级显卡上的部署。为了解决这一问题，学术界提出了多种压缩技术，主要包括网络剪枝（移除冗余参数）和低比特量化（降低数值精度）。

### 2.2 动机
尽管已有 LLM-Pruner 和 SmoothQuant 等成熟算法，但在实际应用中面临以下挑战：
1.  **参数敏感性**：不同的模型对压缩参数（如剪枝率）的敏感度不同，手动调参效率极低。
2.  **方法组合难**：单一方法往往有极限，混合使用（如先剪枝后量化）理论上能获得更高压缩比，但容易导致精度崩塌（Collapse），缺乏自动化的探索工具。

### 2.3 创新点
本项目（CompactMind）的主要贡献在于：
*   **范式迁移 (Paradigm Shift)**：借鉴 **ProxylessNAS** 的思想，构建了针对“压缩方法”的搜索空间，实现了从“人工设计压缩流程”到“自动化搜索流程”的转变。
*   **自动化搜索框架**：引入贝叶斯优化（Bayesian Optimization）替代传统的网格搜索，高效寻找帕累托最优解。
*   **混合压缩管线**：设计了可配置的 Pipeline 执行引擎，支持 `Pruning -> Retraining -> Quantization` 的全链路压缩。
*   **模块化设计**：采用“注册中心-执行器”架构，便于集成新的压缩算法。

---

## 3. 系统设计与方法 (Methodology)

CompactMind 的核心架构由四个主要模块组成：**搜索引擎 (Search Engine)**、**压缩执行器 (Compressor)**、**评估器 (Evaluator)** 和 **算法库 (Methods Registry)**。

### 3.1 核心引擎：贝叶斯搜索 (Bayesian Search)
核心代码位于 `core/engine.py`。我们集成了 **Optuna** 框架来实现贝叶斯优化，目标是在给定的约束（如目标压缩比）下最小化模型的困惑度 (PPL)。

搜索策略 (`_search_bayesian`) 定义了复杂的搜索空间：
1.  **模式选择**：系统首先决定采用 `single`（单一方法）还是 `hybrid`（混合方法）模式。
2.  **参数采样**：
    *   **剪枝阶段**：搜索结构化剪枝的稀疏度 (`sparsity`)。
    *   **量化阶段**：搜索量化平滑因子 (`alpha`) 等超参数。
    *   **再训练开关**：系统根据当前配置自动决策是否开启轻量级微调 (`enable_retrain`) 以恢复精度。

```python
# 代码片段示意 (core/engine.py)
def objective(trial):
    mode = trial.suggest_categorical("mode", ["single", "hybrid"])
    if mode == "hybrid":
        # 自动采样剪枝和量化参数
        p_method = trial.suggest_categorical("hybrid_p_method", prune_methods)
        q_method = trial.suggest_categorical("hybrid_q_method", quant_methods)
        # 动态构建 Pipeline
        pipeline_list = [{"method": p_method}, {"method": q_method}]
```

### 3.2 压缩算法实现
我们复现并集成了两种主流算法作为搜索基元：

1.  **结构化剪枝 (LLM-Pruner)**：
    *   实现于 `methods/pruning/l2.py`。
    *   通过分析参数依赖图（Dependency Graph），移除整行或整列的权重，确保模型结构完整，适合硬件加速。
    *   使用 L2 范数评估重要性。

2.  **平滑量化 (SmoothQuant)**：
    *   实现于 `methods/quantization/int8_sq.py`。
    *   通过数学变换 $W = W \cdot s, X = X / s$ 将激活值的量化难度迁移到权重上，解决了激活值异常点（Outliers）问题，实现了 W8A8 的低精度推理。

### 3.3 轻量级再训练 (Retraining)
为了解决混合压缩带来的精度损失，我们在 `methods/retraining/finetuning.py` 中实现了基于 Causal Language Modeling (CLM) 的微调模块。在剪枝后，系统可自动触发短周期的微调，帮助模型适应结构变化。

---

## 4. 实验设置 (Experimental Setup)

*   **模型**: LLaMA-2-7B-HF
*   **数据集**: WikiText-2 (用于校准和 PPL 评估), GSM8K (用于能力验证)
*   **硬件环境**: NVIDIA GPU (CUDA) / CPU (作为 Fallback)
*   **评估指标**: Perplexity (PPL, 越低越好), Compression Ratio (压缩比)

---

## 5. 实验结果与分析 (Results & Analysis)

### 5.1 单一方法评估
首先验证了各组件的独立性能。

| 方法 | 配置 | PPL (WikiText-2) | 结论 |
| :--- | :--- | :--- | :--- |
| Baseline | FP16 (Original) | **5.47** | 基准性能 |
| Quantization | INT8-SQ (W8A8) | **5.49** (+0.002) | **精度近乎无损**，显存减少约 48%。 |
| Pruning | L2 Structured (20%) | > 100 (无微调) | 结构化剪枝对模型破坏极大，必须配合微调。 |

**分析**：SmoothQuant 在 LLaMA-2-7B 上表现极其出色，验证了 PPT 中提到的“INT8-SQ 是一种高效可落地的 PTQ 方案”。而结构化剪枝直接移除参数会导致模型能力崩溃，必须结合 Retraining 模块使用。

### 5.2 自动化搜索与混合压缩
使用 CompactMind 的贝叶斯搜索功能，设定 `n_trials=30` 进行自动探索。

*   **发现 1：混合压缩的误差累积**
    系统尝试了 `L2 Pruning (10%) -> INT8 Quantization` 的组合。结果显示 PPL 显著上升（>20）。这表明剪枝带来的激活值分布变化破坏了 SmoothQuant 的平滑假设。

*   **发现 2：搜索收敛性**
    Optuna 在约 15 次尝试后迅速收敛，倾向于选择 **纯量化 (INT8-SQ)** 或 **极低稀疏度的剪枝**。这证明了贝叶斯优化能够有效地避开“高稀疏度+量化”这种会导致模型崩溃的参数区域。

*   **发现 3：再训练的必要性**
    在开启 `--retrain True` 后，混合压缩的性能有显著回升，但训练成本（时间）增加了约 10 倍。

---

## 6. 总结与展望 (Conclusion)

本项目成功开发了 **CompactMind**，一个集成了剪枝、量化和微调的自动化大模型压缩系统。
1.  **工程实现**：我们完成了一个模块化、可扩展的代码库，支持通过 `main.py` 一键启动自动化搜索。
2.  **算法验证**：复现了 SmoothQuant 和 LLM-Pruner，确认了量化在当前模型下的优势。
3.  **系统价值**：自动化搜索机制成功识别出了最优策略（在当前约束下为 INT8-SQ），避免了人工试错的成本。

**局限性与未来工作**：
目前混合压缩（剪枝+量化）的效果受限于简单的微调策略。未来计划引入 **知识蒸馏 (Knowledge Distillation)** 模块，利用原始大模型作为教师模型来指导压缩后模型的恢复，以期在更高压缩比下保持模型性能。

---

### 附录：核心命令
```bash
# 启动贝叶斯搜索实验
python main.py --model_path ./models/Llama-2-7b-hf --strategy bayesian --n_trials 30 --retrain True
```
---

### 附录：核心命令
```bash
# 启动贝叶斯搜索实验
python main.py --model_path ./models/Llama-2-7b-hf --strategy bayesian --n_trials 30 --retrain True
```
