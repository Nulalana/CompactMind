import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from methods.base import BaseCompressionMethod
from methods.registry import register_method
import copy

@register_method("finetuning")
class CausalLMFinetuning(BaseCompressionMethod):
    def apply(self, model, **kwargs):
        dataset = kwargs.get("dataset", None)
        if dataset is None:
            print("Warning: No dataset provided for finetuning. Skipping.")
            return model
            
        learning_rate = kwargs.get("learning_rate", 5e-5)
        num_epochs = kwargs.get("epochs", 0.5)
        
        print(f"Applying Finetuning with lr={learning_rate}, epochs={num_epochs}")
        
        # 1. 准备数据
        # 假设 dataset 是一个 list of tensors (input_ids)
        # 我们需要将其包装成 DataLoader
        # 如果 epochs 是小数（如 0.5），我们只迭代部分数据
        
        batch_size = 4 # 保持较小以节省显存
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_steps = len(data_loader)
        target_steps = int(total_steps * num_epochs)
        if target_steps < 1:
            target_steps = 1
            
        print(f"Total steps in dataset: {total_steps}. Target steps for this run: {target_steps}")

        # 2. 准备优化器
        # 只训练 float 类型的参数，跳过已量化的参数（如果有）
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            print("No trainable parameters found. Skipping finetuning.")
            return model
            
        # 显存优化：开启梯度检查点 (Gradient Checkpointing)
        if hasattr(model, "gradient_checkpointing_enable"):
            print("Enabling gradient checkpointing to save memory...")
            model.gradient_checkpointing_enable()
            # 必须设置 use_cache=False 才能配合 checkpointing
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        # 显存优化：使用 SGD 代替 AdamW
        # AdamW 需要维护 2 个状态量 (exp_avg, exp_avg_sq)，显存占用是模型参数的 2-3 倍
        # SGD 无状态，极其节省显存
        print("Using SGD optimizer to save memory (AdamW is too heavy for full finetuning)...")
        optimizer = torch.optim.SGD(trainable_params, lr=learning_rate)
        
        # 3. 训练循环
        torch.cuda.empty_cache() # 训练前清理显存
        model.train()
        device = next(model.parameters()).device
        
        global_step = 0
        running_loss = 0.0
        
        progress_bar = tqdm(total=target_steps, desc="Finetuning")
        
        # 支持 epoch 循环
        # 如果 num_epochs < 1，只跑一部分
        # 如果 num_epochs > 1，跑多轮
        
        # 计算实际需要跑多少个完整 epoch 和剩余步数
        # 这里简化处理：直接无限循环 data_loader 直到达到 target_steps
        
        data_iter = iter(data_loader)
        
        while global_step < target_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader) # 新的 epoch
                batch = next(data_iter)
            
            # 将数据移到设备
            if isinstance(batch, torch.Tensor):
                input_ids = batch.to(device)
            elif isinstance(batch, dict) and 'input_ids' in batch:
                input_ids = batch['input_ids'].to(device)
            else:
                # 兼容不同格式
                input_ids = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

            # 前向传播 (Causal LM loss)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        progress_bar.close()
        print(f"Finetuning completed. Average Loss: {running_loss / target_steps:.4f}")
        
        # 恢复为 eval 模式
        model.eval()
        # 恢复 use_cache
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
            
        torch.cuda.empty_cache() # 训练后清理
        return model

    def get_info(self):
        return {
            "type": "retraining",
            "search_space": {
                "learning_rate": [1e-4, 5e-5, 1e-5],
                "epochs": [0.2, 0.5, 1.0]
            }
        }
