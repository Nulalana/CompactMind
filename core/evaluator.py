import torch
import torch.nn as nn
from tqdm import tqdm

class Evaluator:
    """
    模型评估器：计算困惑度 (Perplexity) 等指标。
    """
    def __init__(self, dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.dataset = dataset
        self.device = device

    @torch.no_grad()
    def evaluate_perplexity(self, model: nn.Module) -> float:
        """
        计算模型在给定数据集上的 Perplexity (PPL)。
        """
        print("Starting Perplexity evaluation...")
        model.eval()
        model.to(self.device)
        
        nlls = []
        
        for batch in tqdm(self.dataset, desc="Evaluating"):
            batch = batch.to(self.device)
            
            # 简单的 PPL 计算逻辑
            # 注意：实际大模型计算 PPL 可能需要 sliding window 策略，这里使用简化的 block 策略
            output = model(batch, labels=batch)
            neg_log_likelihood = output.loss
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).mean())
        print(f"Evaluation result - PPL: {ppl.item():.4f}")
        return ppl.item()
