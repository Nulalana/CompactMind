import sys
import os
import torch
import argparse
import json
import time
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 尝试导入 matplotlib，如果失败则 gracefully degrade
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 依然保留 HF_ENDPOINT，以防万一未来需要扩展
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

sys.path.append(os.getcwd())

from core.engine import SearchEngine
from core.compressor import Compressor
from core.evaluator import Evaluator
from utils.data_loader import get_calib_dataset
from methods.quantization.fp16 import FP16Quantization
from methods.quantization.int8_sq import INT8SQQuantization
from methods.pruning.random import RandomPruning
from methods.pruning.l2 import L2StructuredPruning

def parse_args():
    parser = argparse.ArgumentParser(description="AutoLLM-Compressor: 自动化大模型压缩工具")
    
    default_model_path = os.path.abspath("./models/Llama-2-7b-hf")
    
    parser.add_argument("--model_path", type=str, default=default_model_path, help="模型名称或本地路径")
    parser.add_argument("--strategy", type=str, default="grid", choices=["grid", "random"], help="搜索策略")
    parser.add_argument("--target_ratio", type=float, default=0.5, help="目标压缩比 (0.0-1.0)")
    parser.add_argument("--data_samples", type=int, default=10, help="校准数据样本数量")
    parser.add_argument("--data_path", type=str, default=None, help="外部数据集路径（如 wikitext2 的 test.txt）")
    
    # 修改: 显式支持 --cpu 和 --gpu，且默认使用 cpu (除非有 gpu 且没指定 cpu)
    # 为了实现“默认 CPU”但又允许“自动检测”，我们使用互斥组
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    group.add_argument("--gpu", action="store_true", help="强制使用 GPU (CUDA)")
    
    return parser.parse_args()

def get_device(args):
    """
    根据参数决定使用哪个设备
    """
    if args.gpu:
        if torch.cuda.is_available():
            return "cuda"
        else:
            print("⚠️ Warning: GPU requested but CUDA is not available. Falling back to CPU.")
            return "cpu"
    elif args.cpu:
        return "cpu"
    else:
        # 默认行为：修改为默认 CPU (根据用户需求)，或者保持自动检测
        # 用户需求：默认 CPU
        return "cpu"

def generate_performance_plot(original_ppl, final_ppl, best_config, save_path):
    """
    生成性能对比可视化图表
    """
    plt.figure(figsize=(10, 6))
    
    # 数据准备
    labels = ['Original', 'Compressed']
    values = [original_ppl, final_ppl]
    colors = ['#1f77b4', '#ff7f0e'] # 蓝橙配色
    
    # 绘制柱状图
    bars = plt.bar(labels, values, color=colors, width=0.5)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12)
    
    # 添加标题和标签
    method_name = best_config['method'] if best_config else "None"
    plt.title(f'Model Compression Performance\nMethod: {method_name}', fontsize=14)
    plt.ylabel('Perplexity (Lower is Better)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 计算变化率
    if original_ppl > 0:
        change_pct = ((final_ppl - original_ppl) / original_ppl) * 100
        plt.text(0.5, max(values) * 0.5, 
                f'PPL Change:\n{change_pct:+.2f}%', 
                ha='center', fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_results(args, original_ppl, final_ppl, best_config):
    # 1. 创建基于时间戳的独立运行目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir_name = f"run_{timestamp}"
    base_result_dir = "./results"
    run_dir = os.path.join(base_result_dir, run_dir_name)
    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        
    # 2. 保存 JSON 报告
    report_filename = "report.json"
    report_path = os.path.join(run_dir, report_filename)
    
    report = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "strategy": args.strategy,
        "target_ratio": args.target_ratio,
        "data_samples": args.data_samples,
        "original_ppl": original_ppl,
        "final_ppl": final_ppl,
        "ppl_change": final_ppl - original_ppl,
        "best_config": best_config
    }
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"\n✅ JSON Report saved to: {report_path}")
    except Exception as e:
        print(f"❌ Failed to save JSON report: {e}")

    # 3. 生成可视化图表
    if HAS_MATPLOTLIB:
        plot_path = os.path.join(run_dir, "performance_analysis.png")
        try:
            generate_performance_plot(original_ppl, final_ppl, best_config, plot_path)
            print(f"✅ Visualization saved to: {plot_path}")
        except Exception as e:
            print(f"❌ Failed to generate plot: {e}")
    else:
        print("\n⚠️ Matplotlib not installed. Skipping visualization.")
        print("Tip: Run `pip install matplotlib` to enable charts.")

def load_model(model_name_or_path, device):
    print(f"Loading model from: {model_name_or_path}")
    
    if not os.path.exists(model_name_or_path):
        print(f"\n❌ CRITICAL ERROR: Model path not found locally: {model_name_or_path}")
        print("Please download the model first (e.g., using scripts/download_model.py).")
        print("Exiting to prevent unintended network requests.")
        sys.exit(1)

    print(f"Detected local path: {model_name_or_path}")
    try:
        dtype = torch.float16 if "cuda" in device else torch.float32
        
        # 强制 local_files_only=True，严禁联网
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True 
        )
        print(f"Successfully loaded {model.__class__.__name__}")
        return model
    except Exception as e:
        print(f"\n❌ Failed to load local model: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    device = get_device(args)
    
    print(f"=== AutoLLM-Compressor Project Started ===")
    print(f"Arguments: {vars(args)}")
    print(f"Using Device: {device}")
    
    # 1. 加载模型 (严格本地模式)
    model = load_model(args.model_path, device)
    model.to(device)

    # 1.5 加载 Tokenizer (严格本地模式)
    print(f"Loading tokenizer from: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        print(f"\n❌ Failed to load local tokenizer from {args.model_path}")
        print(f"Error: {e}")
        sys.exit(1)

    # 2. 准备数据
    print("Preparing calibration data...")
    try:
        dataset = get_calib_dataset(
            data_name="wikitext2", 
            tokenizer_name=None, 
            n_samples=args.data_samples,
            tokenizer_obj=tokenizer,
            data_path=args.data_path
        )
    except FileNotFoundError as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error loading data: {e}")
        sys.exit(1)
    
    dataset = [d.to(device) for d in dataset]

    # 3. 初始化评估器
    evaluator = Evaluator(dataset, device=device)
    
    # 4. 评估原始模型
    print("\n--- Evaluating Original Model ---")
    try:
        original_ppl = evaluator.evaluate_perplexity(model)
        print(f"Original PPL: {original_ppl:.4f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        original_ppl = float('inf')
    
    # 5. 初始化搜索引擎
    engine = SearchEngine(search_strategy=args.strategy, evaluator=evaluator)
    
    # 6. 开始自动搜索
    print("\n--- Starting Automatic Search ---")
    
    # 变更: 传递 target_ratio 约束
    constraints = {"target_ratio": args.target_ratio}
    best_config = engine.search(model, constraints)
    
    print(f"\nBest Configuration Found: {best_config}")
    
    # 7. 使用最佳配置执行最终压缩
    print("\n--- Applying Best Compression ---")
    compressor = Compressor()
    final_model = compressor.run(model, best_config)
    
    # 8. 最终评估
    print("\n--- Evaluating Compressed Model ---")
    final_ppl = evaluator.evaluate_perplexity(final_model)
    
    print(f"\n=== Final Report ===")
    print(f"Original PPL: {original_ppl:.4f}")
    print(f"Final PPL:    {final_ppl:.4f}")
    print(f"Best Config:  {best_config}")

    # 9. 保存结果
    save_results(args, original_ppl, final_ppl, best_config)

if __name__ == "__main__":
    main()
