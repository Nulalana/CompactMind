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

import logging

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

# 配置全局日志
logger = logging.getLogger(__name__)

def setup_logging(run_dir):
    """
    配置日志系统：同时输出到控制台和文件
    """
    log_path = os.path.join(run_dir, "run.log")
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="AutoLLM-Compressor: 自动化大模型压缩工具")
    
    default_model_path = os.path.abspath("./models/Llama-2-7b-hf")
    
    parser.add_argument("--model_path", type=str, default=default_model_path, help="模型名称或本地路径")
    parser.add_argument("--strategy", type=str, default="bayesian", choices=["grid", "random", "bayesian"], help="搜索策略 (默认: bayesian)")
    parser.add_argument("--n_trials", type=int, default=30, help="贝叶斯搜索的尝试次数 (默认: 30)")
    parser.add_argument("--target_ratio", type=float, default=0.5, help="目标压缩比 (0.0-1.0)")
    parser.add_argument("--data_samples", type=int, default=10, help="校准数据样本数量")
    parser.add_argument("--data_path", type=str, default=None, help="外部数据集路径（如 wikitext2 的 test.txt）")
    parser.add_argument("--save_to_local", action="store_true", help="是否保存压缩后的模型")
    
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
    # 修复 KeyError: 'method' - best_config 可能包含 'pipeline' 而不是 'method'
    if best_config:
        if 'method' in best_config:
            method_name = best_config['method']
        elif 'pipeline' in best_config:
            # 简化显示 pipeline 名称
            methods = [step['method'] for step in best_config['pipeline']]
            method_name = " + ".join(methods)
        else:
            method_name = "Unknown"
    else:
        method_name = "None"

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

def save_results(args, original_ppl, final_ppl, best_config, final_model=None, tokenizer=None, run_dir=None, picture_dir=None):
    # 如果未传入目录（兼容旧调用），则在此创建
    if run_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir_name = f"result_{timestamp}"
        base_result_dir = "./results"
        run_dir = os.path.join(base_result_dir, run_dir_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            
    if picture_dir is None:
        picture_dir = os.path.join(run_dir, "picture")
        if not os.path.exists(picture_dir):
            os.makedirs(picture_dir)
            
    # 2. 保存 JSON 报告
    report_filename = "report.json"
    report_path = os.path.join(run_dir, report_filename)
    
    # 提取搜索历史（如果有）
    search_history = best_config.pop("search_history", []) if best_config else []
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "strategy": args.strategy,
        "target_ratio": args.target_ratio,
        "data_samples": args.data_samples,
        "original_ppl": original_ppl,
        "final_ppl": final_ppl,
        "ppl_change": final_ppl - original_ppl,
        "best_config": best_config,
        "search_history": search_history # 新增：保存所有搜索记录
    }
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        logger.info(f"JSON Report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON report: {e}")

    # 3. 生成搜索历史 CSV 报告 (可选，方便分析)
    if search_history:
        try:
            csv_path = os.path.join(run_dir, "search_history.csv")
            with open(csv_path, 'w', encoding='utf-8') as f:
                # 写入表头
                f.write("Type,Config,Ratio,PPL\n")
                for item in search_history:
                    # 将 config 转换为字符串以适应 CSV
                    config_str = json.dumps(item["config"], ensure_ascii=False).replace('"', '""')
                    line = f"{item['type']},\"{config_str}\",{item['ratio']:.4f},{item['ppl']:.4f}\n"
                    f.write(line)
            logger.info(f"Search History CSV saved to: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV history: {e}")

    # 4. 生成可视化图表
    if HAS_MATPLOTLIB:
        # 修改: 保存到 picture 目录
        plot_path = os.path.join(picture_dir, "performance_analysis.png")
        try:
            generate_performance_plot(original_ppl, final_ppl, best_config, plot_path)
            logger.info(f"Visualization saved to: {plot_path}")
            
            # 额外：生成搜索历史散点图 (Pareto Frontier)
            if search_history:
                history_plot_path = os.path.join(picture_dir, "search_space_analysis.png")
                generate_search_history_plot(search_history, original_ppl, target_ratio=args.target_ratio, save_path=history_plot_path)
                logger.info(f"Search Space Visualization saved to: {history_plot_path}")
                
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}")
    else:
        logger.warning("Matplotlib not installed. Skipping visualization.")
        logger.warning("Tip: Run `pip install matplotlib` to enable charts.")
    
    # 5. 保存模型 (如果用户指定)
    if args.save_to_local and final_model and tokenizer:
        model_save_dir = os.path.join(run_dir, "model")
        logger.info(f"Saving compressed model to: {model_save_dir}...")
        try:
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            
            final_model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)
            logger.info(f"Compressed model saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save compressed model: {e}")

def generate_search_history_plot(history, original_ppl, target_ratio, save_path):
    """
    生成搜索空间分析图：压缩比 vs PPL，含 Pareto Frontier
    """
    plt.figure(figsize=(12, 8))
    
    # 过滤掉 PPL 为 inf 的点
    valid_history = [h for h in history if h['ppl'] != float('inf')]
    if not valid_history:
        print("No valid history to plot.")
        return

    ratios = [h['ratio'] for h in valid_history]
    ppls = [h['ppl'] for h in valid_history]
    types = [h['type'] for h in valid_history]
    
    # 1. 绘制散点
    single_indices = [i for i, t in enumerate(types) if t == 'single']
    pipeline_indices = [i for i, t in enumerate(types) if t == 'hybrid' or t == 'pipeline']
    
    if single_indices:
        plt.scatter([ratios[i] for i in single_indices], [ppls[i] for i in single_indices], 
                   c='blue', label='Single Method', alpha=0.6, s=80, marker='o', edgecolors='k')
        
    if pipeline_indices:
        plt.scatter([ratios[i] for i in pipeline_indices], [ppls[i] for i in pipeline_indices], 
                   c='red', label='Hybrid Method', alpha=0.6, s=100, marker='^', edgecolors='k')
    
    # 2. 计算并绘制 Pareto Frontier
    # 按 ratio 排序
    sorted_points = sorted(valid_history, key=lambda x: x['ratio'])
    pareto_points = []
    current_min_ppl = float('inf')
    
    for point in sorted_points:
        if point['ppl'] < current_min_ppl:
            pareto_points.append(point)
            current_min_ppl = point['ppl']
    
    if pareto_points:
        p_ratios = [p['ratio'] for p in pareto_points]
        p_ppls = [p['ppl'] for p in pareto_points]
        plt.plot(p_ratios, p_ppls, 'g--', linewidth=2, label='Pareto Frontier', alpha=0.8)
        # 高亮 Pareto 点
        plt.scatter(p_ratios, p_ppls, c='gold', s=150, marker='*', zorder=10, label='Pareto Optimal')

    # 绘制原始基准线
    plt.axhline(y=original_ppl, color='green', linestyle=':', label='Original PPL')
    
    # 绘制目标压缩比线
    plt.axvline(x=target_ratio, color='gray', linestyle='--', label='Target Ratio Limit')
    
    plt.xlabel('Estimated Compression Ratio (Lower is More Compressed)')
    plt.ylabel('Perplexity (Lower is Better)')
    plt.title('Search Space Analysis: Compression Ratio vs PPL (Pareto Frontier)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 标注最佳点
    if ppls:
        best_idx = ppls.index(min(ppls))
        best_x = ratios[best_idx]
        best_y = ppls[best_idx]
        plt.annotate('Best Found', xy=(best_x, best_y), xytext=(best_x, best_y*1.15),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

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
    
    # 0. 提前创建目录以供日志使用
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir_name = f"result_{timestamp}"
    base_result_dir = "./results"
    run_dir = os.path.join(base_result_dir, run_dir_name)
    picture_dir = os.path.join(run_dir, "picture")
    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if not os.path.exists(picture_dir):
        os.makedirs(picture_dir)
        
    # 1. 初始化日志
    setup_logging(run_dir)
    
    device = get_device(args)
    
    logger.info(f"=== AutoLLM-Compressor Project Started ===")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Using Device: {device}")
    
    # 2. 加载模型 (严格本地模式)
    model = load_model(args.model_path, device)
    model.to(device)

    # 3. 加载 Tokenizer (严格本地模式)
    logger.info(f"Loading tokenizer from: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        logger.error(f"Failed to load local tokenizer from {args.model_path}")
        logger.error(f"Error: {e}")
        sys.exit(1)

    # 4. 准备数据
    logger.info("Preparing calibration data...")
    try:
        dataset = get_calib_dataset(
            data_name="wikitext2", 
            tokenizer_name=None, 
            n_samples=args.data_samples,
            tokenizer_obj=tokenizer,
            data_path=args.data_path
        )
    except FileNotFoundError as e:
        logger.critical(f"{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        sys.exit(1)
    
    dataset = [d.to(device) for d in dataset]

    # 5. 初始化评估器
    evaluator = Evaluator(dataset, device=device)
    
    # 6. 评估原始模型
    logger.info("--- Evaluating Original Model ---")
    try:
        original_ppl = evaluator.evaluate_perplexity(model)
        logger.info(f"Original PPL: {original_ppl:.4f}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        original_ppl = float('inf')
    
    # 7. 初始化搜索引擎
    engine = SearchEngine(search_strategy=args.strategy, evaluator=evaluator)
    
    # 8. 开始自动搜索
    logger.info("--- Starting Automatic Search ---")
    
    # 变更: 传递 target_ratio 约束
    constraints = {
        "target_ratio": args.target_ratio,
        "n_trials": args.n_trials
    }
    best_config = engine.search(model, constraints)
    
    logger.info(f"Best Configuration Found: {best_config}")
    
    # 9. 使用最佳配置执行最终压缩
    logger.info("--- Applying Best Compression ---")
    compressor = Compressor()
    final_model = compressor.run(model, best_config)
    
    # 10. 最终评估
    logger.info("--- Evaluating Compressed Model ---")
    final_ppl = evaluator.evaluate_perplexity(final_model)
    
    logger.info("=== Final Report ===")
    logger.info(f"Original PPL: {original_ppl:.4f}")
    logger.info(f"Final PPL:    {final_ppl:.4f}")
    logger.info(f"Best Config:  {best_config}")

    # 11. 保存结果 (传入已创建的目录路径，避免重复创建逻辑)
    # 重构 save_results 以接受 run_dir 和 picture_dir
    save_results(args, original_ppl, final_ppl, best_config, final_model, tokenizer, run_dir, picture_dir)

if __name__ == "__main__":
    main()
