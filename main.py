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
from utils.plotter import generate_performance_plot, generate_search_history_plot
from methods.quantization.fp16 import FP16Quantization
from methods.quantization.int8_sq import INT8SQQuantization
from methods.pruning.random import RandomPruning
from methods.pruning.l2 import L2StructuredPruning
from methods.retraining.finetuning import CausalLMFinetuning

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
    parser.add_argument("--data_samples", type=int, default=10, help="校准数据样本数量")
    parser.add_argument("--data_path", type=str, default=None, help="外部数据集路径（如 wikitext2 的 test.txt）")
    parser.add_argument("--save_to_local", action="store_true", help="是否保存压缩后的模型")
    
    # 新增: 控制是否在混合模式下启用再训练
    parser.add_argument("--retrain", type=lambda x: (str(x).lower() == 'true'), default=True, help="混合模式下是否启用再训练 (True/False), 默认 True")
    
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

def save_results(args, original_ppl, final_ppl, best_config, final_model, tokenizer, run_dir, picture_dir):
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

    search_history = best_config.get("search_history", [])

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "strategy": args.strategy,
        "data_samples": args.data_samples,
        "original_ppl": original_ppl,
        "final_ppl": final_ppl,
        "ppl_change": final_ppl - original_ppl,
        "best_config": best_config,
        "search_history": search_history # 新增：保存所有搜索记录
    }

    report_path = os.path.join(run_dir, "report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False, default=str)
        logger.info(f"Report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")

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
                # 移除 target_ratio 参数
                generate_search_history_plot(search_history, original_ppl, save_path=history_plot_path)
                logger.info(f"Search Space Visualization saved to: {history_plot_path}")
                
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}")
    else:
        logger.warning("Matplotlib not installed. Skipping visualization.")
        logger.warning("Tip: Run `pip install matplotlib` to enable charts.")

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
        "n_trials": args.n_trials,
        "enable_retrain": args.retrain # 传递 retrain 开关
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
