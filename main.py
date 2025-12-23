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
from utils.plotter import generate_performance_plot, generate_search_history_plot, generate_interactive_search_history_plot
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

                # 新增：生成交互式 HTML 图表
                interactive_plot_path = os.path.join(picture_dir, "search_space_analysis.html")
                generate_interactive_search_history_plot(search_history, original_ppl, save_path=interactive_plot_path)

                
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
        # 显存优化：避免初始加载占用过多
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True 
        )
        # 不要在这里 to(device)，让调用者决定何时移动，避免双倍显存占用
        # print(f"Successfully loaded {model.__class__.__name__}")
        return model
    except Exception as e:
        print(f"\n❌ Failed to load local model: {e}")
        sys.exit(1)

import multiprocessing

def run_worker(rank, world_size, args, run_dir, picture_dir, storage_url, study_name):
    """
    工作进程函数：独立加载模型与数据，执行搜索任务
    """
    # 设置当前进程可见的 GPU
    # 如果有多张卡，rank 对应 GPU ID
    if args.gpu and torch.cuda.device_count() > 1:
        # 子进程不需要重新设置 CUDA_VISIBLE_DEVICES，因为在 Process 启动前还没初始化 CUDA
        # 但是在 spawn 模式下，子进程是全新的，所以需要确保它只看到指定的 GPU
        # 注意：在 spawn 模式下，os.environ 的修改会传递给子进程，但如果在主进程改了，可能会影响其他。
        # 最好的方式是在子进程一开始就设置。
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        # 强制 device 为 cuda:0 (因为对子进程来说，它只有这一张卡)
        # 修正：当 CUDA_VISIBLE_DEVICES=rank 时，Python 看到的设备 ID 是 0
        device = "cuda:0" 
    elif args.gpu:
         # 单卡多进程情况（不推荐，但为了兼容性）
         # 或者在 args.gpu 且 device_count==1 时，也应该允许运行
         device = get_device(args)
    else:
        # 单卡或 CPU 模式
        device = get_device(args)
    
    # 初始化日志（每个进程需要独立的 logger 或者是向同一个文件写？这里简单起见，让主进程负责主日志，子进程只输出到控制台或共享文件）
    # 由于多进程写同一个文件可能会冲突，这里我们依赖 setup_logging 在主进程做好的配置（如果是 fork），
    # 但 Windows 是 spawn，所以需要重新配置。为了避免混乱，子进程日志加前缀。
    # 简单起见，重新调用 setup_logging，但可能要注意文件锁。
    # 我们暂时让子进程只输出到 stdout
    
    worker_prefix = f"[Worker-{rank}] "
    print(f"{worker_prefix}Starting process on device {device}")

    # 1. 加载模型
    model = load_model(args.model_path, device)
    model.to(device)
    
    # 2. 加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        print(f"{worker_prefix}Failed to load tokenizer: {e}")
        return

    # 3. 准备数据
    try:
        dataset = get_calib_dataset(
            data_name="wikitext2", 
            tokenizer_name=None, 
            n_samples=args.data_samples,
            tokenizer_obj=tokenizer,
            data_path=args.data_path
        )
        dataset = [d.to(device) for d in dataset]
    except Exception as e:
        print(f"{worker_prefix}Failed to load data: {e}")
        return

    # 4. 初始化评估器
    evaluator = Evaluator(dataset, device=device)
    
    # 5. 初始化搜索引擎
    engine = SearchEngine(search_strategy=args.strategy, evaluator=evaluator)
    
    # 6. 开始搜索 (连接到同一个 Study)
    # 计算此 Worker 分配到的 trials 数 (如果需要平均分配，或者让 Optuna 抢占式分配)
    # Optuna 的 storage 模式支持抢占式，所有 worker 共同完成总 n_trials
    # 但为了简单，我们可以让每个 worker 跑 n_trials / world_size，或者直接设定总数
    # Optuna 的 optimize 是“跑 n_trials 次”，如果是分布式，意味着“这个进程跑 n_trials 次”。
    # 我们希望总共跑 n_trials 次。
    # 正确的做法是：不指定 n_trials 给 optimize，或者动态检查。
    # 但 Optuna 的 API optimize(n_trials=N) 是指“这个 worker 执行 N 次”。
    # 如果我们要总共 N 次，最简单的办法是平均分。
    
    my_trials = args.n_trials // world_size
    if rank < args.n_trials % world_size:
        my_trials += 1
        
    print(f"{worker_prefix}Will execute {my_trials} trials...")
    
    constraints = {
        "n_trials": my_trials,
        "enable_retrain": args.retrain,
        "study_name": study_name,
        "storage": storage_url
    }
    
    try:
        # 执行搜索
        # 注意：这里我们不需要返回值，因为主进程会从 storage 读取最佳结果
        # 但为了复用代码，search 会返回 best_config
        engine.search(model, constraints)
        print(f"{worker_prefix}Finished.")
    except Exception as e:
        print(f"{worker_prefix}Error during search: {e}")
        import traceback
        traceback.print_exc()

def main():
    args = parse_args()
    
    # 0. 提前创建目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir_name = f"result_{timestamp}"
    base_result_dir = "./results"
    run_dir = os.path.join(base_result_dir, run_dir_name)
    picture_dir = os.path.join(run_dir, "picture")
    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if not os.path.exists(picture_dir):
        os.makedirs(picture_dir)
        
    setup_logging(run_dir)
    
    # 检测 GPU 数量
    gpu_count = torch.cuda.device_count()
    use_parallel = args.gpu and gpu_count > 1 and args.strategy == "bayesian"
    
    if use_parallel:
        logger.info(f"🚀 Detected {gpu_count} GPUs. Enabling Parallel Bayesian Search!")
        
        # 释放主进程加载的模型以节省显存，留给子进程使用
        torch.cuda.empty_cache()
        logger.info("Cleared main process model to free up GPU memory for workers.")
        
        # 准备 Optuna Storage (SQLite)
        db_path = os.path.join(run_dir, "optuna.db")
        storage_url = f"sqlite:///{db_path}"
        study_name = f"study_{timestamp}"
        
        logger.info(f"Optuna Storage: {storage_url}")
        
        # 必须设置 spawn 启动方式，否则 CUDA 初始化会报错
        # 注意：set_start_method 只能调用一次，这里加 try-except
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass # 已经设置过也没关系
        
        # 启动多进程 Workers
        processes = []
        for rank in range(gpu_count):
            p = multiprocessing.Process(
                target=run_worker,
                args=(rank, gpu_count, args, run_dir, picture_dir, storage_url, study_name)
            )
            p.start()
            processes.append(p)
            
        # 等待所有 Worker 完成
        for p in processes:
            p.join()
            
        logger.info("All workers finished. Aggregating results...")
        
        # 主进程加载最佳结果并进行最终评估
        # 需要重新加载模型（在主进程设备上，通常是 gpu:0）
        device = "cuda:0"
        model = load_model(args.model_path, device)
        model.to(device)
        
        # 加载数据和评估器用于最终验证
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        dataset = get_calib_dataset(
            data_name="wikitext2", n_samples=args.data_samples, 
            tokenizer_obj=tokenizer, data_path=args.data_path
        )
        dataset = [d.to(device) for d in dataset]
        evaluator = Evaluator(dataset, device=device)
        
        # 从 Storage 中读取最佳 Study
        import optuna
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            logger.info(f"Best params found in study: {study.best_params}")
            logger.info(f"Best value (PPL): {study.best_value}")
        except KeyError:
            logger.error("Study not found in storage. It seems all workers failed.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load study: {e}")
            sys.exit(1)
        
        # 重构最佳配置
        # 注意：Optuna 存储的是扁平的 params，我们需要将其转换回 config 字典
        # 这比较麻烦，因为 engine._search_bayesian 里的 objective 函数做了转换逻辑
        # 更好的办法是：让 search 方法返回 best_config，但这在多进程下拿不到。
        # 替代方案：Worker 已经把 best_config 找到了，但没法传回。
        # 我们需要重新解析 best_params。
        # 或者，我们在 Engine 里把 best_config 存到 UserAttrs？
        # 鉴于 engine.py 已经有了复杂的转换逻辑，我们在主进程里只能手动复现那个转换，或者...
        # 简单方案：直接用 best_params 里的信息构造 config。
        # 由于参数展平了，这有点复杂。
        # 让我们修改 Engine，把 best_config 序列化存到 Study 的 user_attrs 里？
        # 但 Optuna 的 user_attrs 是 trial 级别的。
        # Study 级别的 user_attrs 可以用 study.set_user_attr()。
        # 但多个 worker 同时跑，谁来 set best?
        # 其实我们只需要 trial.user_attrs["config"] = config。
        # 然后 study.best_trial.user_attrs["config"] 就是我们要的。
        
        # 我们需要修改 engine.py，在 objective 里把 config 存入 trial.user_attrs
        
        best_trial = study.best_trial
        if "config" in best_trial.user_attrs:
             best_config = best_trial.user_attrs["config"]
        else:
             # 如果 engine 没改，只能 fallback (或者现在去改 engine)
             logger.warning("Could not retrieve full config from trial.user_attrs. Parallel search requires engine update.")
             # 这里我们先假设 engine 会改，或者在这里直接用 best_params 猜
             best_config = {} # TODO: Fix this by updating engine.py
        
    else:
        # 单进程模式 (原有逻辑)
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
        
        constraints = {
            "n_trials": args.n_trials,
            "enable_retrain": args.retrain
        }
        best_config = engine.search(model, constraints)

    # === 公共结束部分 (应用最佳配置并保存) ===
    # 注意：并行模式下，model, tokenizer, original_ppl, best_config 都需要在 if/else 块中准备好
    # 在并行模式的 if 块里，我们需要补全 original_ppl 和 best_config 的获取
    
    if use_parallel:
        # 并行模式下补全 original_ppl
        logger.info("--- Evaluating Original Model (Final Check) ---")
        original_ppl = evaluator.evaluate_perplexity(model)
        
        # 补全 best_config (依赖 engine 更新)
        # 如果 engine 没存 user_attrs，这里会出错。所以必须更新 engine.py
        pass 

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

    # 11. 保存结果
    save_results(args, original_ppl, final_ppl, best_config, final_model, tokenizer, run_dir, picture_dir)

if __name__ == "__main__":
    main()
