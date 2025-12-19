import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def download_model(model_name, save_dir):
    print(f"=== Starting Download for {model_name} ===")
    print(f"Target Directory: {save_dir}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("Downloading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(save_dir)
        print("Tokenizer saved.")
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        return

    print("Downloading Model (this may take a while)...")
    try:
        # 使用 float16 下载以节省带宽和存储（如果模型支持）
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nNOTE: Llama-2 is a gated model. You must be logged in to Hugging Face.")
        print("Try running `huggingface-cli login` in your terminal first.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Hugging Face Model ID")
    parser.add_argument("--save_dir", type=str, default="./models/Llama-2-7b-hf", help="Local directory to save model")
    args = parser.parse_args()
    
    download_model(args.model, args.save_dir)
