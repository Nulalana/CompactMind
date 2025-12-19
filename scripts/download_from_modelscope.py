import os
import argparse
from modelscope import snapshot_download

def download_from_modelscope(model_id, local_dir):
    print(f"=== Starting Download from ModelScope ===")
    print(f"Model ID: {model_id}")
    print(f"Target Directory: {local_dir}")
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        
    try:
        # cache_dir 指定为当前目录下的 models 文件夹
        # 这样模型会被下载到 ./models/shakechen/Llama-2-7b-hf 类似的结构中
        model_dir = snapshot_download(
            model_id, 
            cache_dir='./models', 
            revision='master'
        )
        print(f"\n✅ Download Success!")
        print(f"Model is saved at: {model_dir}")
        print(f"\nYou can run the project using:")
        print(f"python main.py --model \"{model_dir}\" --strategy grid")
        
    except Exception as e:
        print(f"❌ Download Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认使用你图片中的模型 ID
    parser.add_argument("--model", type=str, default="shakechen/Llama-2-7b-hf", help="ModelScope Model ID")
    parser.add_argument("--save_dir", type=str, default="./models", help="Local directory to save model")
    args = parser.parse_args()
    
    download_from_modelscope(args.model, args.save_dir)
