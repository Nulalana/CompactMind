import os
from datasets import load_dataset

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_and_save_wikitext2(save_dir="./data/wikitext2"):
    print(f"=== Downloading WikiText-2 via HuggingFace Mirror ===")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    try:
        # 下载数据集
        print("Downloading...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        
        # 保存为纯文本
        save_path = os.path.join(save_dir, "test.txt")
        print(f"Saving to {save_path}...")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            for text in dataset['text']:
                f.write(text)
                
        print(f"✅ Success! Data saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    download_and_save_wikitext2()
