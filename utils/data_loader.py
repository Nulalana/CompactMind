import os
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

def get_calib_dataset(data_name="wikitext2", tokenizer_name="gpt2", n_samples=128, seq_len=2048, tokenizer_obj=None, data_path=None):
    """
    加载用于校准或评估的数据集。优先读取本地 data/wikitext2/test.txt。
    """
    print(f"Loading calibration dataset: {data_name}...")
    
    # 1. 准备 Tokenizer
    if tokenizer_obj is not None:
        tokenizer = tokenizer_obj
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        except:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    dataset = []
    
    if data_name == "wikitext2":
        local_path = data_path if data_path is not None else "./data/wikitext2/test.txt"
        
        if os.path.exists(local_path):
            print(f"Found local dataset at: {local_path}")
            with open(local_path, 'r', encoding='utf-8') as f:
                text_data = f.read()
        else:
            raise FileNotFoundError(
                f"Dataset not found at {local_path}.\n"
                f"Please run 'python scripts/download_data.py' to download it first."
            )

        # 3. 处理数据
        encodings = tokenizer(text_data, return_tensors='pt')
        
        for i in range(n_samples):
            begin_loc = i * seq_len
            end_loc = begin_loc + seq_len
            
            if end_loc > encodings.input_ids.size(1):
                break
                
            inp = encodings.input_ids[:, begin_loc:end_loc]
            dataset.append(inp)
            
    else:
        raise NotImplementedError(f"Dataset {data_name} not supported yet.")
        
    print(f"Loaded {len(dataset)} samples.")
    return dataset
