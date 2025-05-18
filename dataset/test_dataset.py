import os
import sys
from dotenv import load_dotenv

# 尝试加载 .env 文件中的环境变量
try:
    load_dotenv()
except Exception as e:
    print(f"加载 .env 文件时出错: {e}，使用默认环境变量")

# 设置 Hugging Face 镜像地址和缓存目录
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_CACHE"] = "/gz-data/datasets"
os.environ["HF_HOME"] = "/gz-data/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/gz-data/huggingface/hub"

import torch
from transformers import AutoTokenizer
from dataset import SFTDataset

def decode_truncated(tokenizer, token_ids, max_length=100):
    """解码并截断文本，用于显示"""
    text = tokenizer.decode(token_ids)
    return text[:max_length] + "..." if len(text) > max_length else text

def main():
    # OpenMathReasoning 数据集路径
    data_path = "/gz-data/datasets/OpenMathReasoning/data/"
    
    # 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained("../model")
        print("使用本地模型分词器")
    except:
        # 如果本地模型不存在，使用 Qwen2.5-0.5B
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        print("使用 Qwen2.5-0.5B 分词器")
    
    # 加载单个数据文件进行测试
    test_file = os.path.join(data_path, os.listdir(data_path)[0])
    print(f"测试数据文件: {test_file}")
    
    # 创建数据集
    dataset = SFTDataset(test_file, tokenizer, max_length=1024)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 检查几个样本
    for i in range(min(3, len(dataset))):
        input_ids, target_ids, loss_mask = dataset[i]
        
        print(f"\n样本 {i+1}:")
        print(f"输入形状: {input_ids.shape}")
        print(f"目标形状: {target_ids.shape}")
        print(f"损失掩码形状: {loss_mask.shape}")
        
        # 解码输入以检查格式是否正确
        print("\n输入文本示例（截断）:")
        print(decode_truncated(tokenizer, input_ids, max_length=300))
        
        # 获取原始样本数据以显示字段
        raw_sample = dataset.data[i]
        print("\n原始样本字段:")
        print(f"问题: {raw_sample['problem'][:100]}..." if len(raw_sample['problem']) > 100 else raw_sample['problem'])
        print(f"解答过程: {raw_sample['generated_solution'][:100]}..." if len(raw_sample['generated_solution']) > 100 else raw_sample['generated_solution'])
        print(f"预期答案: {raw_sample['expected_answer']}")
        
        # 查找特殊标记在输入中的位置
        special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
        special_token_ids = []
        for token in special_tokens:
            special_token_ids.extend(tokenizer.encode(token, add_special_tokens=False))
        
        # 在 input_ids 中找到特殊标记并检查相应的 loss_mask 值
        special_token_positions = []
        for j, token_id in enumerate(input_ids):
            if token_id.item() in special_token_ids:
                special_token_positions.append(j)
        
        if special_token_positions:
            print("\n特殊标记位置及其损失权重:")
            for pos in special_token_positions:
                token = tokenizer.decode([input_ids[pos]])
                print(f"位置 {pos}: 标记 '{token}', 损失权重: {loss_mask[pos].item()}")
    
    # 测试批处理功能
    if len(dataset) >= 2:
        print("\n测试批处理准备:")
        batch = [dataset[0], dataset[1]]
        batch_dict = dataset.prepare_batch(batch)
        
        print(f"批次输入形状: {batch_dict['input_ids'].shape}")
        print(f"批次标签形状: {batch_dict['labels'].shape}")
        print(f"批次损失掩码形状: {batch_dict['loss_mask'].shape}")
        
    print("\n数据集加载和处理测试完成。")

if __name__ == "__main__":
    main() 