import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json

def main():
    parser = argparse.ArgumentParser(description="加载并测试指定的 Qwen 模型")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="要加载的模型 checkpoint 的路径",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你现在是一个智能助手，你需要用以下的格式回答问题：把你的思考过程放在\<think>\</think>中,最终的答案放在 \<answer>\</answer>中。解方程：$x^2-2x-3=0$",
        help="输入给模型的提示文本"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="生成文本的最大长度"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7, # 默认值，可以调整
        help="控制生成文本的随机性，较低的值更保守"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9, # 默认值，可以调整
        help="核心采样参数，控制选择词语的概率阈值"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50, # 默认值，可以调整
        help="核心采样参数，控制选择词语的范围大小"
    )
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction, # 使用 True/False 开关
        default=True,
        help="是否使用采样进行生成，--no-do_sample 表示不采样 (贪婪搜索)"
    )

    args = parser.parse_args()

    print(f"正在从 {args.model_path} 加载模型...")

    # 检查 config.json 是否存在
    config_path = os.path.join(args.model_path, "config.json")
    if not os.path.exists(config_path):
        print(f"错误: 在 '{args.model_path}' 目录下未找到 'config.json' 文件。")
        print("请确保模型路径正确，并且包含完整的模型文件。")
        # Attempt to list directory contents if it exists, otherwise state it doesn't exist
        if os.path.exists(args.model_path) and os.path.isdir(args.model_path):
            print(f"当前 '{args.model_path}' 目录下的文件有: {os.listdir(args.model_path)}")
        else:
            print(f"'{args.model_path}' 目录不存在。")
        return

    # 尝试读取 model_type (可选，用于调试)
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            model_type = config_data.get("model_type")
            if model_type:
                print(f"从 config.json 中检测到 model_type: {model_type}")
            else:
                print("警告: config.json 中未找到 'model_type' 字段。")
    except Exception as e:
        print(f"读取 config.json 时出错: {e}")


    # 检查是否有可用的 GPU
    if torch.cuda.is_available():
        device_map = "auto"
        print("检测到 CUDA，将使用 GPU 加载模型。")
    else:
        device_map = "cpu"
        print("未检测到 CUDA，将使用 CPU 加载模型。这可能会比较慢。")

    try:
        # 加载 tokenizer 和 model
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if device_map != "cpu" and torch.cuda.is_bf16_supported() else torch.float32,
            attn_implementation="eager"
        )
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("可能的原因：")
        print("1. 模型文件不完整或已损坏。")
        print("2. 'config.json' 中的 'model_type' 不被当前版本的 transformers 库支持或无法识别。")
        print(f"3. 显存不足 (如果使用 GPU)。尝试减小模型或使用 CPU (如果之前是 GPU)。")
        return

    messages = [
        {"role": "system", "content": "You are a helpful assistant that can solve math problems and show your reasoning."},
        {"role": "user", "content": args.prompt}
    ]

    print(f"\n输入提示: {args.prompt}")
    print(f"生成参数: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, do_sample={args.do_sample}, max_new_tokens={args.max_new_tokens}")

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        print("正在生成回复...")
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.do_sample else None, # temperature 只在 do_sample=True 时有效
            top_p=args.top_p if args.do_sample else None,             # top_p 只在 do_sample=True 时有效
            top_k=args.top_k if args.do_sample else None,             # top_k 只在 do_sample=True 时有效
            repetition_penalty=1.2,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id # 解决 pad_token_id 未设置的警告
        )
        
        input_token_len = model_inputs.input_ids.shape[1]
        response_tokens = generated_ids[0][input_token_len:]
        clean_response = tokenizer.decode(response_tokens, skip_special_tokens=False)

        print(f"\n模型回复:\n{clean_response}")

    except Exception as e:
        print(f"模型推理失败: {e}")

if __name__ == "__main__":
    main() 