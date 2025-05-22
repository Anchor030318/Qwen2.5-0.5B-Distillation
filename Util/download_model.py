from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置项 ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
# 您可以将此路径修改为您希望存储模型的任何位置
LOCAL_MODEL_DIR = "./hf_models/Qwen2.5-0.5B" 
# --- -------- ---

def download_hf_model(model_name: str, local_dir: str):
    """
    下载 Hugging Face 模型和分词器到本地目录。
    """
    if not model_name or not local_dir:
        logging.error("模型名称或本地目录未指定。")
        return

    logging.info(f"本地模型存储路径设置为: {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    # 检查 HF_ENDPOINT 环境变量
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        logging.info(f"检测到 HF_ENDPOINT 环境变量: {hf_endpoint}")
        logging.info("下载将尝试通过此镜像进行。")
    else:
        logging.info("未检测到 HF_ENDPOINT 环境变量，将从 Hugging Face 官方源下载。")


    # 下载并保存 Tokenizer
    logging.info(f"开始下载 Tokenizer for '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(local_dir)
        logging.info(f"Tokenizer 已成功下载并保存到: {local_dir}")
    except Exception as e:
        logging.error(f"下载/保存 Tokenizer '{model_name}' 失败: {e}")
        logging.error("请检查网络连接、模型名称是否正确，或 HF_ENDPOINT (如果使用) 是否可达。")
        return # 如果tokenizer下载失败，通常模型也无法继续

    # 下载并保存 Model
    logging.info(f"开始下载 Model '{model_name}' (这可能需要一些时间)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model.save_pretrained(local_dir)
        logging.info(f"Model 已成功下载并保存到: {local_dir}")
    except Exception as e:
        logging.error(f"下载/保存 Model '{model_name}' 失败: {e}")
        logging.error("请检查网络连接、磁盘空间，或 HF_ENDPOINT (如果使用) 是否可达。")
        return
    
    logging.info(f"模型 '{model_name}' 的所有文件应已下载至 '{local_dir}'。")
    logging.info("请记得更新您的训练脚本，将模型路径指向此本地目录。")

if __name__ == "__main__":
    download_hf_model(MODEL_NAME, LOCAL_MODEL_DIR) 