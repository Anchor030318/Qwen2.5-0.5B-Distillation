from dotenv import load_dotenv
load_dotenv()  # 自动加载.env文件中的环境变量

import os
# 设置Hugging Face镜像地址
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_CACHE"] = "/gz-data/datasets"
os.environ["HF_HOME"] = "/gz-data/huggingface"  # 主缓存目录
os.environ["HUGGINGFACE_HUB_CACHE"] = "/gz-data/huggingface/hub"  # Hub 缓存

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("parquet", data_files="/gz-data/datasets/OpenMathReasoning/data/*")['train']
print(f"加载的数据集类型: {type(ds)}")
print(f"数据集大小: {len(ds)} 条数据")
print(f"数据集字段: {ds.features}")
# 显示第一条数据示例
print("\n数据示例:")
print(ds[0])