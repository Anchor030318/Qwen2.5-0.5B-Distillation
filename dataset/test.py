from dotenv import load_dotenv
load_dotenv()  # 自动加载.env文件中的环境变量


from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("parquet", data_files="./datasets/OpenMathReasoning/data/*")['train']
print(f"加载的数据集类型: {type(ds)}")
print(f"数据集大小: {len(ds)} 条数据")
print(f"数据集字段: {ds.features}")
# 显示第一条数据示例
print("\n数据示例:")
print(ds[0])