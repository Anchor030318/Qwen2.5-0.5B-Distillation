import polars as pl
import os
from tqdm import tqdm
import glob

# --- 配置参数 ---
# 输入Parquet文件所在的目录
INPUT_DIR = "training_data/data"  # 例如: "data/raw_data"
# 输出过滤后Parquet文件保存的目录
OUTPUT_DIR = "training_data/filtered_data" # 例如: "data/filtered_data"
# problem 和 generated_solution 合并后的最大token长度
MAX_SEQ_LEN = 2048
# pass_rate_72b_tir 的阈值 (0到1之间)，保留通过率 > PASS_RATE_THRESHOLD 的题目
PASS_RATE_THRESHOLD = 0.8 # 例如，保留通过率大于10%的题目
# Hugging Face Hub 上的分词器模型名称 (可选，用于更准确的长度计算)
# 例如: "meta-llama/Llama-2-7b-hf", "Qwen/Qwen1.5-7B-Chat"
# 如果留空或 transformers 未安装，将使用基于空格的简单分词
TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"

# --- 初始化分词器 (如果配置了) ---
tokenizer = None
if TOKENIZER_NAME:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print(f"成功加载分词器: {TOKENIZER_NAME}")
    except ImportError:
        print("警告: 未安装 'transformers' 库。将使用基于空格的简单分词进行长度计算。")
        print("为了更准确的长度过滤，请安装: pip install transformers")
    except Exception as e:
        print(f"警告: 加载分词器 {TOKENIZER_NAME} 失败: {e}。将使用基于空格的简单分词。")

if not tokenizer:
    print("使用基于空格的简单分词进行长度计算。")

def count_tokens(text):
    """使用配置的分词器计算文本的token数量，如果分词器不可用则按空格分割计数。"""
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    # 确保文本是字符串类型，以防传入 None 或其他类型
    return len(str(text).split())

def process_parquet_file(input_file_path, output_file_path):
    """
    处理单个Parquet文件：加载、过滤并保存。
    """
    try:
        print(f"开始处理文件: {input_file_path}")
        # 使用 polars 读取 Parquet 文件
        df = pl.read_parquet(input_file_path)
        print(f"原始数据行数: {len(df)}")

        # 1. 过滤无法提取答案的数据 (problem_type != "no_answer_extracted")
        # 使用 polars 的 filter 方法
        df_filtered = df.filter(pl.col("problem_type") != "no_answer_extracted")
        print(f"过滤 'no_answer_extracted' 后行数: {len(df_filtered)}")

        # 2. 过滤太难的题目 (根据 pass_rate_72b_tir)
        if "pass_rate_72b_tir" in df_filtered.columns:
            # 使用 polars 的表达式进行转换和过滤
            df_filtered = df_filtered.with_columns(
                pl.col("pass_rate_72b_tir")
                .cast(pl.Float64, strict=False) # strict=False 会将 "n/a" 等无法转换的值变为 null
                .alias("pass_rate_72b_tir_numeric")
            )
            # 过滤掉 null 值以及不满足阈条件的值
            df_filtered = df_filtered.filter(
                (pl.col("pass_rate_72b_tir_numeric").is_not_null()) &
                (pl.col("pass_rate_72b_tir_numeric") > PASS_RATE_THRESHOLD)
            )
            df_filtered = df_filtered.drop("pass_rate_72b_tir_numeric") # 移除辅助列
            print(f"根据 pass_rate_72b_tir > {PASS_RATE_THRESHOLD} 过滤后行数: {len(df_filtered)}")
        else:
            print("警告: 数据中未找到 'pass_rate_72b_tir' 列，跳过此过滤步骤。")

        # 3. 根据数据长度进行过滤
        # 确保 'problem' 和 'generated_solution' 列存在
        if "problem" not in df_filtered.columns or "generated_solution" not in df_filtered.columns:
            print("警告: 'problem' 或 'generated_solution' 列不存在，无法进行长度过滤。")
            if df_filtered.height > 0: # 使用 df.height > 0 判断 polars DataFrame 是否为空
                df_filtered.write_parquet(output_file_path) # 使用 polars 直接写入 Parquet
                print(f"过滤后的数据已保存到: {output_file_path}")
            else:
                print("没有数据满足过滤条件，不保存文件。")
            return

        # 计算合并长度并过滤 (使用 polars 的 map_elements)
        # Polars 的操作通常是不可变的，不需要显式 .copy()
        df_filtered = df_filtered.with_columns(
            (
                pl.col("problem").fill_null("").map_elements(lambda text_val: count_tokens(text_val), return_dtype=pl.Int64, strategy="threading") +
                pl.col("generated_solution").fill_null("").map_elements(lambda text_val: count_tokens(text_val), return_dtype=pl.Int64, strategy="threading")
            ).alias("combined_length")
        )
        df_filtered = df_filtered.filter(pl.col("combined_length") <= MAX_SEQ_LEN)
        df_filtered = df_filtered.drop("combined_length") # 移除辅助列
        print(f"根据 MAX_SEQ_LEN ({MAX_SEQ_LEN} tokens) 过滤后行数: {len(df_filtered)}")


        if df_filtered.height > 0: # 使用 df.height > 0 判断 polars DataFrame 是否为空
            # 使用 polars 直接写入 Parquet
            df_filtered.write_parquet(output_file_path)
            print(f"过滤后的数据已保存到: {output_file_path}")
        else:
            print("没有数据满足所有过滤条件，不保存文件。")

    except Exception as e:
        print(f"处理文件 {input_file_path} 时发生错误: {e}")

def main():
    """
    主函数，遍历输入目录中的所有Parquet文件并进行处理。
    """
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录 '{INPUT_DIR}' 不存在。请检查路径。")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 查找所有 .parquet 文件
    parquet_files = glob.glob(os.path.join(INPUT_DIR, "*.parquet"))
    if not parquet_files:
        parquet_files = glob.glob(os.path.join(INPUT_DIR, "**/*.parquet"), recursive=True)


    if not parquet_files:
        print(f"在目录 '{INPUT_DIR}' 中没有找到 .parquet 文件。")
        return

    print(f"找到 {len(parquet_files)} 个 Parquet 文件待处理。")

    for file_path in tqdm(parquet_files, desc="处理Parquet文件"):
        base_name = os.path.basename(file_path)
        output_file = os.path.join(OUTPUT_DIR, base_name.replace(".parquet", "_filtered.parquet"))
        process_parquet_file(file_path, output_file)

    print("所有文件处理完成。")

if __name__ == "__main__":
    # 验证配置
    if INPUT_DIR == "your_input_parquet_directory" or OUTPUT_DIR == "your_output_parquet_directory":
        print("错误: 请在脚本中配置 INPUT_DIR 和 OUTPUT_DIR。")
    elif MAX_SEQ_LEN <= 0:
        print("错误: MAX_SEQ_LEN 必须是正数。")
    elif not (0 <= PASS_RATE_THRESHOLD <= 1):
        print("错误: PASS_RATE_THRESHOLD 必须在 0 和 1 之间。")
    else:
        main() 