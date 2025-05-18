#!/bin/bash

# --- Load .env file if it exists ---
if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  set -a # automatically export all variables subsequently defined or modified
  source .env
  set +a # stop automatically exporting
else
  echo ".env file not found, skipping."
fi
# --- End of .env loading ---

# --- Model Download Configuration & Logic ---
HUGGING_FACE_MODEL_ID="Qwen/Qwen2.5-0.5B"
HF_DOWNLOAD_SCRIPT_PATH="Util/hfd.sh"
# Base directory for local Hugging Face models. 
# The hfd.sh script will download into a subdirectory here named after the model.
LOCAL_MODEL_STORAGE_BASE_DIR="hf_models" # e.g., hf_models/Qwen/Qwen2.5-0.5B

# Construct the full path where the model files are expected to be after download
# This should match how hfd.sh organizes files when using --local-dir
# Based on hfd.sh, if --local-dir is set, it uses that path directly.
# We will instruct hfd.sh to download directly into this specific path.
LOCAL_MODEL_FULL_PATH="${LOCAL_MODEL_STORAGE_BASE_DIR}/${HUGGING_FACE_MODEL_ID}"

EXPECTED_CONFIG_FILE="${LOCAL_MODEL_FULL_PATH}/config.json"

if [ ! -f "${HF_DOWNLOAD_SCRIPT_PATH}" ]; then
  echo "Error: Hugging Face download script (hfd.sh) not found at ${HF_DOWNLOAD_SCRIPT_PATH}. Cannot proceed with automatic download." >&2
  echo "Please ensure it exists and is executable, or download the model manually." >&2
  # Allow to proceed; training script might try to download using HF ID (which could be slow)
else
  if [ ! -x "${HF_DOWNLOAD_SCRIPT_PATH}" ]; then
    echo "Warning: Download script ${HF_DOWNLOAD_SCRIPT_PATH} is not executable. Attempting with 'bash'..." >&2
  fi

  if [ ! -f "${EXPECTED_CONFIG_FILE}" ]; then
    echo "Model config not found at ${EXPECTED_CONFIG_FILE}."
    echo "Attempting to download ${HUGGING_FACE_MODEL_ID} using ${HF_DOWNLOAD_SCRIPT_PATH} to ${LOCAL_MODEL_FULL_PATH}..."
    
    # Ensure the target directory for the model exists before calling hfd.sh
    # hfd.sh with --local-dir should handle the final directory creation, 
    # but let's ensure its parent exists for robustness.
    mkdir -p "${LOCAL_MODEL_FULL_PATH}"
    if [ $? -ne 0 ]; then
        echo "Error: Could not create directory ${LOCAL_MODEL_FULL_PATH}. Please check permissions. Exiting." >&2
        exit 1
    fi

    # Execute the hfd.sh script
    # Pass the REPO_ID and the --local-dir pointing to the exact final path
    echo "Executing: bash ${HF_DOWNLOAD_SCRIPT_PATH} ${HUGGING_FACE_MODEL_ID} --local-dir ${LOCAL_MODEL_FULL_PATH}"
    bash "${HF_DOWNLOAD_SCRIPT_PATH}" "${HUGGING_FACE_MODEL_ID}" --local-dir "${LOCAL_MODEL_FULL_PATH}"
    
    HFD_EXIT_CODE=$?
    if [ ${HFD_EXIT_CODE} -ne 0 ]; then
      echo "Shell: Hugging Face download script (${HF_DOWNLOAD_SCRIPT_PATH}) failed with exit code ${HFD_EXIT_CODE}. Please check errors above. Exiting." >&2
      # Consider removing LOCAL_MODEL_FULL_PATH if download failed partway
      # rm -rf "${LOCAL_MODEL_FULL_PATH}"
      exit 1
    fi
    
    # Verify again if download was successful by checking the config file
    if [ ! -f "${EXPECTED_CONFIG_FILE}" ]; then
        echo "Shell: Model download script ran, but ${EXPECTED_CONFIG_FILE} is still missing. Download may have failed internally or hfd.sh usage is incorrect. Exiting." >&2
        exit 1
    fi
    echo "Shell: Model ${HUGGING_FACE_MODEL_ID} downloaded successfully to ${LOCAL_MODEL_FULL_PATH}."
  else
    echo "Shell: Model ${HUGGING_FACE_MODEL_ID} found locally at ${LOCAL_MODEL_FULL_PATH}."
  fi
fi
# --- End of Model Download Logic ---

# --- Training Parameters ---
# 这些是您可以根据需要修改的主要参数

# Model and Data Paths
if [ -f "${EXPECTED_CONFIG_FILE}" ]; then
  TRAINING_MODEL_PATH="${LOCAL_MODEL_FULL_PATH}"
  echo "Using local model path for training: ${TRAINING_MODEL_PATH}"
else
  echo "Warning: Local model not available at ${EXPECTED_LOCAL_MODEL_PATH} (or download script ${HF_DOWNLOAD_SCRIPT_PATH} missing/failed)." >&2
  echo "Falling back to Hugging Face ID for MODEL_PATH: ${HUGGING_FACE_MODEL_ID}" >&2
  TRAINING_MODEL_PATH="${HUGGING_FACE_MODEL_ID}"
fi

DATA_PATH="/gz-data/datasets/OpenMathReasoning/data/" # 训练数据路径
OUTPUT_DIR="out_qwen_reasoning_distill" # 模型检查点和输出的保存目录 (相对于此脚本位置)
LOG_DIR="logs_qwen_reasoning_distill"   # 日志文件保存目录 (相对于此脚本位置)

# Training Hyperparameters
EPOCHS=3
BATCH_SIZE=4          # 单个 GPU 上的批处理大小
ACCUMULATION_STEPS=4  # 梯度累积步数 (有效批处理大小 = BATCH_SIZE * ACCUMULATION_STEPS)
LEARNING_RATE=5e-6
MAX_SEQ_LEN=8192      # 输入序列的最大长度
GRAD_CLIP=1.0

# Hardware and Precision
DEVICE="cuda:0"       # 使用的设备 (例如 "cuda:0", "cpu")
DTYPE="bfloat16"      # 数据类型 ("bfloat16", "float16", or "float32")

# Quantization (set to true or false)
LOAD_IN_4BIT=false
LOAD_IN_8BIT=false

# Logging and Saving
LOG_INTERVAL=10       # 每 N 步记录一次日志
SAVE_INTERVAL=200     # 每 N 步保存一次检查点 (注意: 脚本也会在每个epoch结束时保存)
USE_WANDB=false        # 是否使用 Weights & Biases (true/false)
WANDB_PROJECT="Qwen-Reasoning-Distill" # WandB 项目名称 (如果 USE_WANDB=true)

# DDP (Distributed Data Parallel) - 对于单GPU训练，保持 DDP=false
# 如果要使用 DDP，您需要使用 torchrun 或类似工具来启动，并相应地设置 DDP=true
# 例如: torchrun --standalone --nproc_per_node=NUM_GPUS run_train.sh (并在此脚本中处理DDP参数)
DDP=false # 当前脚本主要为单GPU设计

# --- Script Execution ---

# 构建 Python 脚本的参数
CMD_ARGS=""
CMD_ARGS+=" --model_path ${TRAINING_MODEL_PATH}"
CMD_ARGS+=" --data_path ${DATA_PATH}"
CMD_ARGS+=" --out_dir ${OUTPUT_DIR}"
CMD_ARGS+=" --log_dir ${LOG_DIR}"
CMD_ARGS+=" --epochs ${EPOCHS}"
CMD_ARGS+=" --batch_size ${BATCH_SIZE}"
CMD_ARGS+=" --accumulation_steps ${ACCUMULATION_STEPS}"
CMD_ARGS+=" --learning_rate ${LEARNING_RATE}"
CMD_ARGS+=" --max_seq_len ${MAX_SEQ_LEN}"
CMD_ARGS+=" --grad_clip ${GRAD_CLIP}"
CMD_ARGS+=" --device ${DEVICE}"
CMD_ARGS+=" --dtype ${DTYPE}"
CMD_ARGS+=" --log_interval ${LOG_INTERVAL}"
CMD_ARGS+=" --save_interval ${SAVE_INTERVAL}"

if [ "$USE_WANDB" = true ]; then
  CMD_ARGS+=" --use_wandb"
  CMD_ARGS+=" --wandb_project ${WANDB_PROJECT}"
fi

if [ "$LOAD_IN_4BIT" = true ]; then
  CMD_ARGS+=" --load_in_4bit"
fi

if [ "$LOAD_IN_8BIT" = true ]; then
  CMD_ARGS+=" --load_in_8bit"
fi

if [ "$DDP" = true ]; then
  CMD_ARGS+=" --ddp"
  # 注意: 对于DDP，您通常需要通过 torchrun 启动，
  # 例如: torchrun --standalone --nproc_per_node=2 trainer/train_qwen_reasoning_distill.py $CMD_ARGS
  # 这个基础脚本假设单GPU，所以我们直接用python执行
fi

# 创建输出和日志目录 (如果不存在)
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# 执行训练脚本
echo "Starting training with the following command:"
echo "python trainer/train_qwen_reasoning_distill.py ${CMD_ARGS}"
echo "--------------------------------------------------"

python trainer/train_qwen_reasoning_distill.py ${CMD_ARGS}

echo "--------------------------------------------------"
echo "Training finished." 