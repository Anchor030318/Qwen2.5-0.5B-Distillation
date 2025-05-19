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
  echo "Error: Hugging Face download script (hfd.sh) not found at ${HF_DOWNLOAD_SCRIPT_PATH}. Cannot proceed." >&2
  # Allow to proceed if model might be manually placed or handled by python script
else
  if [ ! -x "${HF_DOWNLOAD_SCRIPT_PATH}" ]; then
    echo "Warning: Download script ${HF_DOWNLOAD_SCRIPT_PATH} is not executable. Attempting with 'bash'..." >&2
  fi

  if [ ! -f "${EXPECTED_CONFIG_FILE}" ]; then
    echo "Model config not found at ${EXPECTED_CONFIG_FILE}."
    echo "Attempting to download ${HUGGING_FACE_MODEL_ID} using ${HF_DOWNLOAD_SCRIPT_PATH} to ${LOCAL_MODEL_FULL_PATH}..."
    
    mkdir -p "${LOCAL_MODEL_FULL_PATH}"
    if [ $? -ne 0 ]; then
        echo "Error: Could not create directory ${LOCAL_MODEL_FULL_PATH}. Please check permissions. Exiting." >&2
        exit 1
    fi

    echo "Executing: bash ${HF_DOWNLOAD_SCRIPT_PATH} ${HUGGING_FACE_MODEL_ID} --local-dir ${LOCAL_MODEL_FULL_PATH}"
    bash "${HF_DOWNLOAD_SCRIPT_PATH}" "${HUGGING_FACE_MODEL_ID}" --local-dir "${LOCAL_MODEL_FULL_PATH}"
    
    HFD_EXIT_CODE=$?
    if [ ${HFD_EXIT_CODE} -ne 0 ]; then
      echo "Shell: Hugging Face download script (${HF_DOWNLOAD_SCRIPT_PATH}) failed with exit code ${HFD_EXIT_CODE}. Please check errors above. Exiting." >&2
      exit 1
    fi
    
    if [ ! -f "${EXPECTED_CONFIG_FILE}" ]; then
        echo "Shell: Model download script ran, but ${EXPECTED_CONFIG_FILE} is still missing. Download may have failed. Exiting." >&2
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

DATA_PATH="/gz-data/datasets/OpenMathReasoning/data/cot-00000-of-00144.parquet" # 训练数据路径
OUTPUT_DIR="out_qwen_reasoning_distill_continued" # Changed output dir to avoid overwriting previous run
LOG_DIR="logs_qwen_reasoning_distill_continued"   # Changed log dir for the continued run

# Training Hyperparameters
EPOCHS=3 # Total number of epochs desired for the training session.
         # If resuming, training will start from (last_completed_epoch + 1) up to this number.
BATCH_SIZE=2          
ACCUMULATION_STEPS=4  
LEARNING_RATE=5e-6    # Consider if you want to adjust LR for continued training (e.g., smaller LR)
MAX_SEQ_LEN=1024      
GRAD_CLIP=1.0

# Hardware and Precision
DEVICE="cuda:0"       
DTYPE="float16"      

# Quantization (set to true or false)
LOAD_IN_4BIT=false
LOAD_IN_8BIT=false

# Logging and Saving
LOG_INTERVAL=10       
SAVE_INTERVAL=200     
USE_WANDB=false        
WANDB_PROJECT="Qwen-Reasoning-Distill" # WandB 项目名称 (如果 USE_WANDB=true)

# DDP (Distributed Data Parallel) - 对于单GPU训练，保持 DDP=false
# 如果要使用 DDP，您需要使用 torchrun 或类似工具来启动，并相应地设置 DDP=true
# 例如: torchrun --standalone --nproc_per_node=NUM_GPUS run_train.sh (并在此脚本中处理DDP参数)
DDP=false # 当前脚本主要为单GPU设计
NUM_WORKERS=0 # As per previous debugging, keeping num_workers at 0

# --- Script Execution ---

# 构建 Python 脚本的参数
CMD_ARGS=""

# If RESUME_MODEL_WEIGHTS_FROM is set and exists, pass it to --resume_from_checkpoint
# The python script's --model_path will be effectively overridden by --resume_from_checkpoint for loading model/tokenizer.
if [ -n "${RESUME_MODEL_WEIGHTS_FROM}" ] && [ -d "${RESUME_MODEL_WEIGHTS_FROM}" ]; then
  echo "Attempting to load model/tokenizer from checkpoint: ${RESUME_MODEL_WEIGHTS_FROM}"
  CMD_ARGS+=" --resume_from_checkpoint ${RESUME_MODEL_WEIGHTS_FROM}"
  # The python script now uses resume_from_checkpoint as the primary source for model/tokenizer if provided.
  # We still pass a model_path, which can be the resume path or a base path if resume_from_checkpoint is not set.
  CMD_ARGS+=" --model_path ${RESUME_MODEL_WEIGHTS_FROM}" 
elif [ -f "${EXPECTED_CONFIG_FILE}" ]; then
  echo "Starting new training (or no valid resume path specified), using base model: ${TRAINING_MODEL_PATH}"
  CMD_ARGS+=" --model_path ${TRAINING_MODEL_PATH}"
else
  echo "Warning: Local base model not found at ${EXPECTED_CONFIG_FILE} and no resume path. Falling back to Hugging Face ID for MODEL_PATH: ${HUGGING_FACE_MODEL_ID}" >&2
  CMD_ARGS+=" --model_path ${HUGGING_FACE_MODEL_ID}"
fi

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
CMD_ARGS+=" --num_workers ${NUM_WORKERS}"

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
echo "Starting/Resuming training with the following command:"
echo "python trainer/train_qwen_reasoning_distill.py ${CMD_ARGS}"
echo "--------------------------------------------------"

python trainer/train_qwen_reasoning_distill.py ${CMD_ARGS}

echo "--------------------------------------------------"
echo "Training finished." 