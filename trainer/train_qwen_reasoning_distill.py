import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import sys
print(f"--- PyTorch Version from script: {torch.__version__} ---")
print(f"--- CUDA Available from script: {torch.cuda.is_available()} ---")
if torch.cuda.is_available():
    print(f"--- CUDA Version from script: {torch.version.cuda} ---")
    print(f"--- cuDNN Version from script: {torch.backends.cudnn.version()} ---")
    print(f"--- GPU Name: {torch.cuda.get_device_name(0)} ---")
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataset.dataset import SFTDataset
from Util.utils import setup_logger
from trainer.custom_loss_module import calculate_efficiency_loss_reward, calculate_repetition_penalty

warnings.filterwarnings('ignore')


def Logger(content, logger=None):
    if not ddp or dist.get_rank() == 0:
        if logger:
            logger.info(content)
        else:
            print(content)


def get_lr(current_step, total_steps, lr):
    """
    Cosine learning rate schedule with warmup.
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb, logger, optimizer, scaler, model, tokenizer):
    # Get special token IDs for reasoning, to apply higher weight
    start_of_think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    end_of_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    start_of_answer_ids = tokenizer.encode("<answer>", add_special_tokens=False)
    end_of_answer_ids = tokenizer.encode("</answer>", add_special_tokens=False)
    
    # Flatten lists of token IDs for all relevant special markers
    all_special_marker_ids = list(set(start_of_think_ids + end_of_think_ids + start_of_answer_ids + end_of_answer_ids))
    
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    model.train()
    
    for step, batch_data in enumerate(train_loader):
        if batch_data is None: # Handle skipped batch from DataLoader
            if not ddp or dist.get_rank() == 0: # Log only on rank 0 for DDP
                logger.warning(f"Epoch {epoch+1}, Step {step+1}/{iter_per_epoch}: Skipping empty batch (all items filtered out).")
            continue # Skip to the next step

        input_ids = batch_data["input_ids"].to(args.device)
        target_ids = batch_data["labels"].to(args.device) 
        # loss_mask from dataset is the binary mask for target tokens (non-prompt, non-padding)
        binary_target_mask = batch_data["loss_mask"].to(args.device)
        
        # CRITICAL: answer_section_mask marks the actual answer tokens within target_ids (e.g., between <answer> and </answer>).
        # It MUST be provided by your SFTDataset. Shape: [batch_size, seq_len], boolean.
        # It should be 1 for answer tokens that are part of the target (non-padding), 0 otherwise.
        answer_section_mask = batch_data.get("answer_mask", torch.zeros_like(target_ids, dtype=torch.bool)).to(args.device)
        
        # NEW: think_section_mask marks the thinking process tokens (e.g., between <think> and </think>).
        # It MUST be provided by your SFTDataset. Shape: [batch_size, seq_len], boolean.
        think_section_mask = batch_data.get("think_mask", torch.zeros_like(target_ids, dtype=torch.bool)).to(args.device)
        
        # Apply learning rate schedule
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            outputs = model(input_ids=input_ids, labels=target_ids)
            logits = outputs.logits
            
            # Calculate loss with manual masking
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            shift_binary_target_mask = binary_target_mask[..., 1:].contiguous()
            # Shift the answer_section_mask to align with shift_labels and unweighted_token_loss
            shifted_answer_mask = answer_section_mask[..., 1:].contiguous()
            # Shift the think_section_mask as well
            shifted_think_mask = think_section_mask[..., 1:].contiguous()
            
            # --- 1. Base Distillation Loss (Weighted Cross Entropy) ---
            unweighted_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size()) # Reshape to [batch_size, seq_len-1]
            
            weighted_loss_mask = shift_binary_target_mask.float() # Start with binary target mask
            is_special_marker = torch.zeros_like(shift_labels, dtype=torch.bool)
            for token_id in all_special_marker_ids:
                is_special_marker = is_special_marker | (shift_labels == token_id)
            
            actual_special_markers_in_target = is_special_marker & shift_binary_target_mask.bool()
            # Use args.special_token_weight for weighting these tokens
            weighted_loss_mask[actual_special_markers_in_target] = args.special_token_weight 
            
            num_target_tokens = shift_binary_target_mask.float().sum()
            if num_target_tokens > 0:
                base_distill_loss = (unweighted_token_loss * weighted_loss_mask).sum() / num_target_tokens
            else:
                base_distill_loss = torch.tensor(0.0, device=args.device)

            # --- 2. Efficiency Loss/Reward (L_eff) ---
            L_eff = torch.tensor(0.0, device=args.device)
            if args.eff_loss_weight > 0: # Calculate only if weight is positive
                L_eff = calculate_efficiency_loss_reward(
                    unweighted_token_loss_batch=unweighted_token_loss,
                    shifted_labels_batch=shift_labels,
                    shifted_answer_mask_batch=shifted_answer_mask,
                    shift_binary_target_mask_batch=shift_binary_target_mask.bool(), # Ensure boolean
                    eff_loss_threshold=args.eff_loss_threshold,
                    eff_reward_coeff=args.eff_reward_coeff,
                    eff_penalty_coeff=args.eff_penalty_coeff,
                    device=args.device
                )

            # --- 3. Repetition Penalty (L_rep) ---
            L_rep = torch.tensor(0.0, device=args.device)
            if args.rep_loss_weight > 0: # Calculate only if weight is positive
                L_rep = calculate_repetition_penalty(
                    shifted_labels_batch=shift_labels,
                    shifted_think_mask_batch=shifted_think_mask, # Pass the shifted think mask
                    shift_binary_target_mask_batch=shift_binary_target_mask.bool(), # Ensure boolean
                    rep_ngram_n=args.rep_ngram_n,
                    rep_penalty_coeff=args.rep_penalty_coeff, 
                    device=args.device
                )
            
            # --- Total Loss ---
            # Apply weights to the custom loss components
            final_loss = base_distill_loss + args.eff_loss_weight * L_eff + args.rep_loss_weight * L_rep
            
            # Apply gradient accumulation scaling to the final combined loss
            loss_to_backward = final_loss / args.accumulation_steps

        scaler.scale(loss_to_backward).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            estimated_time = spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
            
            log_msg = (
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iter_per_epoch}) '
                f'TotalLoss:{final_loss.item():.4f} (Base:{base_distill_loss.item():.4f} Eff:{L_eff.item():.4f} Rep:{L_rep.item():.4f}) '
                f'LR:{optimizer.param_groups[-1]["lr"]:.8f} GradAccumLoss:{loss_to_backward.item():.4f} '
                f'EpochTime:{estimated_time}min'
            )
            Logger(log_msg, logger)

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "total_loss": final_loss.item(),
                    "base_distill_loss": base_distill_loss.item(),
                    "L_eff (weighted)": args.eff_loss_weight * L_eff.item() if args.eff_loss_weight > 0 else 0,
                    "L_rep (weighted)": args.rep_loss_weight * L_rep.item() if args.rep_loss_weight > 0 else 0,
                    "L_eff_raw": L_eff.item(),
                    "L_rep_raw": L_rep.item(),
                    "learning_rate": optimizer.param_groups[-1]['lr'],
                    "epoch_time_minutes": estimated_time,
                    "loss_for_backward": loss_to_backward.item()
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            step_checkpoint_dir = os.path.join(args.save_dir, f'qwen_reasoning_epoch{epoch+1}_step{step+1}')
            save_checkpoint(model, tokenizer, optimizer, scaler, epoch, step_checkpoint_dir)


def save_checkpoint(model, tokenizer, optimizer, scaler, completed_epoch, save_dir_path):
    model.eval()
    os.makedirs(save_dir_path, exist_ok=True)
    
    model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
    model_to_save.save_pretrained(save_dir_path)
    tokenizer.save_pretrained(save_dir_path)
    
    training_states = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'completed_epoch': completed_epoch # 0-indexed
    }
    torch.save(training_states, os.path.join(save_dir_path, 'training_states.pt'))
    
    Logger(f"Checkpoint (model, tokenizer, optimizer, scaler, epoch) saved to {save_dir_path}", logger)
    model.train()


def init_model(path_to_load_from):
    """Initialize the Qwen model and tokenizer"""
    logger.info(f"Initializing Qwen model and tokenizer from: {path_to_load_from}")
    
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.load_in_4bit else None,
            bnb_4bit_quant_type="nf4" if args.load_in_4bit else None,
        )
    else:
        quantization_config = None
    
    tokenizer = AutoTokenizer.from_pretrained(
        path_to_load_from,
        trust_remote_code=True,
        use_fast=False
    )
    
    special_tokens = {"additional_special_tokens": ["<think>", "</think>", "<answer>", "</answer>"]}
    if tokenizer.eos_token not in special_tokens["additional_special_tokens"] and \
       tokenizer.bos_token not in special_tokens["additional_special_tokens"] and \
       tokenizer.pad_token not in special_tokens["additional_special_tokens"] and \
       tokenizer.unk_token not in special_tokens["additional_special_tokens"]:
        # Only add if not already present as other special tokens (avoids duplicate warnings if already set)
        # A more robust check might be needed if tokenizer explicitly handles these as part of vocabulary vs. special map
        num_added_toks = tokenizer.add_special_tokens(special_tokens)
        if num_added_toks > 0:
            logger.info(f"Added {num_added_toks} special tokens: {special_tokens['additional_special_tokens']}")

    model = AutoModelForCausalLM.from_pretrained(
        path_to_load_from,
        device_map="auto" if not ddp else {"": args.device},
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if args.dtype == "float16" else (torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
    )
    
    model.resize_token_embeddings(len(tokenizer))
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f'Model total parameters: {param_count:.3f} million')
    
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(args.device)
        # If mixed precision with float16 is used, ensure model parameters are initially FP32.
        # Autocast will handle the conversion to FP16 for operations.
        if args.dtype == "float16": 
            model = model.float()
        
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen Reasoning Distillation")
    parser.add_argument("--out_dir", type=str, default="../out_qwen_reasoning_distill")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Qwen-Reasoning-Distillation")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data_path", type=str, default="/gz-data/datasets/OpenMathReasoning/data/")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--log_dir", type=str, default="../logs_qwen_reasoning_distill")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to the checkpoint directory to resume training from (e.g., out_qwen_reasoning_distill/qwen_reasoning_epoch2).")
    
    # --- Parameters for Custom Loss Components ---
    parser.add_argument("--special_token_weight", type=float, default=10.0, 
                        help="Weight for special tokens (e.g., <think>, <answer>) in the base distillation loss.")
    # Efficiency Loss/Reward parameters
    parser.add_argument("--eff_loss_weight", type=float, default=0.1, 
                        help="Overall weight for the efficiency loss/reward component in the total loss.")
    parser.add_argument("--eff_loss_threshold", type=float, default=0.5, 
                        help="Mean answer token loss threshold used in sigmoid for L_eff to determine 'correctness'.")
    parser.add_argument("--eff_reward_coeff", type=float, default=0.1, 
                        help="Coefficient for the reward part of L_eff (applied to 1/length for correct short answers).")
    parser.add_argument("--eff_penalty_coeff", type=float, default=0.1, 
                        help="Coefficient for the penalty part of L_eff (applied to length for incorrect long answers).")
    # Repetition Penalty parameters
    parser.add_argument("--rep_loss_weight", type=float, default=0.05, 
                        help="Overall weight for the repetition penalty component in the total loss.")
    parser.add_argument("--rep_ngram_n", type=int, default=4, 
                        help="N-gram size for calculating repetition penalty (L_rep).")
    parser.add_argument("--rep_penalty_coeff", type=float, default=1.0, # Renamed from previous rep_coeff to avoid confusion
                        help="Internal coefficient for the magnitude of the repetition penalty before applying rep_loss_weight.")

    # --- System Prompt for a priori instructions ---
    default_system_prompt = (
        "You are an AI assistant. Your primary task is to solve the given mathematical problem.\n"
        "First, think step-by-step to break down the problem, outline your reasoning, and perform necessary calculations. "
        "Enclose your entire thought process within <think> and </think> tags.\n"
        "After your thought process, clearly state the final answer. The final answer must be enclosed within \\boxed{}. "
        "For example: <think>This is step 1... This is step 2... Thus the answer is X.</think>The final answer is \\boxed{X}."
    )
    parser.add_argument("--system_prompt", type=str, default=default_system_prompt, 
                        help="The system prompt to prepend to each training example to guide model behavior.")

    args = parser.parse_args()

    # Setup logger
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger("qwen_reasoning_distill", log_dir=args.log_dir)
    
    # Create output directory
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Determine device type
    device_type = "cuda" if "cuda" in args.device else "cpu"
    
    # Set wandb run name
    args.wandb_run_name = f"Qwen2.5-0.5B-Reasoning-Distill-Epochs-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"

    # Set up context for mixed precision training
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=torch.float16 if args.dtype=="float16" else (torch.bfloat16 if args.dtype=="bfloat16" else torch.float32) )
    
    # Check for distributed training
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    
    # Set seed for reproducibility
    base_seed = 42
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)
        logger.info(f"DDP initialized. Rank: {rank}, Local rank: {ddp_local_rank}, World size: {dist.get_world_size()}")

    # Initialize wandb if needed
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        logger.info(f"WandB initialized with project: {args.wandb_project}, run: {args.wandb_run_name}")
    else:
        wandb = None

    # Determine path for loading initial model/tokenizer
    path_for_model_tokenizer = args.model_path
    if args.resume_from_checkpoint:
        if not os.path.isdir(args.resume_from_checkpoint):
            logger.error(f"Resume checkpoint path {args.resume_from_checkpoint} not found or not a directory. Exiting.")
            sys.exit(1)
        path_for_model_tokenizer = args.resume_from_checkpoint
        logger.info(f"Resuming. Model and tokenizer will be loaded from: {path_for_model_tokenizer}")
    else:
        logger.info(f"Starting new training. Model and tokenizer will be loaded from: {args.model_path}")

    # Initialize model and tokenizer
    model, tokenizer = init_model(path_for_model_tokenizer)

    # Setup dataset and dataloader
    logger.info(f"Loading dataset from: {args.data_path}")
    # Pass the system_prompt to SFTDataset
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len, log_dir=args.log_dir, system_prompt=args.system_prompt)
    
    # Setup sampler for distributed training
    train_sampler = DistributedSampler(train_ds) if ddp else None
    
    # Setup dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=not ddp,
        num_workers=0,
        sampler=train_sampler,
        collate_fn=train_ds.prepare_batch
    )

    # Setup optimizer and scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16' or args.dtype == 'bfloat16'))
    
    # Setup optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    if ddp and not (args.load_in_8bit or args.load_in_4bit):
        # Wrap model with DDP
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
        logger.info("Model wrapped with DistributedDataParallel")

    # Training loop
    iter_per_epoch = len(train_loader)
    logger.info(f"Starting training for {args.epochs} epochs, {iter_per_epoch} iterations per epoch")
    
    start_epoch = 0
    if args.resume_from_checkpoint:
        training_states_path = os.path.join(args.resume_from_checkpoint, 'training_states.pt')
        if os.path.isfile(training_states_path):
            logger.info(f"Loading optimizer, scaler, and epoch states from {training_states_path}")
            # map_location ensures tensors are loaded to the correct device, esp. if resuming on different GPU setup
            checkpoint_states = torch.load(training_states_path, map_location=args.device) 
            
            try:
                optimizer.load_state_dict(checkpoint_states['optimizer_state_dict'])
                logger.info("Loaded optimizer state.")
            except Exception as e:
                logger.error(f"Could not load optimizer state: {e}. Optimizer will be reinitialized.")
            
            if 'scaler_state_dict' in checkpoint_states and (args.dtype == 'float16' or args.dtype == 'bfloat16'):
                 try:
                    scaler.load_state_dict(checkpoint_states['scaler_state_dict'])
                    logger.info("Loaded scaler state.")
                 except Exception as e:
                    logger.warning(f"Could not load scaler state: {e}. Continuing with a new scaler state.")
            elif (args.dtype == 'float16' or args.dtype == 'bfloat16'):
                logger.warning("Scaler state not found in checkpoint, but mixed precision is enabled. Initializing new scaler state.")
            
            loaded_completed_epoch = checkpoint_states.get('completed_epoch', -1)
            if loaded_completed_epoch != -1:
                start_epoch = loaded_completed_epoch + 1
                logger.info(f"Resuming from epoch {start_epoch} (0-indexed). Last completed epoch: {loaded_completed_epoch}.")
            else: # Should not happen if 'completed_epoch' key is present and valid
                logger.warning("Key 'completed_epoch' not found or invalid in checkpoint_states. Defaulting to start_epoch 0.")
        else:
            logger.warning(f"Training states file 'training_states.pt' not found in {args.resume_from_checkpoint}. "
                           "Model weights and tokenizer from checkpoint dir will be used. Starting optimizer, scaler, and epoch from scratch.")

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        train_epoch(epoch, wandb, logger, optimizer, scaler, model, tokenizer)
        
        if not ddp or dist.get_rank() == 0:
            epoch_checkpoint_dir = os.path.join(args.save_dir, f'qwen_reasoning_epoch{epoch+1}')
            save_checkpoint(model, tokenizer, optimizer, scaler, epoch, epoch_checkpoint_dir)
    
    if not ddp or dist.get_rank() == 0:
        final_checkpoint_dir = os.path.join(args.save_dir, 'qwen_reasoning_final')
        # For final save, completed_epoch is the last epoch number that ran, which is args.epochs - 1
        save_checkpoint(model, tokenizer, optimizer, scaler, args.epochs - 1, final_checkpoint_dir)
        logger.info("Training completed successfully!")
    
    if ddp:
        dist.destroy_process_group() 