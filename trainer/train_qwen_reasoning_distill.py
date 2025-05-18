import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataset.dataset import SFTDataset
from Util.utils import setup_logger

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


def train_epoch(epoch, wandb, logger):
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
    
    for step, batch_data in enumerate(train_loader): # batch_data is a dict
        input_ids = batch_data["input_ids"].to(args.device)
        target_ids = batch_data["labels"].to(args.device) 
        # loss_mask from dataset is the binary mask for target tokens (non-prompt, non-padding)
        binary_target_mask = batch_data["loss_mask"].to(args.device)
        
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
            
            # Calculate raw per-token loss
            unweighted_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size()) # Reshape to [batch_size, seq_len-1]
            
            # Create weighted_loss_mask based on the binary_target_mask
            # Start with binary target mask (0 for non-targets, 1 for targets)
            weighted_loss_mask = shift_binary_target_mask.float() # Ensure it's float for weights
            
            # Identify special marker tokens in the shifted labels
            is_special_marker = torch.zeros_like(shift_labels, dtype=torch.bool)
            for token_id in all_special_marker_ids:
                is_special_marker = is_special_marker | (shift_labels == token_id)
            
            # Apply higher weight to special markers that are also target tokens
            # The value 10.0 was used in the original script provided in context earlier.
            # Let's use a configurable arg later if needed, for now, hardcode or use existing.
            # The original script had a hardcoded 10.0 for special tokens.
            actual_special_markers_in_target = is_special_marker & shift_binary_target_mask.bool()
            weighted_loss_mask[actual_special_markers_in_target] = 10.0 
            
            # Normalize loss by the sum of effective weights in weighted_loss_mask
            # or by the number of 0/1 target tokens (shift_binary_target_mask.sum())?
            # Original script normalized by sum of (0/1 mask that had 10s for special tokens), which is weighted sum.
            # A more standard way is to normalize by the count of target tokens.
            # Let's use sum of weights as per apparent original logic. If sum is 0, loss is 0.
            effective_loss_weights_sum = weighted_loss_mask.sum()

            if effective_loss_weights_sum > 0:
                final_loss = (unweighted_token_loss * weighted_loss_mask).sum() / effective_loss_weights_sum
            else:
                final_loss = torch.tensor(0.0, device=args.device) # Avoid division by zero
            
            # Apply gradient accumulation
            final_loss = final_loss / args.accumulation_steps

        scaler.scale(final_loss).backward()

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
                f'loss:{final_loss.item() * args.accumulation_steps:.3f} ' # Log the properly scaled loss
                f'lr:{optimizer.param_groups[-1]["lr"]:.8f} '
                f'epoch_Time:{estimated_time}min'
            )
            Logger(log_msg, logger)

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": final_loss.item() * args.accumulation_steps, # Log the properly scaled loss
                    "lr": optimizer.param_groups[-1]['lr'],
                    "epoch_Time": estimated_time
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            save_checkpoint(model, tokenizer, f'qwen_reasoning_epoch{epoch+1}_step{step+1}')


def save_checkpoint(model, tokenizer, checkpoint_name):
    model.eval()
    save_path = f'{args.save_dir}/{checkpoint_name}'
    os.makedirs(save_path, exist_ok=True)
    
    # Save the model state
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    # Save model and tokenizer
    model.save_pretrained(save_path, state_dict=state_dict)
    tokenizer.save_pretrained(save_path)
    
    Logger(f"Model saved to {save_path}")
    model.train()


def init_model():
    """Initialize the Qwen model and tokenizer"""
    logger.info(f"Initializing Qwen2.5 model from: {args.model_path}")
    
    # Configure quantization if needed
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.load_in_4bit else None,
            bnb_4bit_quant_type="nf4" if args.load_in_4bit else None,
        )
    else:
        quantization_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Add special tokens for reasoning
    special_tokens = {"additional_special_tokens": ["<think>", "</think>", "<answer>", "</answer>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if not ddp else {"": args.device},
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    )
    
    # Resize token embeddings for new special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f'Model total parameters: {param_count:.3f} million')
    
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(args.device)
        
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
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Qwen-Reasoning-Distillation")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data_path", type=str, default="/gz-data/datasets/OpenMathReasoning/data/")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--log_dir", type=str, default="../logs")
    
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
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type)
    
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

    # Initialize model and tokenizer
    model, tokenizer = init_model()

    # Setup dataset and dataloader
    logger.info(f"Loading dataset from: {args.data_path}")
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len, log_dir=args.log_dir)
    
    # Setup sampler for distributed training
    train_sampler = DistributedSampler(train_ds) if ddp else None
    
    # Setup dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=not ddp,
        num_workers=args.num_workers,
        sampler=train_sampler,
        collate_fn=train_ds.prepare_batch
    )

    # Setup optimizer and scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'bfloat16' or args.dtype == 'float16'))
    
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
    
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        train_epoch(epoch, wandb, logger)
        
        # Save checkpoint at the end of each epoch
        if not ddp or dist.get_rank() == 0:
            save_checkpoint(model, tokenizer, f'qwen_reasoning_epoch{epoch+1}')
    
    # Final save
    if not ddp or dist.get_rank() == 0:
        save_checkpoint(model, tokenizer, 'qwen_reasoning_final')
        logger.info("Training completed successfully!") 