import os
import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import random
from Util.utils import setup_logger

class SFTDataset(Dataset):
    """
    Dataset for SFT training. Loads data containing a problem and a generated solution 
    (expected to be in a format like "Question: <problem>\n<think><CoT></think><answer><answer_text></answer>").
    Generates a single loss mask for distillation targets.
    """
    
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer, max_length=1024, shuffle=True, log_dir=None):
        self.logger = setup_logger("SFTDataset", log_dir=log_dir)
        self.logger.info(f"Initializing SFTDataset with data from {data_path}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Special tokens are expected to be part of the tokenizer already if they are in generated_solution
        # No explicit storage of their IDs here, as loss masking is based on prompt length primarily.

        # Load data
        if os.path.isdir(data_path):
            import pandas as pd
            parquet_files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
            self.logger.info(f"Found {len(parquet_files)} parquet files in {data_path}")
            for file in parquet_files:
                file_path = os.path.join(data_path, file)
                try:
                    df = pd.read_parquet(file_path)
                    self.data.extend(df.to_dict('records'))
                except Exception as e:
                    self.logger.error(f"Error loading {file}: {e}")
            self.logger.info(f"Finished loading from directory. Total samples from parquet: {len(self.data)}")
        elif data_path.endswith(('.jsonl', '.json')):
            self.logger.info(f"Loading data from JSONL/JSON file: {data_path}")
            with open(data_path, 'r') as f:
                for line_num, line in enumerate(f):
                    try: self.data.append(json.loads(line))
                    except json.JSONDecodeError: self.logger.error(f"Error decoding JSON line {line_num+1}")
            self.logger.info(f"Loaded {len(self.data)} samples.")
        elif data_path.endswith('.parquet'):
            import pandas as pd
            self.logger.info(f"Loading data from single parquet file: {data_path}")
            try: 
                df = pd.read_parquet(data_path)
                self.data = df.to_dict('records')
                self.logger.info(f"Loaded {len(self.data)} samples.")
            except ImportError: self.logger.error("pandas/pyarrow not installed for parquet."); raise
            except Exception as e: self.logger.error(f"Error loading parquet {data_path}: {e}")
        else:
            self.logger.error(f"Unsupported file format: {data_path}"); raise ValueError("Unsupported data format")

        if not self.data: self.logger.error(f"No data loaded from {data_path}.");
        if shuffle: random.shuffle(self.data); self.logger.info("Data shuffled.")
        
        if self.data:
            sample = self.data[0]
            self.logger.info(f"Sample data fields: {list(sample.keys())}")
            # Expect 'problem' and 'generated_solution' for basic distillation
            if not all(k in sample for k in ['problem', 'generated_solution']):
                 self.logger.warning("Sample data missing 'problem' or 'generated_solution' field.")
        else: self.logger.warning("Dataset is empty after initialization.")
        self.logger.info(f"Dataset initialization complete. Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question_str = item.get('problem', '')
        # generated_solution is expected to contain the full teacher output, 
        # including <think>CoT</think><answer>teacher_answer</answer>
        generated_solution_str = item.get('generated_solution', '') 

        # Construct the prompt part and the full text for tokenization
        # The prompt defines what NOT to include in the loss
        prompt_text = f"Question: {question_str}\\n" # Or any other fixed prompt structure
        full_text = f"{prompt_text}{generated_solution_str}"

        tokenized_output = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length", # Pad to max_length
            truncation=True,
            return_attention_mask=True
        )

        input_ids = torch.tensor(tokenized_output["input_ids"])
        attention_mask = torch.tensor(tokenized_output["attention_mask"]) # 1 for real tokens, 0 for padding

        labels = input_ids.clone() # Standard for language modeling/distillation

        # --- Create a single binary loss_mask --- 
        # 0 for prompt & padding, 1 for all other tokens (distillation targets)
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long) # Use long or bool, ensure trainer handles type
        
        # Tokenize prompt_text separately to find its length
        # This should not have BOS/EOS if we're matching its length inside the main tokenized sequence.
        prompt_tokens_for_length_calc = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        len_prompt_tokens = len(prompt_tokens_for_length_calc)

        start_of_targets_idx = 0
        # Adjust for BOS token if tokenizer adds it to the beginning of input_ids
        if self.tokenizer.bos_token_id is not None and input_ids.numel() > 0 and input_ids[0] == self.tokenizer.bos_token_id:
            start_of_targets_idx = 1 
        start_of_targets_idx += len_prompt_tokens # Tokens from here onwards are targets

        # Set 1 for all target tokens
        if input_ids.numel() > start_of_targets_idx:
            loss_mask[start_of_targets_idx:] = 1
        
        # Ensure padding tokens are 0 in the loss_mask
        loss_mask = loss_mask * attention_mask # Element-wise multiplication
        
        return input_ids, labels, loss_mask
        
    def prepare_batch(self, batch):
        """
        Process a batch for training.
        Args:
            batch: List of tuples (input_ids, labels, loss_mask)
        Returns:
            Dictionary with batch data
        """
        input_ids, labels, loss_masks = zip(*batch)
        
        input_ids_stacked = torch.stack(input_ids)
        labels_stacked = torch.stack(labels)
        loss_mask_stacked = torch.stack(loss_masks)
        
        return {
            "input_ids": input_ids_stacked,
            "labels": labels_stacked,
            "loss_mask": loss_mask_stacked
        } 