import os
import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import random
from Util.utils import setup_logger
import pyarrow.parquet as pq # Added for Parquet metadata
import pandas as pd # Ensure pandas is available for loading data in getitem
import re # Import regex module

class SFTDataset(Dataset):
    """
    Dataset for SFT training. Loads data containing a problem and a generated solution 
    (expected to be in a format like "Question: <problem>\n<think><CoT></think><answer><answer_text></answer>").
    Implements lazy loading for Parquet files in a directory.
    Generates a single loss mask for distillation targets.
    """
    
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer, max_length=1024, shuffle=True, log_dir=None, system_prompt=None):
        self.logger = setup_logger("SFTDataset", log_dir=log_dir)
        self.logger.info(f"Initializing SFTDataset with data from {data_path}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        if self.system_prompt:
            self.logger.info(f"Using system prompt: '{self.system_prompt[:100]}...'") # Log a snippet
        else:
            self.logger.info("No system prompt provided.")
        
        self.file_paths = []
        self.num_rows_per_file = []
        self.cumulative_rows = [0] # Starts with 0 for easier index mapping
        self._total_samples = 0
        self.data_type = None # Will be set based on data_path type

        # Cache for lazy loading
        self.cache_file_path = None
        self.cache_df = None

        # Attempt to get special token IDs once during init for efficiency and early error detection
        try:
            self.think_start_id = self.tokenizer.encode("<think>", add_special_tokens=False)[0]
            self.think_end_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
            self.logger.info(f"Successfully encoded <think> (ID: {self.think_start_id}) and </think> (ID: {self.think_end_id}).")
        except IndexError:
            self.logger.error("CRITICAL: <think> or </think> special tags not tokenizable or yield empty lists upon init. Integrity checks for these tags will effectively be disabled or fail. Ensure they are added to tokenizer.")
            # Set to values that won't match, or handle this as a fatal error depending on requirements
            self.think_start_id = -1000 # Unlikely to match actual token IDs
            self.think_end_id = -1001
        # No longer primarily using <answer> tags for filtering, but keep if useful for other logic later
        # try:
        #     self.answer_start_id = self.tokenizer.encode("<answer>", add_special_tokens=False)[0]
        #     self.answer_end_id = self.tokenizer.encode("</answer>", add_special_tokens=False)[0]
        # except IndexError: 
        #     self.answer_start_id, self.answer_end_id = -1, -1 # Placeholder if not found

        if os.path.isdir(data_path):
            self.data_type = "parquet_dir"
            parquet_files = sorted([f for f in os.listdir(data_path) if f.endswith('.parquet')]) # Sorted for consistent order before shuffle
            self.logger.info(f"Found {len(parquet_files)} parquet files in {data_path}. Reading metadata...")
            
            if not parquet_files:
                self.logger.warning(f"No parquet files found in directory {data_path}.")
            
            for file_name in parquet_files:
                file_path = os.path.join(data_path, file_name)
                try:
                    meta = pq.read_metadata(file_path)
                    num_rows = meta.num_rows
                    if num_rows > 0:
                        self.file_paths.append(file_path)
                        self.num_rows_per_file.append(num_rows)
                    else:
                        self.logger.warning(f"Parquet file {file_path} is empty or has 0 rows, skipping.")
                except Exception as e:
                    self.logger.error(f"Error reading metadata for {file_path}: {e}")

            if shuffle and self.file_paths:
                self.logger.info("Shuffling the order of Parquet files for processing.")
                # Shuffle file_paths and their corresponding num_rows_per_file together
                combined = list(zip(self.file_paths, self.num_rows_per_file))
                random.shuffle(combined)
                if combined: # Ensure not empty after potential filtering
                    self.file_paths, self.num_rows_per_file = zip(*combined)
                    self.file_paths = list(self.file_paths) # Convert back to list
                    self.num_rows_per_file = list(self.num_rows_per_file)
                else:
                    self.file_paths, self.num_rows_per_file = [], []


            # Calculate cumulative rows after potential shuffle of file order
            current_sum = 0
            for num_rows_in_file in self.num_rows_per_file:
                current_sum += num_rows_in_file
                self.cumulative_rows.append(current_sum)
            self._total_samples = current_sum
            
            self.logger.info(f"Finished metadata scan. Total usable samples: {self._total_samples} from {len(self.file_paths)} files.")

        elif data_path.endswith(('.jsonl', '.json')):
            self.data_type = "jsonl"
            self.logger.info(f"Loading data from JSONL/JSON file: {data_path} (full load for this type)")
            self._raw_jsonl_data = [] # Store raw JSONL data here
            try:
                with open(data_path, 'r') as f:
                    for line_num, line in enumerate(f):
                        try: 
                            self._raw_jsonl_data.append(json.loads(line))
                        except json.JSONDecodeError: 
                            self.logger.error(f"Error decoding JSON line {line_num+1} in {data_path}")
                self._total_samples = len(self._raw_jsonl_data)
                if shuffle:
                    self.logger.info("Shuffling JSONL data.")
                    random.shuffle(self._raw_jsonl_data)
                self.logger.info(f"Loaded {self._total_samples} samples from {data_path}.")
            except Exception as e:
                self.logger.error(f"Failed to load JSONL file {data_path}: {e}")
                self._total_samples = 0


        elif data_path.endswith('.parquet'):
            self.data_type = "single_parquet"
            self.logger.info(f"Initializing for single parquet file: {data_path}")
            try: 
                meta = pq.read_metadata(data_path)
                self._total_samples = meta.num_rows
                self.single_parquet_path = data_path # Store path for getitem
                self.logger.info(f"Initialized with {self._total_samples} samples from {data_path}. Data will be loaded on demand.")
            except Exception as e: 
                self.logger.error(f"Error reading metadata for single parquet {data_path}: {e}")
                self._total_samples = 0 # Ensure dataset is empty if metadata fails
                # Consider re-raising if this is critical: raise
        else:
            self.logger.error(f"Unsupported file format or path type: {data_path}")
            raise ValueError(f"Unsupported data format or path: {data_path}")

        if self._total_samples == 0:
             self.logger.warning(f"No data loaded or found from {data_path}. Dataset will be empty.")
        
        # Log sample data structure if possible
        if self._total_samples > 0:
            try:
                # Attempt to get the first item to check its structure
                # This will involve actual data loading for the first item
                sample_item_content = self._get_item_content(0) 
                self.logger.info(f"Sample data fields (from first item): {list(sample_item_content.keys())}")
                if not all(k in sample_item_content for k in ['problem', 'generated_solution']):
                     self.logger.warning("First sample item is missing 'problem' or 'generated_solution' field.")
            except Exception as e:
                self.logger.warning(f"Could not retrieve or inspect the first sample item for schema check: {e}")
        
        self.logger.info(f"Dataset initialization complete. Total samples: {self._total_samples}")

    def __len__(self):
        return self._total_samples

    def _get_item_content(self, idx):
        """Internal method to retrieve the raw content of an item by index."""
        if idx < 0 or idx >= self._total_samples:
            self.logger.error(f"Index {idx} is out of bounds (total_samples: {self._total_samples}).")
            raise IndexError("Dataset index out of range")

        item_content = None

        if self.data_type == "parquet_dir":
            if not self.file_paths: # Should have been caught by _total_samples == 0 earlier
                self.logger.error("Parquet directory mode, but no file paths available.")
                raise RuntimeError("Dataset not properly initialized: no parquet files found or processed.")

            file_idx = -1
            # Find which file this global index maps to
            # bisect_left could be more efficient for large numbers of files
            for i in range(len(self.cumulative_rows) -1): # cumulative_rows[0] is 0
                if self.cumulative_rows[i] <= idx < self.cumulative_rows[i+1]:
                    file_idx = i
                    break
            
            if file_idx == -1:
                # This should not happen if idx is valid and cumulative_rows is correct
                self.logger.error(f"Could not map global index {idx} to a file. Cumulative_rows: {self.cumulative_rows}")
                raise IndexError("Error mapping global index to file segment.")

            target_file_path = self.file_paths[file_idx]
            index_in_file = idx - self.cumulative_rows[file_idx]

            if self.cache_file_path == target_file_path:
                df = self.cache_df
                self.logger.debug(f"Cache hit for {target_file_path}. Using cached DataFrame for global index {idx}.")
            else:
                self.logger.debug(f"Cache miss. Loading {target_file_path} for global index {idx}.")
                try:
                    df = pd.read_parquet(target_file_path)
                    self.cache_df = df
                    self.cache_file_path = target_file_path
                except Exception as e:
                    self.logger.error(f"Error loading parquet file {target_file_path} in _get_item_content: {e}")
                    raise RuntimeError(f"Failed to load parquet file {target_file_path}") from e
            
            try:
                item_content = df.iloc[index_in_file].to_dict()
            except IndexError as e_idx:
                self.logger.error(f"Index {index_in_file} out of bounds for file {target_file_path} (len: {len(df)}). Global idx: {idx}. Error: {e_idx}")
                raise IndexError(f"Index {index_in_file} out of bounds for file {target_file_path}") from e_idx

        elif self.data_type == "jsonl":
            item_content = self._raw_jsonl_data[idx]
        
        elif self.data_type == "single_parquet":
            if self.cache_file_path == self.single_parquet_path:
                df = self.cache_df
                self.logger.debug(f"Cache hit for single parquet file {self.single_parquet_path}.")
            else:
                self.logger.debug(f"Cache miss for single_parquet. Loading {self.single_parquet_path}.")
                try:
                    df = pd.read_parquet(self.single_parquet_path)
                    self.cache_df = df
                    self.cache_file_path = self.single_parquet_path
                except Exception as e:
                    self.logger.error(f"Error loading single parquet file {self.single_parquet_path}: {e}")
                    raise RuntimeError(f"Failed to load single parquet file {self.single_parquet_path}") from e
            try:
                item_content = df.iloc[idx].to_dict()
            except IndexError as e_idx:
                self.logger.error(f"Index {idx} out of bounds for single_parquet file {self.single_parquet_path} (len: {len(df)}). Error: {e_idx}")
                raise IndexError(f"Index {idx} out of bounds for file {self.single_parquet_path}") from e_idx
        else:
            self.logger.error(f"Unknown or unhandled data_type '{self.data_type}' in _get_item_content.")
            raise RuntimeError(f"Invalid dataset type '{self.data_type}' during item retrieval.")

        if item_content is None:
            self.logger.error(f"Item content is None for index {idx} with data_type {self.data_type}. This should not happen.")
            raise ValueError(f"Failed to retrieve content for index {idx}")
            
        return item_content

    def __getitem__(self, idx):
        raw_item = self._get_item_content(idx)
        question_str = raw_item.get('problem', '')
        generated_solution_str = raw_item.get('generated_solution', '')

        # Prepend system prompt if available
        effective_prompt_text = f"Question: {question_str}\n"
        if self.system_prompt:
            effective_prompt_text = f"{self.system_prompt}\nQuestion: {question_str}\n" # Ensure newline separation

        full_text = f"{effective_prompt_text}{generated_solution_str}"

        tokenized_output = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_offsets_mapping=True # Ensure offset mapping is returned
        )

        input_ids = torch.tensor(tokenized_output["input_ids"])
        attention_mask = torch.tensor(tokenized_output["attention_mask"])
        offsets = tokenized_output["offset_mapping"]
        labels = input_ids.clone()
        
        # Determine start of solution part (after prompt and potential BOS token)
        prompt_tokens_for_length_calc = self.tokenizer.encode(effective_prompt_text, add_special_tokens=False)
        len_prompt_tokens = len(prompt_tokens_for_length_calc)
        start_of_targets_idx = 0
        if self.tokenizer.bos_token_id is not None and input_ids.numel() > 0 and input_ids[0] == self.tokenizer.bos_token_id:
            start_of_targets_idx = 1 
        start_of_targets_idx += len_prompt_tokens

        # ---- Filtering and Mask Creation ----
        # 1. Check for oxed{...} answer presence in the original string (generated_solution_str)
        boxed_match = re.search(r"\\boxed{(.*?)}", generated_solution_str, re.DOTALL)
        if not boxed_match:
            self.logger.debug(f"Item {idx}: No \\boxed{{...}} found. Skipping.")
            return None # Skip if no boxed answer

        # Initialize masks
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        think_mask_sample = torch.zeros_like(input_ids, dtype=torch.bool)
        answer_mask_sample = torch.zeros_like(input_ids, dtype=torch.bool)

        # Populate base loss_mask for the target section (solution part)
        if input_ids.numel() > start_of_targets_idx:
            loss_mask[start_of_targets_idx:] = 1
        loss_mask = loss_mask * attention_mask # Ensure only non-padding tokens are included

        # 2. <think></think> integrity and mask creation
        # Search for think tags in the tokenized solution part
        solution_input_ids = input_ids[start_of_targets_idx:]
        solution_attention_mask = attention_mask[start_of_targets_idx:]
        actual_solution_len = (solution_attention_mask != 0).sum().item()
        
        # Find first occurrences relative to the start of input_ids
        pos_think_start_abs, pos_think_end_abs = -1, -1
        temp_ids_list = input_ids.tolist() # For easier searching with .index()

        try: 
            # Search only within the actual non-padded, non-prompt part of input_ids
            first_think_start_rel = temp_ids_list[start_of_targets_idx : start_of_targets_idx + actual_solution_len].index(self.think_start_id)
            pos_think_start_abs = start_of_targets_idx + first_think_start_rel
        except ValueError: pass # think_start_id not found

        if pos_think_start_abs != -1: # If <think> is found
            try:
                # Search for </think> *after* <think>
                first_think_end_rel = temp_ids_list[pos_think_start_abs + 1 : start_of_targets_idx + actual_solution_len].index(self.think_end_id)
                pos_think_end_abs = pos_think_start_abs + 1 + first_think_end_rel
            except ValueError: pass # think_end_id not found after think_start_id
            
            if pos_think_end_abs == -1: # <think> found but no subsequent </think>
                self.logger.debug(f"Item {idx}: Incomplete <think> block (start at {pos_think_start_abs}, no end). Skipping.")
                return None # Skip for incomplete think block
            
            # Populate think_mask for tokens between <think> and </think>
            # Excluding the tags themselves
            if pos_think_end_abs > pos_think_start_abs + 1: # Ensure there are tokens between tags
                 for i in range(pos_think_start_abs + 1, pos_think_end_abs):
                    if attention_mask[i] == 1: # Ensure it's not a padding token somehow
                        think_mask_sample[i] = True
        else: # No <think> tag found, check if an orphan </think> exists
            try:
                temp_ids_list[start_of_targets_idx : start_of_targets_idx + actual_solution_len].index(self.think_end_id)
                self.logger.debug(f"Item {idx}: Orphan </think> found without preceding <think>. Skipping.")
                return None # Skip for orphan </think>
            except ValueError: pass # No orphan </think>, which is fine if no <think> either

        # 3. Populate answer_mask using offset_mapping for oxed{} content
        # Boxed match was confirmed earlier. Content is boxed_match.group(1)
        # Character span of the content *inside* oxed{} within generated_solution_str
        boxed_content_char_start_in_solution = boxed_match.start(1)
        boxed_content_char_end_in_solution = boxed_match.end(1)
        
        # Convert to character span within full_text
        prompt_len_chars = len(effective_prompt_text)
        target_content_char_start_in_full = prompt_len_chars + boxed_content_char_start_in_solution
        target_content_char_end_in_full = prompt_len_chars + boxed_content_char_end_in_solution

        for token_idx, (offset_char_start, offset_char_end) in enumerate(offsets):
            if attention_mask[token_idx] == 0 or offset_char_start is None or offset_char_end is None or offset_char_end == 0: 
                continue # Skip padding, special tokens with no span, or (0,0) offsets
            
            # Check if the token is part of the target solution and its span falls within the boxed content
            if token_idx >= start_of_targets_idx and \
               offset_char_start >= target_content_char_start_in_full and \
               offset_char_end <= target_content_char_end_in_full:
                answer_mask_sample[token_idx] = True
        
        if answer_mask_sample.sum() == 0:
             self.logger.warning(f"Item {idx}: \\boxed{{...}} found, but no tokens were mapped to answer_mask. Solution: '{generated_solution_str[:200]}...'. Boxed content char span in full text: ({target_content_char_start_in_full}-{target_content_char_end_in_full}). Skipping.")
             return None # Skip if boxed content is empty or not tokenizable within max_length

        return input_ids, labels, loss_mask, answer_mask_sample, think_mask_sample
        
    def prepare_batch(self, batch):
        """
        Process a batch for training.
        Args:
            batch: List of tuples (input_ids, labels, loss_mask, answer_mask, think_mask) 
                   OR list may contain None values if items were skipped.
        Returns:
            Dictionary with batch data, or None if batch becomes empty after filtering.
        """
        # Filter out None items from the batch
        valid_batch = [item for item in batch if item is not None]

        if not valid_batch: # If all items in the batch were skipped
            # self.logger.warning("prepare_batch: All items in the current batch were skipped due to invalid tag structures. Returning None.")
            # Returning None might require the training loop to handle it. 
            # A dictionary with empty tensors might be an alternative if the trainer expects a dict.
            # For simplicity, if the training loop can skip `None` from data_loader, this is fine.
            # Otherwise, an empty dict would be: 
            # return {"input_ids": torch.empty(0), "labels": torch.empty(0), ...}
            return None 

        input_ids, labels, loss_masks, answer_masks, think_masks = zip(*valid_batch)
        
        input_ids_stacked = torch.stack(input_ids)
        labels_stacked = torch.stack(labels)
        loss_mask_stacked = torch.stack(loss_masks)
        answer_mask_stacked = torch.stack(answer_masks)
        think_mask_stacked = torch.stack(think_masks)
        
        return {
            "input_ids": input_ids_stacked,
            "labels": labels_stacked,
            "loss_mask": loss_mask_stacked,
            "answer_mask": answer_mask_stacked,
            "think_mask": think_mask_stacked
        } 