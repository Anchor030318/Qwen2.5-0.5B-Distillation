import os
import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import random
from Util.utils import setup_logger
import pyarrow.parquet as pq # Added for Parquet metadata
import pandas as pd # Ensure pandas is available for loading data in getitem

class SFTDataset(Dataset):
    """
    Dataset for SFT training. Loads data containing a problem and a generated solution 
    (expected to be in a format like "Question: <problem>\n<think><CoT></think><answer><answer_text></answer>").
    Implements lazy loading for Parquet files in a directory.
    Generates a single loss mask for distillation targets.
    """
    
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer, max_length=1024, shuffle=True, log_dir=None):
        self.logger = setup_logger("SFTDataset", log_dir=log_dir)
        self.logger.info(f"Initializing SFTDataset with data from {data_path}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.file_paths = []
        self.num_rows_per_file = []
        self.cumulative_rows = [0] # Starts with 0 for easier index mapping
        self._total_samples = 0
        self.data_type = None # Will be set based on data_path type

        # Cache for lazy loading
        self.cache_file_path = None
        self.cache_df = None

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
        item = self._get_item_content(idx) # Get raw data
        
        question_str = item.get('problem', '')
        generated_solution_str = item.get('generated_solution', '') 

        prompt_text = f"Question: {question_str}\n"
        full_text = f"{prompt_text}{generated_solution_str}"

        tokenized_output = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True
        )

        input_ids = torch.tensor(tokenized_output["input_ids"])
        attention_mask = torch.tensor(tokenized_output["attention_mask"])
        labels = input_ids.clone()
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        
        prompt_tokens_for_length_calc = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        len_prompt_tokens = len(prompt_tokens_for_length_calc)

        start_of_targets_idx = 0
        if self.tokenizer.bos_token_id is not None and input_ids.numel() > 0 and input_ids[0] == self.tokenizer.bos_token_id:
            start_of_targets_idx = 1 
        start_of_targets_idx += len_prompt_tokens

        if input_ids.numel() > start_of_targets_idx: # Ensure there are tokens beyond the prompt
            loss_mask[start_of_targets_idx:] = 1
        
        loss_mask = loss_mask * attention_mask
        
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