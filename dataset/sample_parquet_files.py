import os
import shutil
import random
import argparse
import pathlib

def get_parquet_files(directory):
    """Get a list of all .parquet files in the given directory."""
    return list(pathlib.Path(directory).glob('*.parquet'))

def sample_and_copy_files(source_dir, target_dir, sample_ratio=None, sample_count=None):
    """Samples Parquet files from source_dir and copies them to target_dir."""
    source_path = pathlib.Path(source_dir)
    target_path = pathlib.Path(target_dir)

    if not source_path.is_dir():
        print(f"Error: Source directory '{source_dir}' does not exist or is not a directory.")
        return

    all_parquet_files = get_parquet_files(source_path)

    if not all_parquet_files:
        print(f"No .parquet files found in '{source_dir}'.")
        return

    num_total_files = len(all_parquet_files)
    print(f"Found {num_total_files} Parquet files in '{source_dir}'.")

    if sample_count is not None:
        if sample_count <= 0:
            print("Error: sample_count must be a positive integer.")
            return
        num_to_sample = min(sample_count, num_total_files)
        print(f"Attempting to sample {num_to_sample} files based on sample_count={sample_count}.")
    elif sample_ratio is not None:
        if not (0 < sample_ratio <= 1):
            print("Error: sample_ratio must be between 0 (exclusive) and 1 (inclusive).")
            return
        num_to_sample = int(num_total_files * sample_ratio)
        if num_to_sample == 0 and num_total_files > 0 and sample_ratio > 0:
             num_to_sample = 1 # Ensure at least one file is sampled if ratio is very small but positive
        print(f"Attempting to sample {num_to_sample} files based on sample_ratio={sample_ratio}.")
    else:
        print("Error: Either sample_ratio or sample_count must be provided.")
        return
        
    if num_to_sample == 0:
        print("Calculated number of files to sample is 0. No files will be copied.")
        return

    sampled_files = random.sample(all_parquet_files, num_to_sample)
    print(f"Selected {len(sampled_files)} files for copying.")

    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Target directory '{target_dir}' ensured.")

    copied_count = 0
    for file_path in sampled_files:
        try:
            shutil.copy(file_path, target_path)
            print(f"Copied: {file_path.name} to {target_dir}")
            copied_count += 1
        except Exception as e:
            print(f"Error copying {file_path.name}: {e}")
    
    print(f"\nSuccessfully copied {copied_count} out of {len(sampled_files)} selected files to '{target_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sample Parquet files from a source directory to a target directory.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the source directory containing Parquet files.")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to the target directory where sampled files will be copied.")
    parser.add_argument("--sample_ratio", type=float, default=None, help="Ratio of files to sample (e.g., 0.1 for 10%%). Overridden by sample_count if provided.")
    parser.add_argument("--sample_count", type=int, default=None, help="Exact number of files to sample. Overrides sample_ratio if provided.")

    args = parser.parse_args()

    if args.sample_ratio is None and args.sample_count is None:
        print("Setting default sample_ratio to 0.1 as neither ratio nor count was provided.")
        args.sample_ratio = 0.1

    sample_and_copy_files(args.source_dir, args.target_dir, args.sample_ratio, args.sample_count) 