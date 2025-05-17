import os
import random
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_file(input_file: str, train_ratio: float = 0.9, seed: int = 42):
    """
    Split a file into train and validation sets.
    
    Args:
        input_file: Path to input file
        train_ratio: Ratio of data to use for training (default: 0.9)
        seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    
    # Create output directory
    input_path = Path(input_file)
    output_dir = input_path.parent / "splits"
    output_dir.mkdir(exist_ok=True)
    
    # Read all lines
    logger.info(f"Reading {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle lines
    random.shuffle(lines)
    
    # Split into train and val
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    # Write train file
    train_file = output_dir / f"{input_path.stem}_train{input_path.suffix}"
    logger.info(f"Writing {len(train_lines)} lines to {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # Write val file
    val_file = output_dir / f"{input_path.stem}_val{input_path.suffix}"
    logger.info(f"Writing {len(val_lines)} lines to {val_file}")
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    return str(train_file), str(val_file)

def main():
    parser = argparse.ArgumentParser(description='Split data files into train and validation sets')
    parser.add_argument('--pretrain_file', type=str, required=True,
                      help='Path to pretraining data file')
    parser.add_argument('--sft_file', type=str, required=True,
                      help='Path to SFT data file')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                      help='Ratio of data to use for training (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Split pretraining data
    logger.info("Splitting pretraining data...")
    pretrain_train, pretrain_val = split_file(
        args.pretrain_file,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    # Split SFT data
    logger.info("Splitting SFT data...")
    sft_train, sft_val = split_file(
        args.sft_file,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    logger.info("Done! Here are the output files:")
    logger.info(f"Pretraining - Train: {pretrain_train}")
    logger.info(f"Pretraining - Val: {pretrain_val}")
    logger.info(f"SFT - Train: {sft_train}")
    logger.info(f"SFT - Val: {sft_val}")

if __name__ == '__main__':
    main() 