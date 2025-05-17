import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from bitsandbytes.optim import AdamW8bit
from config import NL2SQLConfig
from model import NL2SQLTransformer
from tokenizer import NL2SQLTokenizer
from relation_matrix import RelationMatrixBuilder
from utils.training import Trainer
from utils.metrics import compute_metrics
from Pretraining_dataset import PretrainingDataset
from SFT_dataset import SFTDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train NL2SQL model')
    parser.add_argument('--phase', type=str, required=True, choices=['pretrain', 'sft'],
                      help='Training phase: pretrain or sft')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--pretrained_model', type=str,
                      help='Path to pretrained model for SFT phase')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--use_8bit_optimizer', action='store_true',
                      help='Use 8-bit optimizer')
    parser.add_argument('--use_cosine_schedule', action='store_true',
                      help='Use cosine learning rate schedule')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load config
    config = load_config(args.config)
    phase_config = config[args.phase]

    # Initialize tokenizer
    tokenizer = NL2SQLTokenizer(config['paths']['sp_model'])
    if not tokenizer.sp_model:
        logger.error("Failed to load tokenizer model")
        sys.exit(1)

    # Initialize relation matrix builder
    relation_builder = RelationMatrixBuilder(config['model']['num_relations'])

    # Initialize model
    if args.phase == 'sft' and args.pretrained_model:
        logger.info(f"Loading pretrained model from {args.pretrained_model}")
        model = torch.load(args.pretrained_model)
    else:
        model = NL2SQLTransformer(
            vocab_size=config['model']['vocab_size'],
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_layers=config['model']['n_layers'],
            num_relations=config['model']['num_relations'],
            dropout=config['model']['dropout'],
            max_len=config['model']['max_len']
        )

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create datasets
    if args.phase == 'pretrain':
        train_dataset = PretrainingDataset(
            phase_config['train_file'],
            tokenizer,
            max_len=config['model']['max_len']
        )
        eval_dataset = PretrainingDataset(
            phase_config['eval_file'],
            tokenizer,
            max_len=config['model']['max_len']
        )
    else:  # sft
        train_dataset = SFTDataset(
            phase_config['train_file'],
            tokenizer,
            relation_builder,
            max_len=config['model']['max_len']
        )
        eval_dataset = SFTDataset(
            phase_config['eval_file'],
            tokenizer,
            relation_builder,
            max_len=config['model']['max_len']
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=phase_config['micro_batch_size'],
        shuffle=True,
        num_workers=phase_config['num_workers']
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=phase_config['micro_batch_size'],
        shuffle=False,
        num_workers=phase_config['num_workers']
    )

    # Create optimizer
    if args.use_8bit_optimizer:
        optimizer = AdamW8bit(
            model.parameters(),
            lr=phase_config['learning_rate'],
            weight_decay=phase_config['weight_decay']
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=phase_config['learning_rate'],
            weight_decay=phase_config['weight_decay']
        )

    # Create learning rate scheduler
    if args.use_cosine_schedule:
        total_steps = len(train_loader) * phase_config['epochs']
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=phase_config['warmup_steps'],
            num_training_steps=total_steps
        )
    else:
        scheduler = None

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=phase_config,
        compute_metrics=compute_metrics
    )

    # Train model
    logger.info(f"Starting {args.phase} training...")
    trainer.train()

    # Save final model
    output_dir = Path(config['paths']['output_dir']) / args.phase
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_dir / 'final_model.pt')
    logger.info(f"Training complete. Model saved to {output_dir / 'final_model.pt'}")

if __name__ == '__main__':
    main() 