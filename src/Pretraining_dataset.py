import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class PretrainingDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len=512):
        """
        Initialize the pretraining dataset.
        
        Args:
            data_file (str): Path to the data file
            tokenizer: Tokenizer instance
            max_len (int): Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load data
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.examples = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(example)
        
        # Truncate if necessary
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        
        # Create input_ids and attention_mask
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For language modeling, labels are same as input
        }

    @staticmethod
    def collate_fn(batch, pad_id=18):
        """
        Collate function for the dataloader.
        
        Args:
            batch: List of dictionaries containing input_ids, attention_mask, and labels
            pad_id (int): Padding ID
            
        Returns:
            Dictionary of batched tensors
        """
        # Get max length in this batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        # Initialize tensors with pad_id
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        
        # Fill tensors
        for i, item in enumerate(batch):
            length = len(item['input_ids'])
            input_ids[i, :length] = item['input_ids']
            attention_mask[i, :length] = item['attention_mask']
            labels[i, :length] = item['labels']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
