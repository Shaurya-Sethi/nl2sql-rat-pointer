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
        # Split the string of token IDs and convert to integers
        tokens = [int(token_id) for token_id in example.split()]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        seq_len = input_ids.size(0)
        # Dummies for keys expected by the Trainer
        relation_matrix = torch.zeros((seq_len, seq_len), dtype=torch.long)
        schema_mask = torch.zeros(seq_len, dtype=torch.bool)
        return {
            'encoder_input': input_ids,                   # what the Trainer expects
            'encoder_attention_mask': attention_mask,     # what the Trainer expects
            'decoder_target': input_ids.clone(),          # labels = inputs for LM pretraining
            'relation_matrix': relation_matrix,           # dummy, not used in pretrain
            'schema_mask': schema_mask,                   # dummy, not used in pretrain
        }

    @staticmethod
    def collate_fn(batch, pad_id=18):
        """
        Collate function for the dataloader.
        Returns batch dict with all keys Trainer expects, padded.
        """
        # Get max length in this batch
        max_len = max(len(item['encoder_input']) for item in batch)
        batch_size = len(batch)

        encoder_input = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        encoder_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        decoder_target = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        relation_matrix = torch.zeros((batch_size, max_len, max_len), dtype=torch.long)
        schema_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for i, item in enumerate(batch):
            length = len(item['encoder_input'])
            encoder_input[i, :length] = item['encoder_input']
            encoder_attention_mask[i, :length] = item['encoder_attention_mask']
            decoder_target[i, :length] = item['decoder_target']
            relation_matrix[i, :length, :length] = item['relation_matrix']
            schema_mask[i, :length] = item['schema_mask']

        return {
            'encoder_input': encoder_input,
            'encoder_attention_mask': encoder_attention_mask,
            'decoder_target': decoder_target,
            'relation_matrix': relation_matrix,
            'schema_mask': schema_mask,
        }
