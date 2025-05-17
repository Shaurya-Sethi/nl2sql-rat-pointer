import os
import logging
import torch
from pathlib import Path
from model import NL2SQLTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TokenizedTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, max_len=256):
        self.samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                ids = [int(tok) for tok in line.strip().split()]
                if len(ids) > max_len:
                    ids = ids[:max_len]
                self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = self.samples[idx]
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

def collate_fn(batch, pad_id=18):
    max_len = max(len(item['input_ids']) for item in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
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

def test_setup():
    try:
        project_root = Path(__file__).parent.parent
        data_file = project_root / "datasets" / "paired_nl_sql" / "splits" / "tokenized_sft_filtered_train.txt"
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return False

        # Minimal model config for smoke test
        vocab_size = 32000
        d_model = 128
        n_heads = 4
        n_layers = 2
        num_relations = 5
        dropout = 0.1
        max_len = 256

        model = NL2SQLTransformer(
            vocab_size=vocab_size,
            num_relations=num_relations,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len
        )
        logger.info("Model initialized successfully")

        dataset = TokenizedTextDataset(str(data_file), max_len=max_len)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(dataloader))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        logger.info("Running forward pass...")
        # For this test, use input_ids as both encoder and decoder input
        outputs = model(
            encoder_input_ids=batch['input_ids'],
            decoder_input_ids=batch['input_ids'],
            encoder_relation_ids=torch.zeros(batch['input_ids'].shape[0], batch['input_ids'].shape[1], batch['input_ids'].shape[1], dtype=torch.long, device=device),
            encoder_attention_mask=None,
            decoder_attention_mask=None,
            decoder_key_padding_mask=None
        )
        logger.info("Forward pass successful!")
        logger.info(f"Logits shape: {outputs['logits'].shape}")
        logger.info(f"Encoder output shape: {outputs['encoder_output'].shape}")
        return True
    except Exception as e:
        logger.error(f"Smoke test failed: {str(e)}")
        return False

if __name__ == '__main__':
    test_setup() 