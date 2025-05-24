import os
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from config import NL2SQLConfig
from model import NL2SQLTransformer
from SFT_dataset import SFTDataset
from tokenizer import NL2SQLTokenizer
from relation_matrix import RelationMatrixBuilder
from utils.training import Trainer
from utils.validation import validate_batch, validate_model_outputs, validate_metrics
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model: NL2SQLTransformer, config: NL2SQLConfig, tokenizer: NL2SQLTokenizer, device: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            model: NL2SQL model
            config: Model configuration
            tokenizer: Tokenizer for decoding predictions
            device: Device to evaluate on ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize metrics
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: List of predicted SQL queries
            references: List of reference SQL queries
            
        Returns:
            Dictionary of metric values
        """
        try:
            metrics = {}
            
            # Compute BLEU score
            bleu_scores = []
            for pred, ref in zip(predictions, references):
                try:
                    score = sentence_bleu([ref.split()], pred.split(), smoothing_function=self.smooth)
                    bleu_scores.append(score)
                except Exception as e:
                    logger.warning(f"Error computing BLEU score for a pair: Pred='{pred}', Ref='{ref}'. Error: {e}")
                    continue
            metrics['bleu'] = np.mean(bleu_scores) if bleu_scores else 0.0
            
            # Compute ROUGE scores
            try:
                valid_predictions = [p for p, r in zip(predictions, references) if p and r]
                valid_references = [r for p, r in zip(predictions, references) if p and r]
                if valid_predictions and valid_references:
                    rouge_scores = self.rouge.get_scores(valid_predictions, valid_references, avg=True)
                    metrics.update({
                        'rouge-1': rouge_scores['rouge-1']['f'],
                        'rouge-2': rouge_scores['rouge-2']['f'],
                        'rouge-l': rouge_scores['rouge-l']['f']
                    })
                else:
                    logger.warning("No valid (non-empty) prediction/reference pairs for ROUGE calculation.")
                    metrics.update({'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0})
            except Exception as e:
                logger.warning(f"Error computing ROUGE scores: {e}")
                metrics.update({'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0})
                
            # Validate metrics
            if not validate_metrics(metrics):
                logger.error("Invalid metrics computed")
                return {'bleu': 0.0, 'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {'bleu': 0.0, 'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
            
    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        """
        Evaluate a single batch.
        
        Args:
            batch: Dictionary of tensors from SFTDataset (includes schema_mask if SFT)
            
        Returns:
            Tuple of (predictions, references)
        """
        try:
            if not validate_batch(batch, self.config):
                logger.error("Invalid batch, skipping")
                return [], []
                
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model(
                    encoder_input_ids=batch['encoder_input'],
                    decoder_input_ids=batch['decoder_target'][:, :-1],
                    encoder_relation_ids=batch['relation_matrix'],
                    encoder_attention_mask=batch['encoder_attention_mask'],
                    schema_mask=batch.get('schema_mask')
                )
                
                output_key = 'log_probs' if self.config.use_pointer_generator else 'logits'
                if output_key not in outputs:
                    logger.error(f"Expected key '{output_key}' not found in model outputs. Available keys: {outputs.keys()}")
                    return [], []
                
                output_tensor_for_validation = outputs[output_key]

                if not validate_model_outputs(
                    output_tensor_for_validation,
                    outputs['encoder_output'],
                    batch_size=batch['encoder_input'].size(0),
                    seq_len=batch['decoder_target'][:,:-1].size(1),
                    vocab_size=self.config.vocab_size,
                    d_model=self.config.d_model
                ):
                    logger.error("Invalid model outputs, skipping batch")
                    return [], []
                    
                predictions = torch.argmax(outputs[output_key], dim=-1)
                
            pred_queries = [self.tokenizer.decode(pred.tolist()) for pred in predictions]
            ref_queries = [self.tokenizer.decode(ref[1:].tolist()) for ref in batch['decoder_target']]
            
            return pred_queries, ref_queries
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                logger.error("GPU OOM during evaluation. Skipping batch.")
                return [], []
            else:
                logger.error(f"Runtime error evaluating batch: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error evaluating batch: {e}")
            return [], []
            
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: DataLoader for evaluation (expects SFTDataset format)
            
        Returns:
            Dictionary of metric values
        """
        all_predictions: List[str] = []
        all_references: List[str] = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                predictions, references = self.evaluate_batch(batch)
                all_predictions.extend(predictions)
                all_references.extend(references)
                
            except Exception as e:
                logger.error(f"Critical error in batch {batch_idx}, cannot continue with this batch: {e}")
                continue 
                
        if not all_predictions or not all_references:
            logger.warning("No predictions or references collected for metric computation.")
            return {'bleu': 0.0, 'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}

        metrics = self.compute_metrics(all_predictions, all_references)
        
        logger.info("Evaluation results:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            
        # Free up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return metrics

def evaluate_model(model_checkpoint_path: str, config_path: str):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model_checkpoint_path: Path to the trained model checkpoint (.pt file)
        config_path: Path to the configuration YAML file used for the model.
    """
    try:
        config = NL2SQLConfig.from_yaml(config_path, phase='sft')
        
        tokenizer = NL2SQLTokenizer(config.sp_model_path, config.special_tokens)

        relation_builder = RelationMatrixBuilder(
            tokenizer=tokenizer,
            num_relations=config.num_relations
        )
        
        model = NL2SQLTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            num_relations=config.num_relations,
            dropout=config.dropout,
            max_len=config.max_len,
            use_pointer_generator=config.use_pointer_generator,
            pad_token_id=config.pad_token_id
        )
        
        test_dataset = SFTDataset(
            data_file=config.eval_file,
            tokenizer=tokenizer,
            relation_builder=relation_builder,
            max_len=config.max_len,
            pad_token_id=config.pad_token_id
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=lambda batch: SFTDataset.collate_fn(batch, pad_id=config.pad_token_id)
        )
        
        trainer_for_loading = Trainer(
            model=model,
            config=config, 
            train_dataloader=None,
            val_dataloader=None,
            optimizer=None,
            scheduler=None
        )
        
        logger.info(f"Loading model checkpoint from: {model_checkpoint_path}")
        trainer_for_loading.load_checkpoint(model_checkpoint_path)
        logger.info("Model checkpoint loaded successfully.")
        
        evaluator = Evaluator(model, config, tokenizer)
        
        logger.info("Starting evaluation on the test set...")
        metrics = evaluator.evaluate(test_dataloader)
        
        results_path = os.path.join(config.output_dir, 'evaluation_results.txt')
        os.makedirs(config.output_dir, exist_ok=True)
        with open(results_path, 'w') as f:
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
        logger.info(f"Saved evaluation results to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained NL2SQL model.")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (.pt file) to evaluate.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration YAML file (e.g., src/config.yaml).")
    args = parser.parse_args()

    try:
        evaluate_model(args.model_checkpoint, args.config)
    except Exception as e:
        logger.error(f"Error in main evaluation script: {e}", exc_info=True)

if __name__ == '__main__':
    main() 