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
from utils.training import Trainer
from utils.validation import validate_batch, validate_model_outputs, validate_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model: NL2SQLTransformer, config: NL2SQLConfig, device: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            model: NL2SQL model
            config: Model configuration
            device: Device to evaluate on ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config
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
                    logger.warning(f"Error computing BLEU score: {e}")
                    continue
            metrics['bleu'] = np.mean(bleu_scores) if bleu_scores else 0.0
            
            # Compute ROUGE scores
            try:
                rouge_scores = self.rouge.get_scores(predictions, references, avg=True)
                metrics.update({
                    'rouge-1': rouge_scores['rouge-1']['f'],
                    'rouge-2': rouge_scores['rouge-2']['f'],
                    'rouge-l': rouge_scores['rouge-l']['f']
                })
            except Exception as e:
                logger.warning(f"Error computing ROUGE scores: {e}")
                metrics.update({
                    'rouge-1': 0.0,
                    'rouge-2': 0.0,
                    'rouge-l': 0.0
                })
                
            # Validate metrics
            if not validate_metrics(metrics):
                logger.error("Invalid metrics computed")
                return {}
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {}
            
    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        """
        Evaluate a single batch.
        
        Args:
            batch: Dictionary of tensors
            
        Returns:
            Tuple of (predictions, references)
        """
        try:
            # Validate batch
            if not validate_batch(batch, self.config):
                logger.error("Invalid batch, skipping")
                return [], []
                
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model(
                    encoder_input=batch['encoder_input'],
                    decoder_input=batch['decoder_target'][:, :-1],
                    relation_matrix=batch['relation_matrix'],
                    attention_mask=batch['encoder_attention_mask']
                )
                
                # Validate outputs
                if not validate_model_outputs(
                    outputs['logits'],
                    outputs['encoder_output'],
                    batch['encoder_input'].size(0),
                    batch['encoder_input'].size(1),
                    self.config.vocab_size,
                    self.config.d_model
                ):
                    logger.error("Invalid model outputs, skipping batch")
                    return [], []
                    
                # Get predictions
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
            # Convert to strings
            pred_queries = [self.config.tokenizer.decode(pred) for pred in predictions]
            ref_queries = [self.config.tokenizer.decode(ref) for ref in batch['decoder_target']]
            
            return pred_queries, ref_queries
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                logger.error("GPU OOM during evaluation. Skipping batch.")
                return [], []
            else:
                raise e
                
        except Exception as e:
            logger.error(f"Error evaluating batch: {e}")
            return [], []
            
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            Dictionary of metric values
        """
        all_predictions = []
        all_references = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                predictions, references = self.evaluate_batch(batch)
                all_predictions.extend(predictions)
                all_references.extend(references)
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
                
        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_references)
        
        # Log results
        logger.info("Evaluation results:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            
        return metrics

def evaluate_model(model_path: str, config: NL2SQLConfig):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model_path: Path to the trained model checkpoint
        config: Model configuration
    """
    try:
        # Initialize model
        model = NL2SQLTransformer(config)
        
        # Create test dataset and dataloader
        test_dataset = SFTDataset(config, split='test')
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=SFTDataset.collate_fn
        )
        
        # Initialize trainer and evaluator
        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=None,
            val_dataloader=test_dataloader
        )
        evaluator = Evaluator(model, config)
        
        # Load checkpoint
        trainer.load_checkpoint(model_path)
        
        # Evaluate
        logger.info("Starting evaluation...")
        metrics = evaluator.evaluate(test_dataloader)
        
        # Save results
        results_path = os.path.join(config.output_dir, 'evaluation_results.txt')
        with open(results_path, 'w') as f:
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
        logger.info(f"Saved evaluation results to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

def main():
    try:
        # Load configuration
        config = NL2SQLConfig()
        
        # Get path to best model
        model_path = os.path.join(config.output_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            logger.error(f"Model checkpoint not found at {model_path}")
            return
            
        # Evaluate model
        evaluate_model(model_path, config)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main() 