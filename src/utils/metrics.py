import torch
import numpy as np
from typing import Dict, Any

def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        preds: Model predictions
        labels: Ground truth labels
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert to numpy for easier computation
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Compute accuracy
    accuracy = np.mean(preds == labels)
    
    # Compute loss (cross entropy)
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(preds),
        torch.tensor(labels),
        reduction='mean'
    ).item()
    
    return {
        'accuracy': accuracy,
        'loss': loss
    } 