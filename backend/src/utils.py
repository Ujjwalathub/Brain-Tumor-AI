"""
Utility functions for brain tumor MRI classification.

Provides config loading with inheritance, metrics computation,
logging setup, reproducibility helpers, and formatting utilities.
"""

import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.tensorboard import SummaryWriter

# Scripts add `src/` to sys.path and import this module as `utils`; package
# imports (tests, API) use `src.utils`. Support both import styles.
try:  # pragma: no cover - import plumbing
    from .config import configure_utf8_console, load_config
except ImportError:  # pragma: no cover
    from config import configure_utf8_console, load_config


__all__ = [
    "configure_utf8_console",
    "load_config",
    "compute_metrics",
    "setup_logging",
    "set_seed",
    "format_time",
    "get_lr",
]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_names: list[str],
) -> dict:
    """
    Compute comprehensive classification metrics.
    
    Computes accuracy, per-class precision/recall/F1, macro-averaged
    precision/recall/F1, and the confusion matrix using scikit-learn.
    
    Args:
        y_true: True labels, shape (N,)
        y_pred: Predicted labels, shape (N,)
        y_proba: Predicted probabilities, shape (N, num_classes) or None
        class_names: List of class names in order (e.g., ['glioma', 'meningioma', ...])
        
    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - macro_precision: Macro-averaged precision (equal weight per class)
            - macro_recall: Macro-averaged recall (equal weight per class)
            - macro_f1: Macro-averaged F1 (equal weight per class)
            - per_class: Dict with precision, recall, f1 per class
            - confusion_matrix: Confusion matrix as numpy array
            
    Example:
        >>> metrics = compute_metrics(y_true, y_pred, y_proba, ['glioma', 'meningioma', 'notumor', 'pituitary'])
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        >>> print(f"Macro Recall: {metrics['macro_recall']:.3f}")
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class precision, recall, F1. `labels` keeps the arrays aligned with
    # `class_names` even when a split happens to be missing a class.
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # Macro-averaged metrics (equal weight per class)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Build per-class metrics dict
    per_class = {}
    for i, class_name in enumerate(class_names):
        per_class[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        }
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class': per_class,
        'confusion_matrix': cm,
    }
    
    return metrics


def setup_logging(log_dir: str, experiment_name: str) -> SummaryWriter:
    """
    Setup TensorBoard logging.
    
    Creates a SummaryWriter for the given experiment in the specified log directory.
    Creates the log directory if it doesn't exist.
    
    Args:
        log_dir: Base directory for TensorBoard logs (e.g., 'runs')
        experiment_name: Name of the experiment (e.g., 'efficientnet_b0_baseline')
        
    Returns:
        TensorBoard SummaryWriter instance
        
    Example:
        >>> writer = setup_logging('runs', 'efficientnet_b0_baseline')
        >>> writer.add_scalar('Loss/train', 0.5, step=100)
        >>> writer.close()
    """
    log_path = Path(log_dir) / experiment_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_path))
    
    return writer


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for Python's random module, NumPy, and PyTorch (CPU and CUDA).
    Also configures PyTorch for deterministic operations where possible.
    
    Args:
        seed: Random seed value (e.g., 42)
        
    Note:
        Some PyTorch operations have no deterministic implementation.
        This function enables deterministic mode but full reproducibility
        is not guaranteed, especially with CUDA operations.
        
    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
    # Configure PyTorch for determinism (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """
    Format elapsed time as HH:MM:SS.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
        
    Example:
        >>> format_time(3661.5)
        '01:01:01'
        >>> format_time(125.3)
        '00:02:05'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Returns the learning rate from the first parameter group.
    Useful for logging during training.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
        
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> current_lr = get_lr(optimizer)
        >>> print(f"LR: {current_lr:.6f}")
    """
    return optimizer.param_groups[0]['lr']
