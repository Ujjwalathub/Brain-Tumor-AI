"""Utility functions for data processing."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for determinism (may impact performance)
    torch.backends.cudnn.deterministic = False  # Keep False for speed
    torch.backends.cudnn.benchmark = True  # Enable for fixed input size


def get_class_to_idx(class_names: list) -> dict:
    """Create class name to index mapping."""
    return {cls_name: idx for idx, cls_name in enumerate(sorted(class_names))}


def count_images_in_dir(directory: str) -> dict:
    """
    Count images per class in a directory.
    
    Args:
        directory: Path to directory with class subdirectories
        
    Returns:
        Dictionary mapping class names to counts
    """
    counts = {}
    
    if not os.path.exists(directory):
        return counts
    
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            counts[class_name] = len(image_files)
    
    return counts


def ensure_dir(directory: str):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
