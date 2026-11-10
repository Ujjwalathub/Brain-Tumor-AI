"""
Image Transforms - Stage 4
PyTorch transforms v2 for training and validation.
"""

from typing import Any, Dict
import torch
from torchvision import transforms
from torchvision.transforms import v2


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(config: Dict[str, Any] = None) -> v2.Compose:
    """
    Get training transforms with data augmentation.
    
    Args:
        config: Optional configuration dictionary with augmentation parameters.
                Expected keys:
                - rotation_degrees: Maximum rotation angle (default: 15)
                - h_flip_prob / horizontal_flip_prob: Horizontal flip probability (default: 0.5)
                - translate: Max random translation fraction per axis (default: None).
                    When provided, RandomAffine(rotation + translate) replaces the
                    bare RandomRotation. Recommended for brain MRI source-shift
                    robustness (Stage 9 gate finding: glioma sensitive to
                    vertical crop offsets).
                - brightness: ColorJitter brightness factor (default: 0.2)
                - contrast: ColorJitter contrast factor (default: 0.2)
                
    Returns:
        Composed transform pipeline for training
    """
    if config is None:
        config = {}
    
    # Get parameters with defaults
    rotation_degrees = config.get('rotation_degrees', 15)
    h_flip_prob = config.get(
        'h_flip_prob',
        config.get('horizontal_flip_prob', 0.5)
    )
    translate = config.get('translate', None)
    brightness = config.get('brightness', 0.2)
    contrast = config.get('contrast', 0.2)
    
    # Spatial augmentation: bare rotation, or rotation + random translation
    if translate is not None:
        spatial = v2.RandomAffine(
            degrees=rotation_degrees,
            translate=(float(translate[0]), float(translate[1])),
            interpolation=v2.InterpolationMode.BILINEAR,
        )
    else:
        spatial = v2.RandomRotation(degrees=rotation_degrees)
    
    transform = v2.Compose([
        # Convert numpy uint8 array (H, W, C) to a uint8 image tensor (C, H, W)
        v2.ToImage(),
        
        # Data augmentation
        spatial,
        v2.RandomHorizontalFlip(p=h_flip_prob),
        v2.ColorJitter(brightness=brightness, contrast=contrast),
        
        # Convert to float and normalize
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    return transform


def get_val_transforms(config: Dict[str, Any] = None) -> v2.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        config: Optional configuration dictionary (unused, for API consistency)
        
    Returns:
        Composed transform pipeline for validation/testing
    """
    transform = v2.Compose([
        # Convert numpy uint8 array (H, W, C) to a uint8 image tensor (C, H, W)
        v2.ToImage(),
        
        # Convert to float and normalize (no augmentation)
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    return transform


def get_transforms(mode: str = 'train', config: Dict[str, Any] = None) -> v2.Compose:
    """
    Get transforms for specified mode.
    
    Args:
        mode: One of 'train', 'val', or 'test'
        config: Optional configuration dictionary
        
    Returns:
        Appropriate transform pipeline
        
    Raises:
        ValueError: If mode is not recognized
    """
    if mode == 'train':
        return get_train_transforms(config)
    elif mode in ['val', 'test']:
        return get_val_transforms(config)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'train', 'val', or 'test'")


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor using ImageNet stats for visualization.
    
    Args:
        tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W)
        
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    
    # Handle batch dimension
    if tensor.ndim == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)
