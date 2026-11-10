"""
Model Factory - Stage 6
Factory functions for creating timm models with custom classifier heads.
"""

from typing import Optional

import torch
import torch.nn as nn
import timm


# Supported model architectures
SUPPORTED_MODELS = {
    'efficientnet_b0': 'efficientnet_b0',
    'resnet50': 'resnet50',
    'convnext_tiny': 'convnext_tiny',
}


def create_model(config: dict) -> nn.Module:
    """
    Create a timm model with custom classifier head for brain tumor classification.
    
    Loads pretrained ImageNet weights and modifies the classifier head to output
    4 classes (glioma, meningioma, notumor, pituitary). Applies dropout and
    prepares the model for training.
    
    Supported models:
        - efficientnet_b0: ~5M params, best accuracy per VRAM
        - resnet50: ~25M params, standard baseline
        - convnext_tiny: ~28M params, modern architecture
    
    Args:
        config: Configuration dictionary with structure:
            {
                'model': {
                    'name': str,           # Model architecture name
                    'pretrained': bool,    # Whether to use pretrained weights
                    'drop_rate': float     # Dropout rate for classifier head
                },
                'data': {
                    'num_classes': int     # Number of output classes
                }
            }
    
    Returns:
        Model ready for training with modified classifier head
        
    Raises:
        ValueError: If model name is not supported
        KeyError: If required config keys are missing
        
    Example:
        >>> config = {
        ...     'model': {'name': 'efficientnet_b0', 'pretrained': True, 'drop_rate': 0.2},
        ...     'data': {'num_classes': 4}
        ... }
        >>> model = create_model(config)
        >>> model = model.to('cuda')
        >>> # Optionally convert to channels_last
        >>> model = model.to(memory_format=torch.channels_last)
    """
    # Extract config parameters
    try:
        model_name = config['model']['name']
        pretrained = config['model']['pretrained']
        drop_rate = config['model'].get('drop_rate', 0.0)  # Optional, default to 0.0
        num_classes = config['data']['num_classes']
    except KeyError as e:
        raise KeyError(f"Missing required config key: {e}")
    
    # Validate model name
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models: {list(SUPPORTED_MODELS.keys())}"
        )
    
    # Map to timm model name
    timm_model_name = SUPPORTED_MODELS[model_name]
    
    # Create model with timm
    # For some architectures (like ConvNeXt), we need to create with num_classes directly
    # For others, we can use num_classes=0 and add a custom head
    
    # Special handling for ConvNeXt which doesn't work well with num_classes=0
    if 'convnext' in model_name.lower():
        # Create with final num_classes directly
        model = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=config['model'].get('drop_path_rate', 0.0)
        )
    else:
        # For other models, create without classifier and add custom head
        model = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove the classifier head
            drop_rate=drop_rate
        )
        
        # Get the number of features from the backbone
        num_features = model.num_features
        
        # Create custom classifier head
        classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Attach classifier to model
        # Different architectures use different attribute names for the classifier
        if hasattr(model, 'fc'):
            # ResNet uses 'fc'
            model.fc = classifier
        elif hasattr(model, 'classifier'):
            # EfficientNet and others use 'classifier'
            model.classifier = classifier
        elif hasattr(model, 'head'):
            # ConvNeXt and ViT use 'head'
            model.head = classifier
        else:
            # Fallback: try to add as 'head'
            model.head = classifier
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information:
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
            - model_size_mb: Approximate model size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Approximate size in MB (assuming float32)
    model_size_mb = total_params * 4 / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': round(model_size_mb, 2)
    }


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters except the classifier head.
    
    Used for Stage 1 training (head-only).
    
    Args:
        model: Model to freeze
        
    Returns:
        Model with frozen backbone
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier head
    # Check different possible attribute names
    if hasattr(model, 'fc') and model.fc is not None:
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier') and model.classifier is not None:
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'head') and model.head is not None:
        for param in model.head.parameters():
            param.requires_grad = True
    
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """
    Unfreeze all model parameters.
    
    Used for Stage 2 training (full fine-tuning).
    
    Args:
        model: Model to unfreeze
        
    Returns:
        Model with all parameters trainable
    """
    for param in model.parameters():
        param.requires_grad = True
    
    return model


def convert_to_channels_last(model: nn.Module) -> nn.Module:
    """
    Convert model to channels_last memory format for better performance.
    
    Should be called after moving model to GPU.
    
    Args:
        model: Model to convert
        
    Returns:
        Model in channels_last format
        
    Example:
        >>> model = create_model(config)
        >>> model = model.to('cuda')
        >>> model = convert_to_channels_last(model)
    """
    return model.to(memory_format=torch.channels_last)


def prepare_model_for_training(
    config: dict,
    device: str = 'cuda',
    stage: int = 1
) -> nn.Module:
    """
    Create and prepare a model for training.
    
    Convenience function that creates the model, moves it to device,
    optionally converts to channels_last, and freezes the backbone for stage 1.
    
    Args:
        config: Model configuration
        device: Device to move model to ('cuda' or 'cpu')
        stage: Training stage (1 = head only, 2 = full fine-tune)
        
    Returns:
        Model ready for training
        
    Example:
        >>> # Stage 1: head-only training
        >>> model = prepare_model_for_training(config, device='cuda', stage=1)
        >>> 
        >>> # Stage 2: full fine-tuning
        >>> model = prepare_model_for_training(config, device='cuda', stage=2)
    """
    # Create model
    model = create_model(config)
    
    # Move to device
    model = model.to(device)
    
    # Convert to channels_last if configured
    if config.get('training', {}).get('channels_last', False):
        model = convert_to_channels_last(model)
    
    # Freeze backbone for stage 1
    if stage == 1:
        model = freeze_backbone(model)
    else:
        model = unfreeze_all(model)
    
    return model
