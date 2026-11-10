"""
Brain Tumor Dataset - Stage 4
Custom PyTorch Dataset for loading preprocessed MRI images.
"""

from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Class name to index mapping
CLASS_TO_IDX = {
    'glioma': 0,
    'meningioma': 1,
    'notumor': 2,
    'pituitary': 3
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class BrainTumorDataset(Dataset):
    """
    Custom Dataset for brain tumor MRI images.
    
    Loads preprocessed images from cached numpy arrays and applies transforms.
    """
    
    def __init__(
        self,
        data_dir: Path,
        manifest_df: pd.DataFrame,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing preprocessed data
            manifest_df: DataFrame with columns ['filepath', 'class', 'split']
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.manifest = manifest_df.reset_index(drop=True)
        self.transform = transform
        
        # Validate manifest columns
        required_cols = ['filepath', 'class', 'split']
        if not all(col in self.manifest.columns for col in required_cols):
            raise ValueError(f"Manifest must contain columns: {required_cols}")
        
        # Validate classes
        unique_classes = set(self.manifest['class'].unique())
        expected_classes = set(CLASS_TO_IDX.keys())
        if not unique_classes.issubset(expected_classes):
            raise ValueError(
                f"Unknown classes in manifest: {unique_classes - expected_classes}"
            )
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Index of sample to retrieve
            
        Returns:
            Tuple of (image_tensor, label)
            
        Raises:
            FileNotFoundError: If the .npy file doesn't exist
            ValueError: If image cannot be loaded
        """
        # Get metadata
        row = self.manifest.iloc[idx]
        filepath = self.data_dir / row['filepath']
        class_name = row['class']
        label = CLASS_TO_IDX[class_name]
        
        # Load numpy array
        if not filepath.exists():
            raise FileNotFoundError(f"Image file not found: {filepath}")
        
        try:
            image = np.load(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load {filepath}: {e}")
        
        # Validate image shape
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected image shape (H, W, 3), got {image.shape} for {filepath}"
            )
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def get_class_counts(self) -> dict:
        """
        Get count of samples per class.
        
        Returns:
            Dictionary mapping class names to counts
        """
        return self.manifest['class'].value_counts().to_dict()
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for balanced training.
        
        Uses inverse frequency weighting.
        
        Returns:
            Tensor of shape (num_classes,) with weights
        """
        class_counts = self.manifest['class'].value_counts()
        total = len(self.manifest)
        num_classes = len(CLASS_TO_IDX)
        
        # Inverse frequency
        weights = torch.zeros(num_classes)
        for class_name, count in class_counts.items():
            idx = CLASS_TO_IDX[class_name]
            weights[idx] = total / (num_classes * count)
        
        return weights
    
    def get_sample_info(self, idx: int) -> dict:
        """
        Get metadata for a sample without loading the image.
        
        Args:
            idx: Index of sample
            
        Returns:
            Dictionary with sample metadata
        """
        row = self.manifest.iloc[idx]
        return {
            'filepath': str(self.data_dir / row['filepath']),
            'class': row['class'],
            'label': CLASS_TO_IDX[row['class']],
            'split': row['split']
        }


def create_dataloaders(
    manifest_path: Path,
    processed_dir: Path,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders from manifest.
    
    Args:
        manifest_path: Path to manifest CSV file
        processed_dir: Root directory of preprocessed data
        train_transform: Transform for training data
        val_transform: Transform for validation data
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load manifest
    manifest = pd.read_csv(manifest_path)
    
    # Split into train and val
    train_df = manifest[manifest['split'] == 'train'].copy()
    val_df = manifest[manifest['split'] == 'val'].copy()
    
    # Create datasets
    train_dataset = BrainTumorDataset(
        data_dir=processed_dir,
        manifest_df=train_df,
        transform=train_transform
    )
    
    val_dataset = BrainTumorDataset(
        data_dir=processed_dir,
        manifest_df=val_df,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader
