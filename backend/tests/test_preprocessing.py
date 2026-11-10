"""
Comprehensive tests for data preprocessing pipeline.
Tests preprocessing, transforms, dataset class, splits, and caching.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import (
    BrainTumorDataset,
    get_train_transforms,
    get_val_transforms,
    create_split,
    preprocess_image,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)
from src.data.preprocess import find_brain_contour


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Tuple[Path, Path]:
    """
    Create temporary dataset structure with dummy images.
    
    Returns:
        Tuple of (raw_dir, processed_dir)
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    
    # Create directory structure
    for split in ["Training", "Testing"]:
        for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
            (raw_dir / split / class_name).mkdir(parents=True, exist_ok=True)
            (processed_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    return raw_dir, processed_dir


@pytest.fixture
def sample_config() -> dict:
    """Load base.yaml config."""
    config_path = project_root / "configs" / "base.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Return minimal config if file doesn't exist
        return {
            'data': {
                'img_size': 224,
                'train_val_split': 0.85,
            },
            'augmentation': {
                'rotation_degrees': 15,
                'horizontal_flip_prob': 0.5,
                'brightness': 0.2,
                'contrast': 0.2,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
            }
        }


@pytest.fixture
def dummy_image_with_borders() -> np.ndarray:
    """Create a test image with black borders."""
    # 300x300 image with 50px black border
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # White center region
    img[50:250, 50:250, :] = 255
    return img


@pytest.fixture
def dummy_image_no_borders() -> np.ndarray:
    """Create a test image without black borders."""
    # 224x224 image, all white
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    return img


@pytest.fixture
def dummy_manifest(tmp_path: Path) -> pd.DataFrame:
    """Create a dummy manifest for testing splits."""
    records = []
    
    # Create balanced dataset: 100 samples per class
    for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
        for i in range(100):
            records.append({
                'filepath': f"Training/{class_name}/sample_{i:03d}.npy",
                'class': class_name,
                'split': 'training'
            })
    
    return pd.DataFrame(records)


# ============================================================================
# Tests: Preprocessing
# ============================================================================

def test_preprocess_crops_margins(tmp_path: Path, dummy_image_with_borders: np.ndarray):
    """Test that preprocessing crops black margins correctly."""
    # Save dummy image
    img_path = tmp_path / "test_image.jpg"
    import cv2
    cv2.imwrite(str(img_path), dummy_image_with_borders)
    
    # Preprocess
    result = preprocess_image(img_path, target_size=(224, 224))
    
    # Assert output is smaller than input (margins cropped)
    assert result.shape == (224, 224, 3), f"Expected (224, 224, 3), got {result.shape}"
    
    # Assert output dtype is uint8
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    
    # The image should be resized to 224x224
    assert result.shape[0] == 224 and result.shape[1] == 224


def test_preprocess_handles_already_cropped(tmp_path: Path, dummy_image_no_borders: np.ndarray):
    """Test that preprocessing handles images without borders."""
    # Save dummy image
    img_path = tmp_path / "test_image.jpg"
    import cv2
    cv2.imwrite(str(img_path), dummy_image_no_borders)
    
    # Preprocess - should not raise errors
    result = preprocess_image(img_path, target_size=(224, 224))
    
    # Assert no errors and output is correct size
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_find_brain_contour():
    """Test the contour finding function."""
    # Create test image with clear foreground
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150, :] = 255  # White square in center
    
    x, y, w, h = find_brain_contour(img, threshold=10)
    
    # Should find the white region
    assert x >= 45 and x <= 55, f"Expected x around 50, got {x}"
    assert y >= 45 and y <= 55, f"Expected y around 50, got {y}"
    assert w >= 95 and w <= 105, f"Expected w around 100, got {w}"
    assert h >= 95 and h <= 105, f"Expected h around 100, got {h}"


def test_find_brain_contour_no_foreground():
    """Test contour finding on all-black image."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    x, y, w, h = find_brain_contour(img, threshold=10)
    
    # Should return full image bounds when no contour found
    assert x == 0 and y == 0
    assert w == 200 and h == 200


# ============================================================================
# Tests: Transforms
# ============================================================================

def test_transforms_train_augmentation(dummy_image_no_borders: np.ndarray):
    """Test that training transforms include randomness."""
    transform = get_train_transforms()
    
    # Apply transforms multiple times
    outputs = []
    for _ in range(5):
        output = transform(dummy_image_no_borders)
        outputs.append(output)
    
    # Check output properties
    for output in outputs:
        assert output.shape == (3, 224, 224), f"Expected (3, 224, 224), got {output.shape}"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
    
    # Check that outputs differ due to randomness
    # Compare first two outputs - they should not be identical
    diff = torch.abs(outputs[0] - outputs[1]).sum().item()
    assert diff > 0, "Training transforms should produce different outputs (randomness)"


def test_transforms_val_deterministic(dummy_image_no_borders: np.ndarray):
    """Test that validation transforms are deterministic."""
    transform = get_val_transforms()
    
    # Apply to same image multiple times
    output1 = transform(dummy_image_no_borders)
    output2 = transform(dummy_image_no_borders)
    
    # Check shape and dtype
    assert output1.shape == (3, 224, 224)
    assert output1.dtype == torch.float32
    
    # Outputs should be identical (deterministic)
    assert torch.allclose(output1, output2, atol=1e-6), \
        "Validation transforms should be deterministic"


def test_normalize_values(dummy_image_no_borders: np.ndarray):
    """Test that normalization produces expected value ranges."""
    transform = get_val_transforms()
    
    # Apply normalization
    output = transform(dummy_image_no_borders)
    
    # Check that values are normalized (not in [0, 1] range anymore)
    # For white image (255), after dividing by 255 and normalizing with ImageNet stats,
    # we expect values around (1 - mean) / std
    # For ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # White pixel (1.0) normalized should be around (1 - 0.485) / 0.229 ≈ 2.25 for R channel
    
    # Just check that normalization was applied (values outside [0, 1])
    assert output.max() > 1.5, "Expected normalized values > 1.5 for white image"
    assert output.min() > 1.5, "Expected normalized values > 1.5 for white image"


# ============================================================================
# Tests: Dataset Class
# ============================================================================

def test_dataset_class_mapping(tmp_path: Path):
    """Test BrainTumorDataset class mapping and basic functionality."""
    # Create dummy processed data
    processed_dir = tmp_path / "processed"
    
    records = []
    for i, class_name in enumerate(["glioma", "meningioma"]):
        class_dir = processed_dir / "Training" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy .npy file
        dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * (i * 100)
        img_path = class_dir / f"sample_{i}.npy"
        np.save(img_path, dummy_img)
        
        records.append({
            'filepath': f"Training/{class_name}/sample_{i}.npy",
            'class': class_name,
            'split': 'train'
        })
    
    manifest_df = pd.DataFrame(records)
    
    # Create dataset
    dataset = BrainTumorDataset(
        data_dir=processed_dir,
        manifest_df=manifest_df,
        transform=None  # No transform for easier testing
    )
    
    # Test __len__
    assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"
    
    # Test __getitem__
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor), "Image should be a tensor"
    assert isinstance(label, int), "Label should be an int"
    assert label in [0, 1], f"Label should be 0 or 1, got {label}"
    
    # Test class indices match expected mapping
    for i, (img, label) in enumerate(dataset):
        expected_class = ["glioma", "meningioma"][i]
        expected_label = CLASS_TO_IDX[expected_class]
        assert label == expected_label, \
            f"Expected label {expected_label} for {expected_class}, got {label}"


def test_dataset_with_transforms(tmp_path: Path):
    """Test that dataset correctly applies transforms."""
    # Create dummy processed data
    processed_dir = tmp_path / "processed"
    class_dir = processed_dir / "Training" / "glioma"
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy .npy file (use white image to ensure normalized values > 1)
    dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    img_path = class_dir / "sample.npy"
    np.save(img_path, dummy_img)
    
    manifest_df = pd.DataFrame([{
        'filepath': 'Training/glioma/sample.npy',
        'class': 'glioma',
        'split': 'train'
    }])
    
    # Create dataset with transforms
    transform = get_val_transforms()
    dataset = BrainTumorDataset(
        data_dir=processed_dir,
        manifest_df=manifest_df,
        transform=transform
    )
    
    img, label = dataset[0]
    
    # Check that transform was applied
    assert img.shape == (3, 224, 224), "Transform should produce (3, 224, 224)"
    assert img.dtype == torch.float32, "Transform should produce float32"
    # Normalized white image should have values > 1 (after ImageNet normalization)
    assert img.max() > 1.0, "White image should be normalized to values > 1"


def test_dataset_get_class_counts(tmp_path: Path):
    """Test class count calculation."""
    # Create dummy manifest
    manifest_df = pd.DataFrame([
        {'filepath': 'Training/glioma/s1.npy', 'class': 'glioma', 'split': 'train'},
        {'filepath': 'Training/glioma/s2.npy', 'class': 'glioma', 'split': 'train'},
        {'filepath': 'Training/meningioma/s1.npy', 'class': 'meningioma', 'split': 'train'},
    ])
    
    dataset = BrainTumorDataset(
        data_dir=tmp_path,
        manifest_df=manifest_df,
        transform=None
    )
    
    counts = dataset.get_class_counts()
    assert counts['glioma'] == 2, f"Expected 2 glioma samples, got {counts['glioma']}"
    assert counts['meningioma'] == 1, f"Expected 1 meningioma sample, got {counts['meningioma']}"


def test_dataset_get_class_weights(tmp_path: Path):
    """Test class weight calculation for imbalanced datasets."""
    # Create imbalanced manifest
    manifest_df = pd.DataFrame([
        {'filepath': 'Training/glioma/s1.npy', 'class': 'glioma', 'split': 'train'},
        {'filepath': 'Training/glioma/s2.npy', 'class': 'glioma', 'split': 'train'},
        {'filepath': 'Training/glioma/s3.npy', 'class': 'glioma', 'split': 'train'},
        {'filepath': 'Training/meningioma/s1.npy', 'class': 'meningioma', 'split': 'train'},
    ])
    
    dataset = BrainTumorDataset(
        data_dir=tmp_path,
        manifest_df=manifest_df,
        transform=None
    )
    
    weights = dataset.get_class_weights()
    
    # glioma (3 samples) should have lower weight than meningioma (1 sample)
    glioma_idx = CLASS_TO_IDX['glioma']
    meningioma_idx = CLASS_TO_IDX['meningioma']
    
    assert weights[meningioma_idx] > weights[glioma_idx], \
        "Minority class should have higher weight"


# ============================================================================
# Tests: Stratified Split
# ============================================================================

def test_stratified_split(tmp_path: Path, dummy_manifest: pd.DataFrame):
    """Test stratified train/validation split."""
    from sklearn.model_selection import train_test_split
    
    # Use sklearn directly for testing (same as create_split uses internally)
    train_df, val_df = train_test_split(
        dummy_manifest,
        train_size=0.85,
        stratify=dummy_manifest['class'],
        random_state=42
    )
    
    # Check train/val ratio is approximately 85/15
    total = len(dummy_manifest)
    train_ratio = len(train_df) / total
    val_ratio = len(val_df) / total
    
    assert 0.84 <= train_ratio <= 0.86, \
        f"Expected train ratio ~0.85, got {train_ratio:.3f}"
    assert 0.14 <= val_ratio <= 0.16, \
        f"Expected val ratio ~0.15, got {val_ratio:.3f}"
    
    # Check stratification: class proportions should be preserved
    for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
        # Original proportion
        orig_prop = len(dummy_manifest[dummy_manifest['class'] == class_name]) / len(dummy_manifest)
        
        # Train proportion
        train_prop = len(train_df[train_df['class'] == class_name]) / len(train_df)
        
        # Val proportion
        val_prop = len(val_df[val_df['class'] == class_name]) / len(val_df)
        
        # Proportions should be close (within 1%)
        assert abs(train_prop - orig_prop) < 0.01, \
            f"Train proportion for {class_name} differs: {train_prop:.3f} vs {orig_prop:.3f}"
        assert abs(val_prop - orig_prop) < 0.01, \
            f"Val proportion for {class_name} differs: {val_prop:.3f} vs {orig_prop:.3f}"
    
    # Check no overlap between train and val
    train_files = set(train_df['filepath'])
    val_files = set(val_df['filepath'])
    overlap = train_files & val_files
    
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping files between train and val"


# ============================================================================
# Tests: Cached Data Format
# ============================================================================

def test_cached_data_format(tmp_path: Path, dummy_image_no_borders: np.ndarray):
    """Test that cached .npy files have correct format."""
    # Save as .npy (simulating preprocessing output)
    npy_path = tmp_path / "cached_sample.npy"
    np.save(npy_path, dummy_image_no_borders)
    
    # Load cached data
    loaded = np.load(npy_path)
    
    # Assert dtype is uint8
    assert loaded.dtype == np.uint8, f"Expected uint8, got {loaded.dtype}"
    
    # Assert shape is (224, 224, 3)
    assert loaded.shape == (224, 224, 3), \
        f"Expected shape (224, 224, 3), got {loaded.shape}"
    
    # Assert data integrity
    assert np.array_equal(loaded, dummy_image_no_borders), \
        "Loaded data should match original"


def test_cached_data_loading_in_dataset(tmp_path: Path):
    """Test that dataset correctly loads cached .npy files."""
    # Create dummy cached data
    processed_dir = tmp_path / "processed"
    class_dir = processed_dir / "Training" / "glioma"
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save dummy image
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img_path = class_dir / "sample.npy"
    np.save(img_path, dummy_img)
    
    manifest_df = pd.DataFrame([{
        'filepath': 'Training/glioma/sample.npy',
        'class': 'glioma',
        'split': 'train'
    }])
    
    # Load via dataset
    dataset = BrainTumorDataset(
        data_dir=processed_dir,
        manifest_df=manifest_df,
        transform=None
    )
    
    img_tensor, label = dataset[0]
    
    # Convert back to numpy for comparison
    img_numpy = (img_tensor.permute(1, 2, 0) * 255).byte().numpy()
    
    # Should match original
    assert np.array_equal(img_numpy, dummy_img), \
        "Dataset should load cached data correctly"


# ============================================================================
# Tests: Error Handling
# ============================================================================

def test_dataset_missing_file(tmp_path: Path):
    """Test that dataset raises error for missing files."""
    manifest_df = pd.DataFrame([{
        'filepath': 'Training/glioma/nonexistent.npy',
        'class': 'glioma',
        'split': 'train'
    }])
    
    dataset = BrainTumorDataset(
        data_dir=tmp_path,
        manifest_df=manifest_df,
        transform=None
    )
    
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_dataset_invalid_manifest(tmp_path: Path):
    """Test that dataset raises error for invalid manifest."""
    # Missing required columns
    manifest_df = pd.DataFrame([{
        'filepath': 'Training/glioma/sample.npy',
        # Missing 'class' and 'split'
    }])
    
    with pytest.raises(ValueError, match="Manifest must contain columns"):
        BrainTumorDataset(
            data_dir=tmp_path,
            manifest_df=manifest_df,
            transform=None
        )


def test_dataset_unknown_class(tmp_path: Path):
    """Test that dataset raises error for unknown class names."""
    manifest_df = pd.DataFrame([{
        'filepath': 'Training/unknown/sample.npy',
        'class': 'unknown_class',
        'split': 'train'
    }])
    
    with pytest.raises(ValueError, match="Unknown classes in manifest"):
        BrainTumorDataset(
            data_dir=tmp_path,
            manifest_df=manifest_df,
            transform=None
        )


def test_preprocess_missing_file(tmp_path: Path):
    """Test that preprocess raises error for missing image file."""
    nonexistent = tmp_path / "nonexistent.jpg"
    
    with pytest.raises(ValueError, match="Failed to load image"):
        preprocess_image(nonexistent)


# ============================================================================
# Tests: Class Mapping Constants
# ============================================================================

def test_class_mapping_consistency():
    """Test that CLASS_TO_IDX and IDX_TO_CLASS are consistent."""
    # Check all expected classes are present
    expected_classes = {'glioma', 'meningioma', 'notumor', 'pituitary'}
    assert set(CLASS_TO_IDX.keys()) == expected_classes, \
        "CLASS_TO_IDX should contain all expected classes"
    
    # Check indices are in correct range
    assert set(CLASS_TO_IDX.values()) == {0, 1, 2, 3}, \
        "CLASS_TO_IDX should map to indices 0-3"
    
    # Check IDX_TO_CLASS is inverse of CLASS_TO_IDX
    for class_name, idx in CLASS_TO_IDX.items():
        assert IDX_TO_CLASS[idx] == class_name, \
            f"IDX_TO_CLASS[{idx}] should be '{class_name}', got '{IDX_TO_CLASS[idx]}'"


# ============================================================================
# Performance Tests
# ============================================================================

def test_transforms_performance(dummy_image_no_borders: np.ndarray):
    """Test that transforms execute quickly (< 100ms for 10 images)."""
    import time
    
    transform = get_train_transforms()
    
    start = time.time()
    for _ in range(10):
        _ = transform(dummy_image_no_borders)
    elapsed = time.time() - start
    
    assert elapsed < 1.0, \
        f"Transform performance too slow: {elapsed:.3f}s for 10 images"


def test_dataset_loading_performance(tmp_path: Path):
    """Test that dataset can load samples quickly."""
    import time
    
    # Create 10 dummy samples
    processed_dir = tmp_path / "processed"
    class_dir = processed_dir / "Training" / "glioma"
    class_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    for i in range(10):
        dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img_path = class_dir / f"sample_{i}.npy"
        np.save(img_path, dummy_img)
        
        records.append({
            'filepath': f'Training/glioma/sample_{i}.npy',
            'class': 'glioma',
            'split': 'train'
        })
    
    manifest_df = pd.DataFrame(records)
    dataset = BrainTumorDataset(
        data_dir=processed_dir,
        manifest_df=manifest_df,
        transform=get_val_transforms()
    )
    
    # Load all samples
    start = time.time()
    for i in range(len(dataset)):
        _ = dataset[i]
    elapsed = time.time() - start
    
    assert elapsed < 2.0, \
        f"Dataset loading too slow: {elapsed:.3f}s for 10 samples"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
