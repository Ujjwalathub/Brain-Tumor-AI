#!/usr/bin/env python3
"""
Image Preprocessing Script - Stage 4
Crops black margins, resizes to 224x224, and caches as numpy arrays.
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import configure_utf8_console  # noqa: E402

configure_utf8_console()


def find_brain_contour(image: np.ndarray, threshold: int = 10) -> Tuple[int, int, int, int]:
    """
    Find bounding box of brain region by removing black margins.
    
    Args:
        image: Input image as numpy array (H, W, C)
        threshold: Pixel value threshold for considering as non-black
        
    Returns:
        Tuple of (x, y, width, height) for cropping
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Threshold to create binary mask
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no contours found, return full image bounds
        return 0, 0, image.shape[1], image.shape[0]
    
    # Get bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return x, y, w, h


def preprocess_image(image_path: Path, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess a single image: crop margins and resize.
    
    Args:
        image_path: Path to input image
        target_size: Target size (width, height) for resizing
        
    Returns:
        Preprocessed image as uint8 numpy array in **RGB** channel order
        (matching the inference API and torchvision's image conventions).
        
    Raises:
        ValueError: If image cannot be loaded or processed
    """
    # Read image. OpenCV loads BGR; the rest of the pipeline (PIL/torchvision)
    # expects RGB, so convert once after cropping. Grayscale source data is
    # unaffected, but genuinely colour scans are now cached correctly.
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Find brain region (find_brain_contour expects BGR / single channel)
    x, y, w, h = find_brain_contour(image)
    
    # Crop to brain region
    cropped = image[y:y+h, x:x+w]
    
    # BGR -> RGB
    if cropped.ndim == 3 and cropped.shape[2] == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    return resized


def preprocess_dataset(
    raw_dir: Path,
    processed_dir: Path,
    img_size: Tuple[int, int] = (224, 224),
    splits: list = None
) -> dict:
    """
    Preprocess entire dataset and cache to disk.
    
    Args:
        raw_dir: Root directory containing raw images (Training, Testing subdirs)
        processed_dir: Output directory for cached preprocessed images
        img_size: Target image size (width, height)
        splits: List of splits to process. Defaults to ['Training', 'Testing']
        
    Returns:
        Dictionary with statistics about processed images
    """
    if splits is None:
        splits = ['Training', 'Testing']
    
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    
    if not raw_dir.exists():
        raise ValueError(f"Raw dataset directory not found: {raw_dir}")
    
    # Expected classes
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    stats = {
        'total_processed': 0,
        'failed': [],
        'splits': {}
    }
    
    print("="*70)
    print("Brain Tumor MRI - Image Preprocessing")
    print("="*70)
    print(f"Raw data: {raw_dir}")
    print(f"Output: {processed_dir}")
    print(f"Target size: {img_size}")
    print()
    
    # Process each split
    for split in splits:
        split_raw = raw_dir / split
        split_processed = processed_dir / split
        
        if not split_raw.exists():
            print(f"⚠ Split not found: {split}, skipping...")
            continue
        
        print(f"Processing {split} split...")
        print("-"*70)
        
        split_stats = {
            'total': 0,
            'classes': {}
        }
        
        # Process each class
        for class_name in classes:
            class_raw = split_raw / class_name
            class_processed = split_processed / class_name
            
            if not class_raw.exists():
                print(f"  ⚠ Class not found: {class_name}, skipping...")
                continue
            
            # Create output directory
            class_processed.mkdir(parents=True, exist_ok=True)
            
            # Get all image files (sorted for deterministic output)
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(class_raw.glob(ext))
            image_files = sorted(set(image_files), key=lambda p: str(p).lower())
            
            class_count = 0
            class_failed = 0
            used_names = set()
            
            # Process each image
            for img_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
                try:
                    # Preprocess
                    processed = preprocess_image(img_path, target_size=img_size)
                    
                    # Save as numpy array. Two source files may share a stem
                    # (e.g. scan.jpg and scan.png); disambiguate instead of
                    # silently overwriting one of them.
                    stem = img_path.stem
                    if stem.lower() in used_names:
                        digest = hashlib.md5(str(img_path).encode('utf-8')).hexdigest()[:8]
                        stem = f"{stem}__{digest}"
                    used_names.add(stem.lower())
                    output_path = class_processed / f"{stem}.npy"
                    np.save(output_path, processed)
                    
                    class_count += 1
                    
                except Exception as e:
                    class_failed += 1
                    stats['failed'].append({
                        'file': str(img_path),
                        'error': str(e)
                    })
            
            split_stats['classes'][class_name] = {
                'processed': class_count,
                'failed': class_failed
            }
            split_stats['total'] += class_count
            
            print(f"  {class_name}: {class_count} processed, {class_failed} failed")
        
        stats['splits'][split] = split_stats
        stats['total_processed'] += split_stats['total']
        
        print(f"  Total {split}: {split_stats['total']} images")
        print()
    
    # Summary
    print("="*70)
    print("Preprocessing Summary")
    print("="*70)
    print(f"Total images processed: {stats['total_processed']}")
    print(f"Failed: {len(stats['failed'])}")
    
    if stats['failed']:
        print("\nFailed files:")
        for fail in stats['failed'][:10]:
            print(f"  {fail['file']}: {fail['error']}")
        if len(stats['failed']) > 10:
            print(f"  ... and {len(stats['failed']) - 10} more")
    
    # Per-split summary
    print("\nPer-split breakdown:")
    for split, split_stats in stats['splits'].items():
        print(f"\n{split}:")
        for class_name, class_stats in split_stats['classes'].items():
            print(f"  {class_name}: {class_stats['processed']}")
    
    print("\n✓ Preprocessing complete")
    print(f"Cached data saved to: {processed_dir}")
    
    return stats


def main():
    """Main preprocessing routine."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / "Dataset"
    processed_dir = project_root / "data" / "processed"
    
    try:
        stats = preprocess_dataset(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            img_size=(224, 224)
        )
        
        # Check if we got expected number of images
        expected_total = 7200  # 5600 train + 1600 test
        actual_total = stats['total_processed']
        
        if actual_total >= expected_total * 0.95:  # Allow 5% tolerance
            print(f"\n✓ Dataset preprocessing successful!")
            print(f"  Expected: ~{expected_total} images")
            print(f"  Processed: {actual_total} images")
            sys.exit(0)
        else:
            print(f"\n⚠ Processed fewer images than expected")
            print(f"  Expected: ~{expected_total} images")
            print(f"  Processed: {actual_total} images")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
