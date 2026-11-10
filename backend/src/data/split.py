#!/usr/bin/env python3
"""
Dataset Split Script - Stage 4
Creates stratified train/validation split and manifest CSV.
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import configure_utf8_console  # noqa: E402

configure_utf8_console()


def content_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """MD5 of a cached file's bytes (used to detect identical source scans)."""
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def add_content_hashes(df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``content_hash`` column."""
    df = df.copy()
    df['content_hash'] = [
        content_hash(processed_dir / rel_path) for rel_path in df['filepath']
    ]
    return df


def scan_processed_data(
    processed_dir: Path,
    split_name: str,
    classes: list = None
) -> pd.DataFrame:
    """
    Scan processed data directory and create file list.
    
    Args:
        processed_dir: Root directory of preprocessed data
        split_name: Name of the split (Training or Testing)
        classes: List of class names to scan
        
    Returns:
        DataFrame with columns ['filepath', 'class', 'split']
    """
    if classes is None:
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    split_dir = processed_dir / split_name
    
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    records = []
    
    for class_name in classes:
        class_dir = split_dir / class_name
        
        if not class_dir.exists():
            print(f"⚠ Warning: Class directory not found: {class_dir}")
            continue
        
        # Find all .npy files
        npy_files = list(class_dir.glob('*.npy'))
        
        for npy_file in npy_files:
            # Store path relative to processed_dir
            rel_path = npy_file.relative_to(processed_dir)
            
            records.append({
                'filepath': str(rel_path).replace('\\', '/'),  # Use forward slashes for consistency
                'class': class_name,
                'split': split_name.lower()  # 'training' or 'testing'
            })
    
    return pd.DataFrame(records)


def create_split(
    processed_dir: Path,
    train_val_ratio: float = 0.85,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create stratified train/validation split from Training data.
    
    The Testing split is kept locked for final evaluation and not included
    in train/val split.
    
    Args:
        processed_dir: Root directory containing preprocessed data
        train_val_ratio: Ratio of training samples (rest goes to validation)
        seed: Random seed for reproducibility
        
    Returns:
        Complete manifest DataFrame with train/val/test splits
    """
    processed_dir = Path(processed_dir)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    print("="*70)
    print("Brain Tumor MRI - Dataset Split")
    print("="*70)
    print(f"Processed data: {processed_dir}")
    print(f"Train/Val ratio: {train_val_ratio:.0%} / {1-train_val_ratio:.0%}")
    print(f"Random seed: {seed}")
    print()
    
    # Scan Training data
    print("Scanning Training data...")
    training_df = scan_processed_data(processed_dir, 'Training', classes)
    
    if training_df.empty:
        raise ValueError("No training data found. Run preprocess.py first.")
    
    print(f"  Found {len(training_df)} training images")
    for class_name in classes:
        count = len(training_df[training_df['class'] == class_name])
        print(f"    {class_name}: {count}")
    
    # ------------------------------------------------------------------
    # Leakage control (content-based)
    #
    # The Kaggle dataset contains byte-identical duplicates, and the same
    # scan can appear in both the Training and Testing folders. Without
    # deduplication those scans end up in train *and* test, inflating the
    # reported test metrics. Hash every cached image and:
    #   1. drop test rows whose content already exists in the training pool;
    #   2. collapse duplicate training rows so train/val cannot share a scan.
    # ------------------------------------------------------------------
    print("\nAuditing duplicate content (train/val/test leakage)...")
    training_df = add_content_hashes(training_df, processed_dir)
    test_df = scan_processed_data(processed_dir, 'Testing', classes)

    if not test_df.empty:
        test_df = add_content_hashes(test_df, processed_dir)
        train_hashes = set(training_df['content_hash'])
        leaked_mask = test_df['content_hash'].isin(train_hashes)
        leaked_count = int(leaked_mask.sum())
        if leaked_count:
            leaked_classes = (
                test_df.loc[leaked_mask, 'class'].value_counts().to_dict()
            )
            print(
                f"  [WARN] Dropping {leaked_count} test scans whose content "
                f"also appears in the training pool: {leaked_classes}"
            )
            test_df = test_df.loc[~leaked_mask].copy()
        else:
            print("  [OK] No test content found in the training pool")

        before = len(test_df)
        test_df = test_df.drop_duplicates(subset='content_hash', keep='first').copy()
        if len(test_df) != before:
            print(f"  [WARN] Dropped {before - len(test_df)} duplicate test scans")
        print(f"  Test scans after leakage audit: {len(test_df)}")

    before = len(training_df)
    training_df = training_df.drop_duplicates(
        subset='content_hash', keep='first'
    ).reset_index(drop=True)
    if len(training_df) != before:
        print(
            f"  [WARN] Dropped {before - len(training_df)} duplicate training "
            "scans (prevents train/val leakage)"
        )
    else:
        print("  [OK] No duplicate training content")

    train_df, val_df = train_test_split(
        training_df,
        train_size=train_val_ratio,
        stratify=training_df['class'],
        random_state=seed
    )

    # Remove the audit column before writing the manifest.
    train_df = train_df.drop(columns=['content_hash']).copy()
    val_df = val_df.drop(columns=['content_hash']).copy()
    test_df = test_df.drop(columns=['content_hash'], errors='ignore').copy()
    
    # Update split labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    print(f"  Train: {len(train_df)} images")
    for class_name in classes:
        count = len(train_df[train_df['class'] == class_name])
        print(f"    {class_name}: {count}")
    
    print(f"  Val: {len(val_df)} images")
    for class_name in classes:
        count = len(val_df[val_df['class'] == class_name])
        print(f"    {class_name}: {count}")
    
    # Testing data was scanned and leak-audited above (locked for evaluation)
    print("\nTesting split (locked for final evaluation)...")
    
    if test_df.empty:
        print("  [WARN] No testing data found")
    else:
        test_df = test_df.copy()
        test_df['split'] = 'test'
        print(f"  Found {len(test_df)} test images")
        for class_name in classes:
            count = len(test_df[test_df['class'] == class_name])
            print(f"    {class_name}: {count}")
    
    # Combine all splits
    manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Sort by split then class for cleaner manifest
    manifest = manifest.sort_values(['split', 'class', 'filepath']).reset_index(drop=True)
    
    return manifest


def verify_manifest(manifest: pd.DataFrame, processed_dir: Path = None) -> bool:
    """
    Verify manifest integrity.
    
    Args:
        manifest: Manifest DataFrame to verify
        processed_dir: Optional processed-data root. When provided, cached
            files are hashed so identical scans leaking across splits are
            detected (the check that filepath uniqueness alone misses).
        
    Returns:
        True if manifest is valid, False otherwise
    """
    print("\n" + "="*70)
    print("Manifest Verification")
    print("="*70)
    
    valid = True
    
    # Check required columns
    required_cols = ['filepath', 'class', 'split']
    if not all(col in manifest.columns for col in required_cols):
        print("✗ Missing required columns")
        valid = False
    else:
        print("✓ Required columns present")
    
    # Check for duplicates
    duplicates = manifest['filepath'].duplicated().sum()
    if duplicates > 0:
        print(f"✗ Found {duplicates} duplicate filepaths")
        valid = False
    else:
        print("✓ No duplicate filepaths")
    
    # Check splits
    expected_splits = {'train', 'val', 'test'}
    actual_splits = set(manifest['split'].unique())
    
    if 'test' not in actual_splits:
        print("⚠ Warning: No test split (testing data may not be available)")
    elif not actual_splits.issubset(expected_splits | {'training', 'testing'}):
        print(f"✗ Unexpected split values: {actual_splits - expected_splits}")
        valid = False
    else:
        print("✓ Split values valid")
    
    # Check classes
    expected_classes = {'glioma', 'meningioma', 'notumor', 'pituitary'}
    actual_classes = set(manifest['class'].unique())
    
    if actual_classes != expected_classes:
        print(f"⚠ Expected classes: {expected_classes}")
        print(f"  Actual classes: {actual_classes}")
    else:
        print("✓ All expected classes present")
    
    # Check class balance
    print("\nClass distribution:")
    for split in ['train', 'val', 'test']:
        split_df = manifest[manifest['split'] == split]
        if len(split_df) > 0:
            print(f"  {split}:")
            for class_name in sorted(expected_classes):
                count = len(split_df[split_df['class'] == class_name])
                pct = 100 * count / len(split_df)
                print(f"    {class_name}: {count} ({pct:.1f}%)")
    
    # Content-level duplicate check across splits
    if processed_dir is not None:
        processed_dir = Path(processed_dir)
        print("\nContent-level leakage check:")
        hashes = add_content_hashes(manifest, processed_dir)
        split_sets = hashes.groupby('content_hash')['split'].nunique()
        leaking = split_sets[split_sets > 1]
        if len(leaking) > 0:
            affected = int(hashes['content_hash'].isin(leaking.index).sum())
            print(
                f"✗ {len(leaking)} duplicate image groups span multiple splits "
                f"({affected} rows) — test metrics would be inflated"
            )
            valid = False
        else:
            print("✓ No content duplicates across splits")
    
    return valid


def main():
    """Main split routine."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    manifest_path = processed_dir / "manifest.csv"
    
    # Check if preprocessed data exists
    if not processed_dir.exists():
        print(f"✗ Preprocessed data not found: {processed_dir}")
        print("  Run preprocess.py first to create cached data")
        sys.exit(1)
    
    try:
        # Create split
        manifest = create_split(
            processed_dir=processed_dir,
            train_val_ratio=0.85,
            seed=42
        )
        
        # Verify manifest
        valid = verify_manifest(manifest, processed_dir=processed_dir)
        
        if not valid:
            print("\n✗ Manifest validation failed — not saving")
            sys.exit(1)
        
        # Save manifest
        manifest.to_csv(manifest_path, index=False)
        print(f"\n✓ Manifest saved to: {manifest_path}")
        print(f"  Total samples: {len(manifest)}")
        print(f"  Train: {len(manifest[manifest['split'] == 'train'])}")
        print(f"  Val: {len(manifest[manifest['split'] == 'val'])}")
        print(f"  Test: {len(manifest[manifest['split'] == 'test'])}")
        
        print("\n" + "="*70)
        print("✓ Dataset split complete!")
        print("="*70)
        print("\nNext steps:")
        print("1. Review manifest.csv")
        print("2. Configure training in configs/")
        print("3. Run training script")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Split creation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
