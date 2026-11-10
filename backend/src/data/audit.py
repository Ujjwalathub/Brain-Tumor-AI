#!/usr/bin/env python3
"""
Data Audit Script - Stage 3
Checks for duplicates, corruption, and dataset integrity.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import hashlib
from PIL import Image
import imagehash
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_perceptual_hash(filepath: str) -> str:
    """Compute perceptual hash using difference hash."""
    try:
        img = Image.open(filepath)
        return str(imagehash.dhash(img, hash_size=8))
    except Exception as e:
        return None


def audit_dataset(base_dir: str = "Dataset"):
    """
    Audit the brain tumor dataset.
    
    Checks:
    1. File counts per class and split
    2. Image properties (size, channels, format)
    3. Exact duplicates (file hash)
    4. Near-duplicates (perceptual hash)
    5. Corrupt files
    6. Cross-split leakage
    """
    print("="*70)
    print("Brain Tumor MRI Dataset Audit")
    print("="*70)
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"✗ Dataset directory not found: {base_dir}")
        print("  Expected structure: Dataset/Training/ and Dataset/Testing/")
        return False
    
    results = {
        "splits": {},
        "duplicates": [],
        "near_duplicates": [],
        "corrupt": [],
        "cross_split_leakage": []
    }
    
    # Expected classes
    expected_classes = ["glioma", "meningioma", "notumor", "pituitary"]
    
    # Process each split
    for split in ["Training", "Testing"]:
        split_dir = base_path / split
        if not split_dir.exists():
            print(f"\n✗ {split} directory not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"Auditing {split} split...")
        print(f"{'='*70}")
        
        split_data = {
            "classes": {},
            "file_hashes": {},
            "perceptual_hashes": defaultdict(list),
            "total_files": 0
        }
        
        # Process each class
        for class_name in expected_classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"  ✗ Class directory not found: {class_name}")
                continue
            
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            
            class_data = {
                "count": len(image_files),
                "sizes": set(),
                "channels": set(),
                "formats": set()
            }
            
            # Analyze each image
            for img_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
                # File hash
                file_hash = compute_file_hash(str(img_path))
                
                # Check for exact duplicates within split
                if file_hash in split_data["file_hashes"]:
                    results["duplicates"].append({
                        "split": split,
                        "file1": split_data["file_hashes"][file_hash],
                        "file2": str(img_path)
                    })
                else:
                    split_data["file_hashes"][file_hash] = str(img_path)
                
                # Perceptual hash
                phash = compute_perceptual_hash(str(img_path))
                if phash:
                    split_data["perceptual_hashes"][phash].append(str(img_path))
                
                # Image properties
                try:
                    with Image.open(img_path) as img:
                        class_data["sizes"].add(img.size)
                        class_data["channels"].add(img.mode)
                        class_data["formats"].add(img.format)
                except Exception as e:
                    results["corrupt"].append({
                        "file": str(img_path),
                        "error": str(e)
                    })
            
            split_data["classes"][class_name] = class_data
            split_data["total_files"] += class_data["count"]
            
            # Print class summary
            print(f"  {class_name}: {class_data['count']} images")
            if class_data["sizes"]:
                size_summary = list(class_data["sizes"])[:5]
                print(f"    Sizes (sample): {size_summary}")
            if class_data["channels"]:
                print(f"    Channels: {class_data['channels']}")
        
        results["splits"][split] = split_data
        print(f"\n  Total {split} images: {split_data['total_files']}")
    
    # Check for near-duplicates within each split
    print(f"\n{'='*70}")
    print("Checking for near-duplicates...")
    print(f"{'='*70}")
    
    for split, split_data in results["splits"].items():
        for phash, file_list in split_data["perceptual_hashes"].items():
            if len(file_list) > 1:
                results["near_duplicates"].append({
                    "split": split,
                    "files": file_list,
                    "hash": phash
                })
    
    # Check for cross-split leakage
    print("\nChecking for cross-split leakage...")
    
    if "Training" in results["splits"] and "Testing" in results["splits"]:
        train_hashes = set(results["splits"]["Training"]["perceptual_hashes"].keys())
        test_hashes = set(results["splits"]["Testing"]["perceptual_hashes"].keys())
        
        common_hashes = train_hashes & test_hashes
        
        for phash in common_hashes:
            train_files = results["splits"]["Training"]["perceptual_hashes"][phash]
            test_files = results["splits"]["Testing"]["perceptual_hashes"][phash]
            
            results["cross_split_leakage"].append({
                "hash": phash,
                "train_files": train_files,
                "test_files": test_files
            })
    
    # Print summary
    print(f"\n{'='*70}")
    print("Audit Summary")
    print(f"{'='*70}")
    
    # Dataset size
    expected_train = 5600
    expected_test = 1600
    actual_train = results["splits"].get("Training", {}).get("total_files", 0)
    actual_test = results["splits"].get("Testing", {}).get("total_files", 0)
    
    print(f"\nDataset size:")
    print(f"  Training: {actual_train} (expected {expected_train})")
    print(f"  Testing: {actual_test} (expected {expected_test})")
    
    if actual_train == expected_train and actual_test == expected_test:
        print("  ✓ File counts match expected")
    else:
        print("  ⚠ File counts do not match expected")
    
    # Issues found
    issues_found = False
    
    if results["duplicates"]:
        print(f"\n✗ Exact duplicates found: {len(results['duplicates'])}")
        for dup in results["duplicates"][:5]:
            print(f"    {dup['file1']}")
            print(f"    {dup['file2']}")
        issues_found = True
    else:
        print("\n✓ No exact duplicates found")
    
    if results["near_duplicates"]:
        print(f"\n⚠ Near-duplicates found: {len(results['near_duplicates'])} groups")
        for i, dup_group in enumerate(results["near_duplicates"][:3]):
            print(f"    Group {i+1} ({len(dup_group['files'])} files):")
            for f in dup_group['files'][:2]:
                print(f"      {f}")
    else:
        print("✓ No near-duplicates found")
    
    if results["cross_split_leakage"]:
        print(f"\n✗ Cross-split leakage found: {len(results['cross_split_leakage'])} cases")
        print("  ⚠ Training and testing sets contain similar images!")
        for i, leak in enumerate(results["cross_split_leakage"][:3]):
            print(f"    Case {i+1}:")
            print(f"      Train: {leak['train_files'][0]}")
            print(f"      Test: {leak['test_files'][0]}")
        issues_found = True
    else:
        print("✓ No cross-split leakage detected")
    
    if results["corrupt"]:
        print(f"\n✗ Corrupt files found: {len(results['corrupt'])}")
        for corrupt in results["corrupt"][:5]:
            print(f"    {corrupt['file']}: {corrupt['error']}")
        issues_found = True
    else:
        print("✓ No corrupt files found")
    
    # Save detailed report
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    report_path = report_dir / "audit_report.txt"
    with open(report_path, "w") as f:
        f.write("Brain Tumor MRI Dataset Audit Report\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Training images: {actual_train}\n")
        f.write(f"Testing images: {actual_test}\n\n")
        
        f.write(f"Exact duplicates: {len(results['duplicates'])}\n")
        f.write(f"Near-duplicates: {len(results['near_duplicates'])}\n")
        f.write(f"Cross-split leakage: {len(results['cross_split_leakage'])}\n")
        f.write(f"Corrupt files: {len(results['corrupt'])}\n\n")
        
        if results["cross_split_leakage"]:
            f.write("\nCross-split leakage details:\n")
            for leak in results["cross_split_leakage"]:
                f.write(f"\nHash: {leak['hash']}\n")
                f.write(f"  Training: {leak['train_files']}\n")
                f.write(f"  Testing: {leak['test_files']}\n")
    
    print(f"\nDetailed report saved to: {report_path}")
    
    print(f"\n{'='*70}")
    if not issues_found:
        print("✓ Dataset audit passed - ready for preprocessing")
        return True
    else:
        print("⚠ Issues found - review before proceeding")
        print("\nRecommendations:")
        if results["duplicates"] or results["cross_split_leakage"]:
            print("  - Remove duplicates from Training set (keep Testing untouched)")
        if results["corrupt"]:
            print("  - Remove or fix corrupt files")
        return False


def main():
    """Main audit routine."""
    try:
        success = audit_dataset()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Audit failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
