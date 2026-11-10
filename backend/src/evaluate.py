"""
Evaluation Script - Stage 7
Comprehensive model evaluation with metrics, visualizations, and error analysis.

Hardware: RTX 3050 6GB, CUDA-only evaluation
Metrics: Accuracy, per-class precision/recall/F1, macro-recall, ROC-AUC, confusion matrix
"""

import argparse
import json
import sys
import time
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import BrainTumorDataset, CLASS_TO_IDX, IDX_TO_CLASS
from data.transforms import get_val_transforms
from models import create_model, get_model_info
from utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate brain tumor classifier with comprehensive metrics"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pth) or glob pattern for multiple checkpoints"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/efficientnet_b0.yaml)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device (default: auto — CUDA if available)",
    )
    parser.add_argument(
        "--exclude-leaked",
        action="store_true",
        help=(
            "Drop evaluated samples whose content also appears in another "
            "split (honest, leakage-free metrics)"
        ),
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    """Resolve 'auto' to CUDA when available, otherwise CPU."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return requested


def setup_evaluation(config: dict, requested_device: str = "auto") -> Tuple[str, Path, Path]:
    """
    Setup evaluation environment: device, seeds, directories.
    
    Args:
        config: Configuration dictionary
        requested_device: 'auto', 'cuda' or 'cpu'
        
    Returns:
        Tuple of (device, processed_dir, reports_dir)
    """
    device = resolve_device(requested_device)
    
    print("=" * 70)
    print("Brain Tumor MRI Classifier - Model Evaluation")
    print("=" * 70)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Device: CPU (CUDA not used)")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    # Set seeds for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Get directories
    workspace_root = Path.cwd()
    processed_dir = workspace_root / config['data']['processed_dir']
    reports_dir = workspace_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {processed_dir}")
    print(f"Reports directory: {reports_dir}")
    print()
    
    return device, processed_dir, reports_dir


def load_dataloader(
    config: dict,
    processed_dir: Path,
    split: str
) -> Tuple[DataLoader, pd.DataFrame]:
    """
    Create dataloader for evaluation.
    
    Args:
        config: Configuration dictionary
        processed_dir: Directory with processed data
        split: Data split to load (train/val/test)
        
    Returns:
        Tuple of (dataloader, manifest_df)
    """
    print("=" * 70)
    print(f"Loading Data (split: {split})")
    print("=" * 70)
    
    # Load manifest
    manifest_path = processed_dir / "manifest.csv"
    print(f"Loading manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    
    # Filter for specified split
    split_df = manifest[manifest['split'] == split].copy()
    
    if len(split_df) == 0:
        raise ValueError(f"No samples found for split '{split}' in manifest")
    
    print(f"Total samples in {split}: {len(split_df)}")
    
    # Class distribution
    print(f"\nClass distribution:")
    class_counts = split_df['class'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name:12s}: {count:4d} ({count/len(split_df)*100:.1f}%)")
    print()
    
    # Create dataset with validation transforms (no augmentation)
    val_transform = get_val_transforms()
    dataset = BrainTumorDataset(
        data_dir=processed_dir,
        manifest_df=split_df,
        transform=val_transform
    )
    
    # DataLoader with larger batch size for evaluation
    batch_size = config.get('evaluation', {}).get('batch_size', 128)
    num_workers = config['dataloader']['num_workers']
    pin_memory = config['dataloader']['pin_memory']
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    print()
    
    return dataloader, split_df


def load_model_and_checkpoint(
    config: dict,
    checkpoint_path: str,
    device: str
) -> Tuple[nn.Module, Dict]:
    """
    Load model architecture and checkpoint weights.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, checkpoint_data)
    """
    print("=" * 70)
    print("Loading Model")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    # weights_only=False: our own checkpoints contain numpy arrays in `metrics`.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint if available, otherwise use provided config
    model_config = checkpoint.get('config', config)
    
    # Create model
    model = create_model(model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Model info
    info = get_model_info(model)
    print(f"Model: {model_config['model']['name']}")
    print(f"Parameters: {info['total_params']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        val_metrics = checkpoint['metrics']
        if 'macro_recall' in val_metrics:
            print(f"Validation macro recall: {val_metrics['macro_recall']:.4f}")
    print()
    
    # Set to eval mode
    model.eval()
    
    # Convert to channels_last if needed
    if model_config['training'].get('channels_last', False):
        model = model.to(memory_format=torch.channels_last)
        print("Memory format: channels_last")
    
    # Move to GPU
    model = model.to(device)
    
    print()
    
    return model, checkpoint


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    manifest_df: pd.DataFrame,
    device: str,
    config: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Run inference on dataset and collect predictions.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        manifest_df: Manifest DataFrame with file paths
        device: Device to run on
        config: Configuration dictionary
        
    Returns:
        Tuple of (predictions, probabilities, ground_truth, file_paths)
    """
    print("=" * 70)
    print("Running Inference")
    print("=" * 70)
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    channels_last = config['training'].get('channels_last', False)
    use_amp = config['training']['use_amp']
    amp_dtype = torch.float16 if config['training']['amp_dtype'] == 'float16' else torch.bfloat16
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Evaluating", total=len(dataloader))
    
    for images, labels in pbar:
        # Move to device
        images = images.to(device, non_blocking=True)
        
        # Convert to channels_last if configured
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        
        # Mixed precision forward pass
        with autocast(enabled=use_amp, dtype=amp_dtype):
            outputs = model(images)
        
        # Get predictions and probabilities
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)
        
        # Collect results
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    # Convert to arrays
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)
    y_true = np.array(all_labels)
    
    # Get file paths from manifest
    file_paths = manifest_df['filepath'].tolist()
    
    # Log memory usage
    max_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"\nGPU max memory used: {max_memory_mb:.2f} MB")
    print()
    
    return y_pred, y_proba, y_true, file_paths


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        class_names: List of class names
        
    Returns:
        Dictionary with all metrics
    """
    print("=" * 70)
    print("Computing Metrics")
    print("=" * 70)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro-averaged recall (primary metric for medical imaging)
    macro_recall = recall.mean()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC-AUC for each class (one-vs-rest)
    roc_auc = {}
    roc_curves = {}
    
    try:
        for i, class_name in enumerate(class_names):
            # One-vs-rest binary labels
            y_true_binary = (y_true == i).astype(int)
            y_proba_class = y_proba[:, i]
            
            # Compute ROC-AUC
            auc = roc_auc_score(y_true_binary, y_proba_class)
            roc_auc[class_name] = float(auc)
            
            # Compute ROC curve for plotting
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_proba_class)
            roc_curves[class_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC: {e}")
        roc_auc = {name: 0.0 for name in class_names}
        roc_curves = {}
    
    # Build per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'roc_auc': roc_auc.get(class_name, 0.0)
        }
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_recall': float(macro_recall),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'roc_curves': roc_curves
    }
    
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Macro Recall: {macro_recall*100:.2f}%")
    print()
    
    return metrics


def print_metrics_table(metrics: Dict):
    """
    Print metrics in a nicely formatted table.
    
    Args:
        metrics: Dictionary with computed metrics
    """
    print("=" * 70)
    print("Detailed Metrics")
    print("=" * 70)
    print()
    
    # Create DataFrame for better formatting
    rows = []
    for class_name, class_metrics in metrics['per_class'].items():
        rows.append({
            'Class': class_name,
            'Precision': f"{class_metrics['precision']:.3f}",
            'Recall': f"{class_metrics['recall']:.3f}",
            'F1-Score': f"{class_metrics['f1']:.3f}",
            'ROC-AUC': f"{class_metrics['roc_auc']:.3f}",
            'Support': class_metrics['support']
        })
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    print(f"Overall Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Macro Recall:      {metrics['macro_recall']:.3f}")
    print()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    split: str,
    model_name: str,
    save_path: Path
):
    """
    Plot and save confusion matrix heatmap.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        split: Data split name
        model_name: Model name for title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy array if needed
    if isinstance(confusion_matrix, list):
        confusion_matrix = np.array(confusion_matrix)
    
    # Normalize confusion matrix to percentages
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'}
    )
    
    plt.title(f'Confusion Matrix - {model_name} ({split} set)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix: {save_path}")


def plot_roc_curves(
    roc_curves: Dict,
    class_names: List[str],
    metrics: Dict,
    split: str,
    model_name: str,
    save_path: Path
):
    """
    Plot and save ROC curves for all classes.
    
    Args:
        roc_curves: Dictionary with ROC curve data for each class
        class_names: List of class names
        metrics: Metrics dictionary with ROC-AUC scores
        split: Data split name
        model_name: Model name for title
        save_path: Path to save figure
    """
    if not roc_curves:
        print("Warning: No ROC curves to plot")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    colors = ['blue', 'red', 'green', 'orange']
    for i, class_name in enumerate(class_names):
        if class_name not in roc_curves:
            continue
        
        curve_data = roc_curves[class_name]
        fpr = np.array(curve_data['fpr'])
        tpr = np.array(curve_data['tpr'])
        auc = metrics['per_class'][class_name]['roc_auc']
        
        plt.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f'{class_name} (AUC = {auc:.3f})'
        )
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {model_name} ({split} set)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ROC curves: {save_path}")


def analyze_misclassifications(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    file_paths: List[str],
    class_names: List[str],
    split: str,
    model_name: str,
    save_path: Path
) -> pd.DataFrame:
    """
    Analyze and save misclassifications.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        file_paths: List of file paths
        class_names: List of class names
        split: Data split name
        model_name: Model name
        save_path: Path to save CSV
        
    Returns:
        DataFrame with misclassifications
    """
    # Find misclassified samples
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassifications found!")
        return pd.DataFrame()
    
    # Build DataFrame
    rows = []
    for idx in misclassified_indices:
        true_label = int(y_true[idx])
        pred_label = int(y_pred[idx])
        confidence = float(y_proba[idx, pred_label])
        
        rows.append({
            'filepath': file_paths[idx],
            'true_class': class_names[true_label],
            'predicted_class': class_names[pred_label],
            'confidence': confidence,
            'true_label': true_label,
            'predicted_label': pred_label
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by confidence (highest confidence errors first)
    df = df.sort_values('confidence', ascending=False)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    
    print(f"\nMisclassification Analysis:")
    print(f"  Total misclassifications: {len(df)}")
    print(f"  Misclassification rate: {len(df)/len(y_true)*100:.2f}%")
    print(f"\n  Sample misclassifications (highest confidence errors):")
    
    # Show top 5 high-confidence errors
    for i, row in df.head(5).iterrows():
        print(f"    {row['filepath']}")
        print(f"      True: {row['true_class']}, Predicted: {row['predicted_class']} "
              f"(confidence: {row['confidence']:.3f})")
    
    print(f"\nSaved misclassifications: {save_path}")
    
    return df


def identify_worst_class(metrics: Dict) -> Tuple[str, float]:
    """
    Identify the worst-performing class by recall.
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        Tuple of (class_name, recall)
    """
    worst_class = None
    worst_recall = 1.0
    
    for class_name, class_metrics in metrics['per_class'].items():
        recall = class_metrics['recall']
        if recall < worst_recall:
            worst_recall = recall
            worst_class = class_name
    
    return worst_class, worst_recall


def save_metrics_json(
    metrics: Dict,
    split: str,
    model_name: str,
    save_path: Path
):
    """
    Save detailed metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        split: Data split name
        model_name: Model name
        save_path: Path to save JSON
    """
    # Add metadata
    output = {
        'model': model_name,
        'split': split,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics
    }
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved detailed metrics: {save_path}")


def evaluate_single_checkpoint(
    checkpoint_path: str,
    config: dict,
    split: str,
    device: str,
    processed_dir: Path,
    reports_dir: Path
) -> Dict:
    """
    Evaluate a single checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        split: Data split to evaluate on
        device: Device to run on
        processed_dir: Directory with processed data
        reports_dir: Directory to save reports
        
    Returns:
        Dictionary with evaluation results
    """
    # Load data
    dataloader, manifest_df = load_dataloader(config, processed_dir, split)
    
    # Load model
    model, checkpoint = load_model_and_checkpoint(config, checkpoint_path, device)
    
    # Get model name for file naming
    model_name = config['model']['name']
    checkpoint_name = Path(checkpoint_path).stem
    
    # Run inference
    y_pred, y_proba, y_true, file_paths = run_inference(
        model, dataloader, manifest_df, device, config
    )
    
    # Compute metrics
    class_names = config['data']['classes']
    metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba, class_names)
    
    # Print metrics table
    print_metrics_table(metrics)
    
    # Identify worst-performing class
    worst_class, worst_recall = identify_worst_class(metrics)
    print(f"Worst-performing class: {worst_class} (recall: {worst_recall:.3f})")
    print()
    
    # Save visualizations
    print("=" * 70)
    print("Saving Visualizations")
    print("=" * 70)
    
    # Confusion matrix
    cm_path = reports_dir / f"confusion_matrix_{split}_{checkpoint_name}.png"
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        class_names,
        split,
        model_name,
        cm_path
    )
    
    # ROC curves
    roc_path = reports_dir / f"roc_curves_{split}_{checkpoint_name}.png"
    plot_roc_curves(
        metrics['roc_curves'],
        class_names,
        metrics,
        split,
        model_name,
        roc_path
    )
    
    # Misclassifications
    misclass_path = reports_dir / f"misclassifications_{split}_{checkpoint_name}.csv"
    misclass_df = analyze_misclassifications(
        y_true, y_pred, y_proba, file_paths, class_names, split, model_name, misclass_path
    )
    
    # Save detailed metrics JSON
    metrics_path = reports_dir / f"metrics_{split}_{checkpoint_name}.json"
    save_metrics_json(metrics, split, model_name, metrics_path)
    
    print()
    
    # Return summary for comparison
    return {
        'checkpoint': checkpoint_name,
        'model': model_name,
        'accuracy': metrics['accuracy'],
        'macro_recall': metrics['macro_recall'],
        'per_class_recalls': {name: m['recall'] for name, m in metrics['per_class'].items()},
        'num_params': get_model_info(model)['total_params']
    }


def evaluate_multiple_checkpoints(
    checkpoint_pattern: str,
    config: dict,
    split: str,
    device: str,
    processed_dir: Path,
    reports_dir: Path
):
    """
    Evaluate multiple checkpoints and create comparison table.
    
    Args:
        checkpoint_pattern: Glob pattern for checkpoint files
        config: Configuration dictionary
        split: Data split to evaluate on
        device: Device to run on
        processed_dir: Directory with processed data
        reports_dir: Directory to save reports
    """
    # Find all matching checkpoints
    checkpoint_paths = sorted(glob(checkpoint_pattern))
    
    if len(checkpoint_paths) == 0:
        raise ValueError(f"No checkpoints found matching pattern: {checkpoint_pattern}")
    
    print(f"Found {len(checkpoint_paths)} checkpoints to evaluate")
    print()
    
    # Evaluate each checkpoint
    results = []
    for i, checkpoint_path in enumerate(checkpoint_paths, 1):
        print(f"\n{'='*70}")
        print(f"Evaluating Checkpoint {i}/{len(checkpoint_paths)}")
        print(f"{'='*70}\n")
        
        result = evaluate_single_checkpoint(
            checkpoint_path, config, split, device, processed_dir, reports_dir
        )
        results.append(result)
    
    # Create comparison table
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print()
    
    # Build comparison DataFrame
    comparison_rows = []
    for result in results:
        row = {
            'Checkpoint': result['checkpoint'],
            'Model': result['model'],
            'Accuracy': f"{result['accuracy']:.4f}",
            'Macro Recall': f"{result['macro_recall']:.4f}",
            'Parameters': f"{result['num_params']:,}"
        }
        
        # Add per-class recalls
        for class_name, recall in result['per_class_recalls'].items():
            row[f'{class_name}_recall'] = f"{recall:.4f}"
        
        comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Print comparison
    print(comparison_df.to_string(index=False))
    print()
    
    # Save comparison table
    comparison_path = reports_dir / f"model_comparison_{split}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison table: {comparison_path}")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    device, processed_dir, reports_dir = setup_evaluation(config)
    
    # Check if checkpoint is a glob pattern (contains wildcards)
    if '*' in args.checkpoint or '?' in args.checkpoint:
        # Multiple checkpoints
        evaluate_multiple_checkpoints(
            args.checkpoint,
            config,
            args.split,
            device,
            processed_dir,
            reports_dir
        )
    else:
        # Single checkpoint
        result = evaluate_single_checkpoint(
            args.checkpoint,
            config,
            args.split,
            device,
            processed_dir,
            reports_dir
        )
        
        print("\n" + "=" * 70)
        print("Evaluation Complete")
        print("=" * 70)
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Macro Recall: {result['macro_recall']:.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
