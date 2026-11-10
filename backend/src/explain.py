"""
Explainability Script - Stage 9
Grad-CAM visualization for brain tumor MRI classification.

Generates heatmaps showing which regions the model focuses on for predictions.
Supports GradCAM, GradCAM++, and ScoreCAM methods.

Hardware: RTX 3050 6GB, CUDA-only inference
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
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
        description="Generate Grad-CAM heatmaps for brain tumor classifier"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pth)"
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
        help="Data split to generate heatmaps for (default: test)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per class to visualize (default: 5)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        choices=["gradcam", "gradcam++", "scorecam"],
        help="CAM method to use (default: gradcam)"
    )
    return parser.parse_args()


def setup_explainability(config: dict) -> Tuple[str, Path, Path, Path]:
    """
    Setup environment: CUDA check, seeds, directories.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (device, processed_dir, reports_dir, gradcam_dir)
    """
    # Assert CUDA availability
    assert torch.cuda.is_available(), "CUDA GPU required. Aborting."
    device = "cuda"
    
    print("=" * 70)
    print("Brain Tumor MRI Classifier - Grad-CAM Explainability")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    # Set seeds for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Get directories
    workspace_root = Path.cwd()
    processed_dir = workspace_root / config['data']['processed_dir']
    reports_dir = workspace_root / "reports"
    gradcam_dir = reports_dir / "gradcam"
    
    # Create gradcam directory and subdirectories for each class
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    for class_name in config['data']['classes']:
        (gradcam_dir / class_name).mkdir(exist_ok=True)
    
    print(f"Data directory: {processed_dir}")
    print(f"Reports directory: {reports_dir}")
    print(f"Grad-CAM output: {gradcam_dir}")
    print()
    
    return device, processed_dir, reports_dir, gradcam_dir


def sample_images_per_class(
    manifest: pd.DataFrame,
    split: str,
    num_samples: int,
    class_names: List[str]
) -> pd.DataFrame:
    """
    Sample stratified images from each class.
    
    Args:
        manifest: Full manifest DataFrame
        split: Data split to sample from
        num_samples: Number of samples per class
        class_names: List of class names
        
    Returns:
        DataFrame with sampled images
    """
    print("=" * 70)
    print(f"Sampling Images (split: {split})")
    print("=" * 70)
    
    # Filter for split
    split_df = manifest[manifest['split'] == split].copy()
    
    # Sample from each class
    sampled_dfs = []
    for class_name in class_names:
        class_df = split_df[split_df['class'] == class_name]
        
        # Sample min(num_samples, available) images
        n_samples = min(num_samples, len(class_df))
        
        if n_samples == 0:
            print(f"  {class_name:12s}: No samples available")
            continue
        
        sampled = class_df.sample(n=n_samples, random_state=42)
        sampled_dfs.append(sampled)
        
        print(f"  {class_name:12s}: Sampled {n_samples} images")
    
    if not sampled_dfs:
        raise ValueError(f"No samples found for split '{split}'")
    
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\nTotal sampled: {len(result_df)} images")
    print()
    
    return result_df


def load_model_and_checkpoint(
    config: dict,
    checkpoint_path: str,
    device: str
) -> nn.Module:
    """
    Load model architecture and checkpoint weights.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Model in eval mode
    """
    print("=" * 70)
    print("Loading Model")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    # weights_only=False: our own checkpoints contain numpy arrays in `metrics`
    # (confusion matrix etc.), which the default safe loader rejects.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint if available
    model_config = checkpoint.get('config', config)
    
    # Create model
    model = create_model(model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Model info
    info = get_model_info(model)
    print(f"Model: {model_config['model']['name']}")
    print(f"Parameters: {info['total_params']:,}")
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        val_metrics = checkpoint['metrics']
        if 'macro_recall' in val_metrics:
            print(f"Validation macro recall: {val_metrics['macro_recall']:.4f}")
    print()
    
    # Set to eval mode
    model.eval()
    
    # Move to GPU
    model = model.to(device)
    
    return model


def get_target_layer(model: nn.Module, model_name: str) -> List[nn.Module]:
    """
    Determine the target layer for Grad-CAM based on model architecture.
    
    Args:
        model: Model to extract target layer from
        model_name: Model architecture name
        
    Returns:
        List containing the target layer(s)
        
    Raises:
        ValueError: If model architecture is not supported
    """
    print("=" * 70)
    print("Determining Target Layer")
    print("=" * 70)
    
    target_layers = None
    
    # EfficientNet
    if 'efficientnet' in model_name.lower():
        # Try blocks[-1] first, fall back to conv_head
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            target_layers = [model.blocks[-1]]
            layer_name = "blocks[-1]"
        elif hasattr(model, 'conv_head'):
            target_layers = [model.conv_head]
            layer_name = "conv_head"
        else:
            raise ValueError(f"Cannot find target layer for EfficientNet model")
    
    # ResNet
    elif 'resnet' in model_name.lower():
        if hasattr(model, 'layer4'):
            target_layers = [model.layer4[-1]]
            layer_name = "layer4[-1]"
        else:
            raise ValueError(f"Cannot find target layer for ResNet model")
    
    # ConvNeXt
    elif 'convnext' in model_name.lower():
        if hasattr(model, 'stages') and len(model.stages) > 0:
            target_layers = [model.stages[-1]]
            layer_name = "stages[-1]"
        else:
            raise ValueError(f"Cannot find target layer for ConvNeXt model")
    
    else:
        raise ValueError(
            f"Unsupported model architecture: {model_name}. "
            f"Supported: EfficientNet, ResNet, ConvNeXt"
        )
    
    print(f"Model: {model_name}")
    print(f"Target layer: {layer_name}")
    print()
    
    return target_layers


def create_cam_object(
    model: nn.Module,
    target_layers: List[nn.Module],
    method: str
):
    """
    Create Grad-CAM object based on specified method.
    
    Args:
        model: Model to visualize
        target_layers: Target layers for CAM
        method: CAM method ('gradcam', 'gradcam++', 'scorecam')
        
    Returns:
        CAM object
    """
    if method == 'gradcam':
        cam = GradCAM(model=model, target_layers=target_layers)
    elif method == 'gradcam++':
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    elif method == 'scorecam':
        cam = ScoreCAM(model=model, target_layers=target_layers)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return cam


def generate_heatmap(
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    cam_object,
    target_class: int,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM heatmap for an image.
    
    Args:
        image_tensor: Normalized image tensor (C, H, W)
        original_image: Original image for visualization (H, W, 3), range [0, 1]
        cam_object: Grad-CAM object
        target_class: Target class index for visualization
        device: Device to run on
        
    Returns:
        Tuple of (heatmap, overlay_image)
    """
    # Add batch dimension and move to device
    input_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Generate CAM
    grayscale_cam = cam_object(
        input_tensor=input_tensor,
        targets=None,  # Use predicted class (or specify target_class if needed)
        aug_smooth=False,
        eigen_smooth=False
    )
    
    # Get the heatmap for the first (and only) image in the batch
    grayscale_cam = grayscale_cam[0, :]
    
    # Create overlay visualization
    overlay = show_cam_on_image(
        original_image,
        grayscale_cam,
        use_rgb=True,
        colormap=cv2.COLORMAP_JET,
        image_weight=0.6  # 0.6 alpha blending for original image
    )
    
    # Convert grayscale heatmap to color for visualization
    heatmap_colored = cv2.applyColorMap(
        (grayscale_cam * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    return heatmap_colored, overlay


def save_heatmap_visualization(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    class_name: str,
    confidence: float,
    filename: str,
    save_path: Path
):
    """
    Create and save a visualization with original, heatmap, and overlay.
    
    Args:
        original_image: Original image (H, W, 3), range [0, 1]
        heatmap: Colored heatmap (H, W, 3)
        overlay: Overlay image (H, W, 3)
        class_name: Predicted class name
        confidence: Prediction confidence
        filename: Original filename for title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add prediction info
    fig.suptitle(
        f'{filename}\nPrediction: {class_name} (confidence: {confidence:.3f})',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyze_activation_focus(
    grayscale_cam: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Analyze where the model is focusing (center vs margins).
    
    Args:
        grayscale_cam: Grayscale CAM heatmap (H, W), range [0, 1]
        threshold: Threshold for high activation
        
    Returns:
        Dictionary with focus metrics
    """
    h, w = grayscale_cam.shape
    
    # Define center region (middle 50% of image)
    center_h_start = h // 4
    center_h_end = 3 * h // 4
    center_w_start = w // 4
    center_w_end = 3 * w // 4
    
    # Extract center and margin regions
    center_region = grayscale_cam[center_h_start:center_h_end, center_w_start:center_w_end]
    
    # Create margin mask
    margin_mask = np.ones_like(grayscale_cam, dtype=bool)
    margin_mask[center_h_start:center_h_end, center_w_start:center_w_end] = False
    margin_region = grayscale_cam[margin_mask]
    
    # Compute statistics
    center_mean = float(np.mean(center_region))
    margin_mean = float(np.mean(margin_region))
    max_activation = float(np.max(grayscale_cam))
    
    # Find location of max activation
    max_loc = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
    max_in_center = (
        center_h_start <= max_loc[0] < center_h_end and
        center_w_start <= max_loc[1] < center_w_end
    )
    
    return {
        'center_mean': center_mean,
        'margin_mean': margin_mean,
        'max_activation': max_activation,
        'max_in_center': max_in_center,
        'focus_ratio': center_mean / (margin_mean + 1e-8)  # Avoid division by zero
    }


def generate_all_heatmaps(
    model: nn.Module,
    sampled_df: pd.DataFrame,
    processed_dir: Path,
    gradcam_dir: Path,
    cam_object,
    class_names: List[str],
    device: str,
    config: dict
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate heatmaps for all sampled images.
    
    Args:
        model: Model for inference
        sampled_df: DataFrame with sampled images
        processed_dir: Directory with processed data
        gradcam_dir: Directory to save heatmaps
        cam_object: Grad-CAM object
        class_names: List of class names
        device: Device to run on
        config: Configuration dictionary
        
    Returns:
        Tuple of (heatmap_info_list, suspicious_cases_list)
    """
    print("=" * 70)
    print("Generating Heatmaps")
    print("=" * 70)
    
    val_transform = get_val_transforms()
    heatmap_info = []
    suspicious_cases = []
    
    channels_last = config['training'].get('channels_last', False)
    use_amp = config['training']['use_amp']
    amp_dtype = torch.float16 if config['training']['amp_dtype'] == 'float16' else torch.bfloat16
    
    # Progress bar
    pbar = tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Processing images")
    
    for idx, row in pbar:
        filepath = processed_dir / row['filepath']
        true_class = row['class']
        
        # Load original image (before normalization) for visualization
        raw_image = np.load(filepath)  # uint8, exactly as the training pipeline feeds it
        image_np = raw_image.astype(np.float32) / 255.0  # Range [0, 1] copy for visualization

        # Apply transforms for model input
        # NOTE: pass the uint8 array. ToPILImage assumes float inputs are in [0,1]
        # and rescales them; feeding it [0,255] floats saturates all pixels to white.
        image_tensor = val_transform(raw_image)
        
        # Move to device with a batch dimension, then channels_last
        # (channels_last requires rank-4 tensors; a single image is rank-3)
        input_tensor = image_tensor.unsqueeze(0).to(device)
        if channels_last:
            input_tensor = input_tensor.to(memory_format=torch.channels_last)
        
        # Get prediction
        with autocast(enabled=use_amp, dtype=amp_dtype):
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = outputs.argmax(dim=1).item()
                confidence = probabilities[0, predicted_class].item()
        
        predicted_class_name = class_names[predicted_class]
        
        # Generate Grad-CAM heatmap
        grayscale_cam = cam_object(
            input_tensor=input_tensor,
            targets=None,
            aug_smooth=False,
            eigen_smooth=False
        )[0, :]
        
        # Create overlay visualization
        overlay = show_cam_on_image(
            image_np,
            grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
            image_weight=0.6
        )
        
        # Convert grayscale heatmap to color
        heatmap_colored = cv2.applyColorMap(
            (grayscale_cam * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Analyze activation focus
        focus_analysis = analyze_activation_focus(grayscale_cam)
        
        # Save individual heatmap
        filename = Path(filepath).stem
        save_path = gradcam_dir / true_class / f"{filename}_gradcam.png"
        
        save_heatmap_visualization(
            image_np,
            heatmap_colored,
            overlay,
            predicted_class_name,
            confidence,
            filename,
            save_path
        )
        
        # Store info
        info = {
            'filepath': str(filepath),
            'filename': filename,
            'true_class': true_class,
            'predicted_class': predicted_class_name,
            'confidence': confidence,
            'correct': true_class == predicted_class_name,
            'save_path': str(save_path),
            'focus_analysis': focus_analysis
        }
        heatmap_info.append(info)
        
        # Flag suspicious cases (low center activation)
        if focus_analysis['focus_ratio'] < 1.0:  # Margin activation > center activation
            suspicious_cases.append({
                'filepath': str(filepath),
                'class': true_class,
                'predicted': predicted_class_name,
                'focus_ratio': focus_analysis['focus_ratio'],
                'center_mean': focus_analysis['center_mean'],
                'margin_mean': focus_analysis['margin_mean']
            })
    
    print(f"\nGenerated {len(heatmap_info)} heatmaps")
    print(f"Found {len(suspicious_cases)} suspicious cases (low center activation)")
    print()
    
    return heatmap_info, suspicious_cases


def create_class_gallery(
    heatmap_info: List[Dict],
    class_name: str,
    gradcam_dir: Path,
    num_samples: int
):
    """
    Create a gallery of heatmaps for a single class.
    
    Args:
        heatmap_info: List of heatmap information dictionaries
        class_name: Class name to create gallery for
        gradcam_dir: Directory containing heatmaps
        num_samples: Number of samples expected per class
    """
    # Filter for this class
    class_info = [info for info in heatmap_info if info['true_class'] == class_name]
    
    if not class_info:
        return
    
    n_samples = len(class_info)
    
    # Create figure: n_samples rows × 3 columns
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    
    # Handle case of single sample
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, info in enumerate(class_info):
        # Load original image
        image_np = np.load(info['filepath']).astype(np.float32) / 255.0
        
        # Load saved visualization (we'll recreate from source)
        # Actually, let's load the components
        heatmap_path = Path(info['save_path'])
        
        # We need to regenerate these since we only saved the combined visualization
        # For simplicity, just load the saved figure
        # Actually, let's load the original images and recreate
        
        # Load original
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Original\n{info['filename']}", fontsize=10)
        axes[i, 0].axis('off')
        
        # Load the saved heatmap visualization and extract components
        # For simplicity, we'll load the full saved image
        saved_img = plt.imread(heatmap_path)
        axes[i, 1].imshow(saved_img)
        axes[i, 1].set_title(
            f"Pred: {info['predicted_class']}\nConf: {info['confidence']:.3f}",
            fontsize=10
        )
        axes[i, 1].axis('off')
        
        # Add text info in third column
        axes[i, 2].axis('off')
        info_text = (
            f"File: {info['filename']}\n"
            f"True: {info['true_class']}\n"
            f"Predicted: {info['predicted_class']}\n"
            f"Confidence: {info['confidence']:.3f}\n"
            f"Correct: {info['correct']}\n\n"
            f"Focus Analysis:\n"
            f"  Center: {info['focus_analysis']['center_mean']:.3f}\n"
            f"  Margin: {info['focus_analysis']['margin_mean']:.3f}\n"
            f"  Ratio: {info['focus_analysis']['focus_ratio']:.3f}\n"
            f"  Max in center: {info['focus_analysis']['max_in_center']}"
        )
        axes[i, 2].text(
            0.1, 0.5, info_text,
            fontsize=10,
            verticalalignment='center',
            family='monospace'
        )
    
    fig.suptitle(f"Grad-CAM Gallery - {class_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save gallery
    gallery_path = gradcam_dir / f"{class_name}_gallery.png"
    plt.savefig(gallery_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved gallery for {class_name}: {gallery_path}")


def create_master_gallery(
    heatmap_info: List[Dict],
    class_names: List[str],
    gradcam_dir: Path,
    num_samples: int
):
    """
    Create a master gallery with samples from all classes.
    
    Args:
        heatmap_info: List of heatmap information dictionaries
        class_names: List of all class names
        gradcam_dir: Directory containing heatmaps
        num_samples: Number of samples per class
    """
    print("=" * 70)
    print("Creating Galleries")
    print("=" * 70)
    
    # Create class galleries
    for class_name in class_names:
        create_class_gallery(heatmap_info, class_name, gradcam_dir, num_samples)
    
    # Create master gallery: select first sample from each class
    fig, axes = plt.subplots(len(class_names), 3, figsize=(15, 5 * len(class_names)))
    
    # Handle case of single class
    if len(class_names) == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_name in enumerate(class_names):
        # Get first sample for this class
        class_info = [info for info in heatmap_info if info['true_class'] == class_name]
        
        if not class_info:
            # No samples for this class
            for j in range(3):
                axes[i, j].axis('off')
                axes[i, j].text(0.5, 0.5, f"No samples\nfor {class_name}",
                              ha='center', va='center')
            continue
        
        info = class_info[0]  # First sample
        
        # Load original image
        image_np = np.load(info['filepath']).astype(np.float32) / 255.0
        
        # Original
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"{class_name}\n{info['filename']}", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Saved heatmap
        saved_img = plt.imread(info['save_path'])
        axes[i, 1].imshow(saved_img)
        axes[i, 1].set_title(
            f"Pred: {info['predicted_class']}\nConf: {info['confidence']:.3f}",
            fontsize=10
        )
        axes[i, 1].axis('off')
        
        # Info
        axes[i, 2].axis('off')
        info_text = (
            f"True: {info['true_class']}\n"
            f"Pred: {info['predicted_class']}\n"
            f"Conf: {info['confidence']:.3f}\n"
            f"Correct: {info['correct']}\n\n"
            f"Focus ratio: {info['focus_analysis']['focus_ratio']:.3f}"
        )
        axes[i, 2].text(
            0.1, 0.5, info_text,
            fontsize=10,
            verticalalignment='center',
            family='monospace'
        )
    
    fig.suptitle("Grad-CAM Master Gallery - All Classes", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save master gallery
    master_path = gradcam_dir.parent / "gradcam_gallery.png"
    plt.savefig(master_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved master gallery: {master_path}")
    print()


def save_analysis_report(
    heatmap_info: List[Dict],
    suspicious_cases: List[Dict],
    class_names: List[str],
    reports_dir: Path
):
    """
    Save analysis report with statistics and suspicious cases.
    
    Args:
        heatmap_info: List of heatmap information dictionaries
        suspicious_cases: List of suspicious case dictionaries
        class_names: List of class names
        reports_dir: Directory to save report
    """
    print("=" * 70)
    print("Saving Analysis Report")
    print("=" * 70)
    
    report_path = reports_dir / "gradcam" / "gradcam_analysis.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Grad-CAM Explainability Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total heatmaps generated: {len(heatmap_info)}\n")
        f.write(f"Suspicious cases (low center activation): {len(suspicious_cases)}\n\n")
        
        # Per-class statistics
        f.write("PER-CLASS HEATMAP COUNT\n")
        f.write("-" * 70 + "\n")
        for class_name in class_names:
            count = sum(1 for info in heatmap_info if info['true_class'] == class_name)
            correct = sum(1 for info in heatmap_info 
                         if info['true_class'] == class_name and info['correct'])
            accuracy = correct / count * 100 if count > 0 else 0
            f.write(f"  {class_name:12s}: {count:3d} heatmaps "
                   f"({correct}/{count} correct, {accuracy:.1f}%)\n")
        f.write("\n")
        
        # Focus analysis statistics
        f.write("FOCUS ANALYSIS STATISTICS\n")
        f.write("-" * 70 + "\n")
        focus_ratios = [info['focus_analysis']['focus_ratio'] for info in heatmap_info]
        f.write(f"  Mean focus ratio: {np.mean(focus_ratios):.3f}\n")
        f.write(f"  Median focus ratio: {np.median(focus_ratios):.3f}\n")
        f.write(f"  Min focus ratio: {np.min(focus_ratios):.3f}\n")
        f.write(f"  Max focus ratio: {np.max(focus_ratios):.3f}\n")
        f.write(f"  Std focus ratio: {np.std(focus_ratios):.3f}\n\n")
        
        # Suspicious cases
        if suspicious_cases:
            f.write("SUSPICIOUS CASES (Low Center Activation)\n")
            f.write("-" * 70 + "\n")
            f.write("Images where the model focuses more on margins than center.\n")
            f.write("This may indicate the model is using artifacts or background features.\n\n")
            
            # Sort by focus ratio (lowest first)
            suspicious_cases_sorted = sorted(suspicious_cases, key=lambda x: x['focus_ratio'])
            
            for i, case in enumerate(suspicious_cases_sorted, 1):
                f.write(f"{i}. {case['filepath']}\n")
                f.write(f"   Class: {case['class']}\n")
                f.write(f"   Predicted: {case['predicted']}\n")
                f.write(f"   Focus ratio: {case['focus_ratio']:.3f}\n")
                f.write(f"   Center activation: {case['center_mean']:.3f}\n")
                f.write(f"   Margin activation: {case['margin_mean']:.3f}\n")
                f.write("\n")
        else:
            f.write("SUSPICIOUS CASES\n")
            f.write("-" * 70 + "\n")
            f.write("No suspicious cases found. All models show good center focus.\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")
        if suspicious_cases:
            f.write("1. Review suspicious cases to check for artifacts or biases\n")
            f.write("2. Consider additional preprocessing to remove margins\n")
            f.write("3. Augment training data with better margin handling\n")
        else:
            f.write("1. Model shows good focus on center regions\n")
            f.write("2. Continue monitoring with more samples if needed\n")
        f.write("\n")
    
    print(f"Saved analysis report: {report_path}")
    print()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    device, processed_dir, reports_dir, gradcam_dir = setup_explainability(config)
    
    # Load manifest and sample images
    manifest_path = processed_dir / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    
    class_names = config['data']['classes']
    sampled_df = sample_images_per_class(
        manifest,
        args.split,
        args.num_samples,
        class_names
    )
    
    # Load model
    model = load_model_and_checkpoint(config, args.checkpoint, device)
    
    # Get target layer
    model_name = config['model']['name']
    target_layers = get_target_layer(model, model_name)
    
    # Create Grad-CAM object
    print("=" * 70)
    print(f"Creating Grad-CAM Object (method: {args.method})")
    print("=" * 70)
    cam_object = create_cam_object(model, target_layers, args.method)
    print(f"Method: {args.method}")
    print()
    
    # Generate heatmaps
    heatmap_info, suspicious_cases = generate_all_heatmaps(
        model,
        sampled_df,
        processed_dir,
        gradcam_dir,
        cam_object,
        class_names,
        device,
        config
    )
    
    # Create galleries
    create_master_gallery(heatmap_info, class_names, gradcam_dir, args.num_samples)
    
    # Save analysis report
    save_analysis_report(heatmap_info, suspicious_cases, class_names, reports_dir)
    
    print("=" * 70)
    print("Explainability Analysis Complete")
    print("=" * 70)
    print(f"Heatmaps saved to: {gradcam_dir}")
    print(f"Galleries saved to: {reports_dir}")
    print(f"Analysis report: {reports_dir / 'gradcam' / 'gradcam_analysis.txt'}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
