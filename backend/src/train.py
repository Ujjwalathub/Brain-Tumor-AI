"""
Training Script - Stage 6
Two-stage training with mixed precision and GPU optimizations for brain tumor classification.

Hardware: RTX 3050 6GB, CUDA-only training
Training stages:
    1. Head-only: Freeze backbone, train classifier (stage1_epochs)
    2. Full fine-tune: Unfreeze all, train entire model (remaining epochs)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import BrainTumorDataset, CLASS_TO_IDX, IDX_TO_CLASS
from data.transforms import get_train_transforms, get_val_transforms
from models import create_model, freeze_backbone, unfreeze_all, get_model_info
from utils import load_config, set_seed, compute_metrics, get_lr, format_time


class SafeSummaryWriter:
    """
    TensorBoard writer wrapper that never raises on logging failures.

    Rationale (learned the hard way): an externally deleted events file used to
    abort a multi-hour training run mid-epoch. Logging must be best-effort;
    model checkpoints are the ground truth, not the event log.
    """

    def __init__(self, writer):
        self._writer = writer

    def __getattr__(self, name):
        attr = getattr(self._writer, name)

        if callable(attr):
            def safe_method(*args, **kwargs):
                try:
                    return attr(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - logging must not kill training
                    print(f"[warn] TensorBoard '{name}' failed: {exc}")
            return safe_method

        return attr

    def close(self):
        try:
            self._writer.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] TensorBoard close failed: {exc}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train brain tumor classifier with two-stage training and AMP"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/efficientnet_b0.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (optional)"
    )
    return parser.parse_args()


def setup_training(config: dict) -> Tuple[str, Path, Path, SummaryWriter]:
    """
    Setup training environment: CUDA check, seeds, directories, logging.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (device, checkpoint_dir, reports_dir, tensorboard_writer)
    """
    # Assert CUDA availability
    assert torch.cuda.is_available(), "CUDA GPU required. Aborting."
    device = "cuda"
    
    print("=" * 70)
    print("Brain Tumor MRI Classifier - Two-Stage Training")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    # Set seeds for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Enable cuDNN benchmark for faster training
    torch.backends.cudnn.benchmark = True
    print("cuDNN benchmark: enabled")
    print()
    
    # Create directories
    workspace_root = Path.cwd()
    checkpoint_dir = workspace_root / config['logging']['checkpoint_dir']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    reports_dir = workspace_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard logging
    tensorboard_dir = workspace_root / config['logging']['tensorboard_dir']
    experiment_name = config.get('experiment_name', 'experiment')
    log_path = tensorboard_dir / experiment_name
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_path))
    # Wrap so that a deleted/locked events file cannot abort training.
    writer = SafeSummaryWriter(writer)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"TensorBoard logs: {log_path}")
    print(f"Reports directory: {reports_dir}")
    print()
    
    return device, checkpoint_dir, reports_dir, writer


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, dataset_info)
    """
    print("=" * 70)
    print("Loading Data")
    print("=" * 70)
    
    workspace_root = Path.cwd()
    processed_dir = workspace_root / config['data']['processed_dir']
    manifest_path = processed_dir / "manifest.csv"
    
    # Load manifest
    print(f"Loading manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    
    # Split into train and val
    train_df = manifest[manifest['split'] == 'train'].copy()
    val_df = manifest[manifest['split'] == 'val'].copy()
    
    print(f"Total samples: {len(manifest)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print()
    
    # Class distribution
    print("Training class distribution:")
    train_class_counts = train_df['class'].value_counts()
    for class_name, count in train_class_counts.items():
        print(f"  {class_name:12s}: {count:4d} ({count/len(train_df)*100:.1f}%)")
    print()
    
    print("Validation class distribution:")
    val_class_counts = val_df['class'].value_counts()
    for class_name, count in val_class_counts.items():
        print(f"  {class_name:12s}: {count:4d} ({count/len(val_df)*100:.1f}%)")
    print()
    
    # Get transforms
    train_transform = get_train_transforms(config.get('augmentation', {}))
    val_transform = get_val_transforms()
    
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
    
    # DataLoader settings
    batch_size = config['dataloader']['batch_size']
    num_workers = config['dataloader']['num_workers']
    pin_memory = config['dataloader']['pin_memory']
    persistent_workers = config['dataloader'].get('persistent_workers', True)
    prefetch_factor = config['dataloader'].get('prefetch_factor', 2)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    print(f"Batch size (train): {batch_size}")
    print(f"Batch size (val): {config['evaluation']['batch_size']}")
    print(f"Gradient accumulation steps: {config['training']['grad_accumulation_steps']}")
    print(f"Effective batch size: {batch_size * config['training']['grad_accumulation_steps']}")
    print(f"DataLoader workers: {num_workers}")
    print()
    
    dataset_info = {
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'train_class_counts': train_class_counts.to_dict(),
        'val_class_counts': val_class_counts.to_dict(),
    }
    
    return train_loader, val_loader, dataset_info


def setup_model(config: dict, device: str) -> nn.Module:
    """
    Create and setup model for training.
    
    Args:
        config: Configuration dictionary
        device: Device to move model to
        
    Returns:
        Model ready for training
    """
    print("=" * 70)
    print("Creating Model")
    print("=" * 70)
    
    # Create model
    model = create_model(config)
    
    # Get model info
    info = get_model_info(model)
    print(f"Model: {config['model']['name']}")
    print(f"Pretrained: {config['model']['pretrained']}")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    print()
    
    # Move to GPU
    model = model.to(device)
    
    # Convert to channels_last if configured
    if config['training'].get('channels_last', False):
        model = model.to(memory_format=torch.channels_last)
        print("Memory format: channels_last")
    else:
        print("Memory format: contiguous")
    
    print()
    
    return model


def create_optimizer_scheduler(
    model: nn.Module,
    config: dict,
    steps_per_epoch: int,
    stage: int
) -> Tuple[AdamW, SequentialLR]:
    """
    Create optimizer and learning rate scheduler for a training stage.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        steps_per_epoch: Number of training steps per epoch
        stage: Training stage (1 or 2)
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Select learning rate based on stage
    if stage == 1:
        lr = config['training']['stage1_lr']
        total_epochs = config['training']['stage1_epochs']
        stage_name = "Stage 1 (Head-only)"
    else:
        lr = config['training']['stage2_lr']
        total_epochs = config['training']['epochs'] - config['training']['stage1_epochs']
        stage_name = "Stage 2 (Full fine-tune)"
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler: Linear warmup + Cosine annealing
    warmup_epochs = config['training']['warmup_epochs']
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    # Linear warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config['scheduler']['min_lr']
    )
    
    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    print(f"{stage_name}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {config['training']['weight_decay']}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Total epochs: {total_epochs}")
    print()
    
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scheduler: SequentialLR,
    scaler: GradScaler,
    device: str,
    config: dict,
    epoch: int,
    global_step: int,
    writer: SummaryWriter
) -> Tuple[float, float, int]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        config: Configuration dictionary
        epoch: Current epoch number
        global_step: Current global step
        writer: TensorBoard writer
        
    Returns:
        Tuple of (avg_loss, avg_accuracy, new_global_step)
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    grad_accumulation_steps = config['training']['grad_accumulation_steps']
    use_amp = config['training']['use_amp']
    amp_dtype = torch.float16 if config['training']['amp_dtype'] == 'float16' else torch.bfloat16
    channels_last = config['training'].get('channels_last', False)
    log_every_n_steps = config['logging']['log_every_n_steps']
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Convert to channels_last if configured
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        
        # Mixed precision forward pass
        with autocast(enabled=use_amp, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / grad_accumulation_steps  # Scale loss for accumulation
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Scheduler step (per iteration)
            scheduler.step()
            
            global_step += 1
            
            # Logging
            if global_step % log_every_n_steps == 0:
                current_lr = get_lr(optimizer)
                writer.add_scalar('Train/LearningRate', current_lr, global_step)
                writer.add_scalar('Train/Loss', loss.item() * grad_accumulation_steps, global_step)
        
        # Statistics
        running_loss += loss.item() * grad_accumulation_steps * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        current_loss = running_loss / total
        current_acc = 100.0 * correct / total
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%',
            'lr': f'{get_lr(optimizer):.6f}'
        })
    
    avg_loss = running_loss / total
    avg_accuracy = 100.0 * correct / total
    
    return avg_loss, avg_accuracy, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    config: dict,
    class_names: list
) -> Dict:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        config: Configuration dictionary
        class_names: List of class names
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    channels_last = config['training'].get('channels_last', False)
    use_amp = config['training']['use_amp']
    amp_dtype = torch.float16 if config['training']['amp_dtype'] == 'float16' else torch.bfloat16
    
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    
    for images, labels in pbar:
        # Move to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Convert to channels_last if configured
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        
        # Mixed precision forward pass
        with autocast(enabled=use_amp, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        
        # Collect predictions
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
    
    # Compute metrics
    avg_loss = running_loss / len(val_loader.dataset)
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)
    
    metrics = compute_metrics(y_true, y_pred, y_proba, class_names)
    metrics['loss'] = avg_loss
    
    return metrics


def log_validation_metrics(
    metrics: Dict,
    epoch: int,
    global_step: int,
    writer: SummaryWriter,
    stage: str = ""
):
    """
    Log validation metrics to console and TensorBoard.
    
    Args:
        metrics: Dictionary with validation metrics
        epoch: Current epoch
        global_step: Current global step
        writer: TensorBoard writer
        stage: Stage name for logging (optional)
    """
    prefix = f"{stage} - " if stage else ""
    
    print(f"\n{prefix}Validation Results (Epoch {epoch}):")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Macro Recall: {metrics['macro_recall']*100:.2f}%")
    print()
    print("  Per-class metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"    {class_name:12s}: "
              f"P={class_metrics['precision']:.3f}, "
              f"R={class_metrics['recall']:.3f}, "
              f"F1={class_metrics['f1']:.3f}, "
              f"N={class_metrics['support']}")
    print()
    
    # TensorBoard logging
    writer.add_scalar('Val/Loss', metrics['loss'], global_step)
    writer.add_scalar('Val/Accuracy', metrics['accuracy'], global_step)
    writer.add_scalar('Val/MacroRecall', metrics['macro_recall'], global_step)
    
    # Per-class metrics
    for class_name, class_metrics in metrics['per_class'].items():
        writer.add_scalar(f'Val/Precision_{class_name}', class_metrics['precision'], global_step)
        writer.add_scalar(f'Val/Recall_{class_name}', class_metrics['recall'], global_step)
        writer.add_scalar(f'Val/F1_{class_name}', class_metrics['f1'], global_step)
    
    # Confusion matrix as image
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        metrics['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=list(metrics['per_class'].keys()),
        yticklabels=list(metrics['per_class'].keys()),
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - Epoch {epoch}')
    
    writer.add_figure('Val/ConfusionMatrix', fig, global_step)
    plt.close(fig)


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: SequentialLR,
    scaler: GradScaler,
    config: dict,
    metrics: Dict,
    checkpoint_dir: Path,
    filename: str
):
    """
    Save training checkpoint.
    
    Args:
        epoch: Current epoch
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        scaler: Gradient scaler state
        config: Configuration dictionary
        metrics: Validation metrics
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': config,
        'metrics': metrics
    }
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: AdamW = None,
    scheduler: SequentialLR = None,
    scaler: GradScaler = None
) -> Tuple[int, Dict]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        scaler: Gradient scaler to load state into (optional)
        
    Returns:
        Tuple of (start_epoch, metrics)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    # weights_only=False: our own checkpoints contain numpy arrays in `metrics`
    # (confusion matrix etc.), which the default safe loader rejects.
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    metrics = checkpoint.get('metrics', {})
    
    print(f"Resumed from epoch {checkpoint['epoch']}")
    return start_epoch, metrics


def train(config: dict, resume_checkpoint: str = None):
    """
    Main training function with two-stage training.
    
    Args:
        config: Configuration dictionary
        resume_checkpoint: Path to checkpoint to resume from (optional)
    """
    # Setup
    device, checkpoint_dir, reports_dir, writer = setup_training(config)
    
    # Data
    train_loader, val_loader, dataset_info = create_dataloaders(config)
    
    # Model
    model = setup_model(config, device)
    
    # Loss function with label smoothing.
    # Optional per-class weights (training.class_weights in the YAML, order =
    # data.classes, e.g. glioma/meningioma/notumor/pituitary) were added to
    # close the glioma-recall gap: augaffine ep13 test macro-recall 0.9619
    # with glioma recall 0.858 vs meningioma 0.990 (47/61 misses are
    # glioma->meningioma). Absolute weights are normalized by CrossEntropy,
    # so a moderate 1.3 pull on glioma is a safe first step.
    class_weights = config['training'].get('class_weights')
    if class_weights:
        weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(
            label_smoothing=config['training']['label_smoothing'],
            weight=weights,
        ).to(device)
        print(f"  Class weights: {class_weights} (data.classes order)")
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=config['training']['label_smoothing']
        ).to(device)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training']['use_amp'])
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_macro_recall': [],
        'stage': []
    }
    
    # Best model tracking
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    saved_checkpoints = []
    
    class_names = config['data']['classes']
    
    # Total epochs
    stage1_epochs = config['training']['stage1_epochs']
    total_epochs = config['training']['epochs']
    early_stopping_patience = config['training']['early_stopping_patience']
    
    # Start training time
    training_start_time = time.time()
    
    # Resume from checkpoint if provided
    start_epoch = 1
    if resume_checkpoint is not None:
        start_epoch, _ = load_checkpoint(resume_checkpoint, model)
        print()
    
    # ========================================================================
    # STAGE 1: Head-only training (freeze backbone)
    # ========================================================================
    if start_epoch <= stage1_epochs:
        print("=" * 70)
        print("STAGE 1: Head-Only Training (Frozen Backbone)")
        print("=" * 70)
        print()
        
        # Freeze backbone
        model = freeze_backbone(model)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters (head only): {trainable_params:,}")
        print()
        
        # Create optimizer and scheduler for stage 1
        steps_per_epoch = len(train_loader) // config['training']['grad_accumulation_steps']
        optimizer, scheduler = create_optimizer_scheduler(model, config, steps_per_epoch, stage=1)
        
        global_step = 0
        
        # Train stage 1
        for epoch in range(start_epoch, stage1_epochs + 1):
            # Train
            train_loss, train_acc, global_step = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler,
                device, config, epoch, global_step, writer
            )
            
            # Validate
            val_metrics = validate(model, val_loader, criterion, device, config, class_names)
            
            # Log
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Accuracy', train_acc, epoch)
            log_validation_metrics(val_metrics, epoch, global_step, writer, stage="Stage 1")
            
            # Track history
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_macro_recall'].append(val_metrics['macro_recall'])
            history['stage'].append(1)
            
            # GPU memory stats
            max_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"GPU max memory: {max_memory_mb:.2f} MB")
            torch.cuda.reset_peak_memory_stats()

            # Per-epoch checkpoint so interruption resumes from the last epoch
            save_checkpoint(
                epoch, model, optimizer, scheduler, scaler, config, val_metrics,
                checkpoint_dir, "last.pth"
            )
            print()

        # Save stage 1 checkpoint
        save_checkpoint(
            stage1_epochs, model, optimizer, scheduler, scaler, config, val_metrics,
            checkpoint_dir, f"stage1_final_epoch{stage1_epochs:03d}.pth"
        )
        print()
    
    # ========================================================================
    # STAGE 2: Full fine-tuning (unfreeze all)
    # ========================================================================
    print("=" * 70)
    print("STAGE 2: Full Fine-Tuning (Unfrozen Backbone)")
    print("=" * 70)
    print()
    
    # Unfreeze all parameters
    model = unfreeze_all(model)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (all): {trainable_params:,}")
    print()
    
    # Create optimizer and scheduler for stage 2
    stage2_start_epoch = max(stage1_epochs + 1, start_epoch)
    stage2_epochs = total_epochs - stage1_epochs
    steps_per_epoch = len(train_loader) // config['training']['grad_accumulation_steps']
    optimizer, scheduler = create_optimizer_scheduler(model, config, steps_per_epoch, stage=2)
    
    global_step = 0
    
    # Train stage 2
    for epoch in range(stage2_start_epoch, total_epochs + 1):
        # Train
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, config, epoch, global_step, writer
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config, class_names)
        
        # Log
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        log_validation_metrics(val_metrics, epoch, global_step, writer, stage="Stage 2")
        
        # Track history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_macro_recall'].append(val_metrics['macro_recall'])
        history['stage'].append(2)
        
        # GPU memory stats
        max_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"GPU max memory: {max_memory_mb:.2f} MB")
        torch.cuda.reset_peak_memory_stats()

        # Per-epoch checkpoint so interruption resumes from the last epoch
        save_checkpoint(
            epoch, model, optimizer, scheduler, scaler, config, val_metrics,
            checkpoint_dir, "last.pth"
        )

        # Check for best model
        current_metric = val_metrics['macro_recall']
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            
            # Save best checkpoint
            filename = f"best_epoch{epoch:03d}_macro-recall{current_metric:.4f}.pth"
            save_checkpoint(
                epoch, model, optimizer, scheduler, scaler, config, val_metrics,
                checkpoint_dir, filename
            )
            
            # Track saved checkpoints
            saved_checkpoints.append((current_metric, checkpoint_dir / filename))
            saved_checkpoints.sort(reverse=True, key=lambda x: x[0])
            
            # Keep only top_k checkpoints
            top_k = config['logging']['save_top_k']
            if len(saved_checkpoints) > top_k:
                _, old_checkpoint = saved_checkpoints.pop()
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    print(f"Removed old checkpoint: {old_checkpoint}")
            
            print(f"New best macro recall: {current_metric:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
        
        print()
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
            break
    
    # Training complete
    training_time = time.time() - training_start_time
    
    print("=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Total training time: {format_time(training_time)}")
    print(f"Best validation macro recall: {best_metric:.4f} (epoch {best_epoch})")
    print()
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = reports_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")
    
    # Final GPU memory stats
    print(f"\nFinal GPU memory stats:")
    print(f"  Max allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
    print(f"  Max reserved: {torch.cuda.max_memory_reserved() / (1024**2):.2f} MB")
    
    # Close TensorBoard writer
    writer.close()
    
    print("\nDone!")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train(config, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
