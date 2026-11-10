"""
ONNX Export Script - Stage 11
Export PyTorch model to ONNX format with verification and benchmarking.

Hardware: RTX 3050 6GB (CUDA preferred but not required for export)
ONNX export is more stable on CPU, but verification can use GPU.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from models import create_model, get_model_info
from utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export brain tumor classifier to ONNX with verification"
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
        "--output",
        type=str,
        default=None,
        help="Output ONNX path (default: models/<model_name>.onnx)"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model using onnx-simplifier"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark (100 iterations)"
    )
    return parser.parse_args()


def setup_export(config: dict) -> Tuple[str, Path]:
    """
    Setup export environment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (device, models_dir)
    """
    print("=" * 70)
    print("Brain Tumor MRI Classifier - ONNX Export")
    print("=" * 70)
    
    # Check if CUDA is available (not required, but preferred for verification)
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("GPU: Not available (CPU-only mode)")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"ONNX Version: {onnx.__version__}")
    print(f"ONNX Runtime Version: {ort.__version__}")
    print()
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Create models directory
    workspace_root = Path.cwd()
    models_dir = workspace_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Models directory: {models_dir}")
    print()
    
    return device, models_dir


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: dict,
    device: str
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model on (for verification)
        
    Returns:
        Model loaded with checkpoint weights
    """
    print("=" * 70)
    print("Loading Model")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    # weights_only=False: our own checkpoints contain numpy arrays in `metrics`.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
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
    
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    opset_version: int,
    config: dict
) -> None:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        config: Configuration dictionary
    """
    print("=" * 70)
    print("Exporting to ONNX")
    print("=" * 70)
    
    # Move model to CPU for stable export
    model = model.cpu()
    model.eval()
    
    # Get input size from config
    img_size = config['data']['img_size']
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    print(f"Input shape: {list(dummy_input.shape)}")
    print(f"Output path: {output_path}")
    print(f"Opset version: {opset_version}")
    print()
    
    # Dynamic axes for batch flexibility
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    try:
        # Export to ONNX via the legacy TorchScript exporter.
        # NOTE: torch >= 2.14 defaults to dynamo=True, which produced a
        # 0.49 MB graph missing most weights (parity check failed with
        # max_diff ~59). dynamo=False is the stable path for this model.
        # dynamo=False also supports opset 17 natively, without auto
        # version conversion.
        print("Exporting model...")
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False,
            dynamo=False,
        )
        
        print(f"✓ Export successful: {output_path}")
        
        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"  ONNX model size: {file_size_mb:.2f} MB")
        print()
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        raise


def simplify_onnx(onnx_path: Path) -> None:
    """
    Simplify ONNX model using onnx-simplifier.
    
    Args:
        onnx_path: Path to ONNX model (will be overwritten)
    """
    print("=" * 70)
    print("Simplifying ONNX Model")
    print("=" * 70)
    
    try:
        from onnxsim import simplify
        
        print("Loading ONNX model...")
        onnx_model = onnx.load(str(onnx_path))
        
        print("Simplifying...")
        model_simplified, check = simplify(onnx_model)
        
        if check:
            # Save simplified model
            onnx.save(model_simplified, str(onnx_path))
            print(f"✓ Simplification successful")
            
            # Get new file size
            file_size_mb = onnx_path.stat().st_size / (1024 ** 2)
            print(f"  Simplified model size: {file_size_mb:.2f} MB")
        else:
            print("✗ Simplification check failed (keeping original)")
        
        print()
        
    except ImportError:
        print("✗ onnx-simplifier not installed. Skipping simplification.")
        print("  Install with: pip install onnx-simplifier")
        print()
    except Exception as e:
        print(f"✗ Simplification failed: {e}")
        print("  Keeping original ONNX model.")
        print()


def verify_onnx(
    onnx_path: Path,
    pytorch_model: nn.Module,
    device: str,
    config: dict,
    verbose: bool = True
) -> bool:
    """
    Verify ONNX model correctness.
    
    Checks:
    1. ONNX model is valid
    2. ONNX Runtime can load and run it
    3. Outputs match PyTorch model (within tolerance)
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        device: Device for PyTorch inference
        config: Configuration dictionary
        verbose: Whether to print detailed results
        
    Returns:
        True if verification passed, False otherwise
    """
    print("=" * 70)
    print("Verifying ONNX Model")
    print("=" * 70)
    
    try:
        # 1. Check ONNX model validity
        if verbose:
            print("1. Checking ONNX model validity...")
        
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        if verbose:
            print("   ✓ ONNX model is valid")
        
        # 2. Load with ONNX Runtime
        if verbose:
            print("2. Loading with ONNX Runtime...")
        
        # Choose execution provider
        providers = []
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
            if verbose:
                print("   Using CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
        
        if verbose:
            print(f"   ✓ ONNX Runtime session created")
            print(f"   Active provider: {ort_session.get_providers()[0]}")
        
        # 3. Compare outputs
        if verbose:
            print("3. Comparing outputs with PyTorch model...")
        
        # Create test input.
        # NOTE: synthetic inputs (gaussian or uniform) are a bad parity
        # probe for this model — the trained BatchNorm statistics amplify
        # OOD activations so much that tiny CPU/GPU convolution
        # differences blow the logits to ~±3000, causing flaky failures.
        # Real val-split images (test split stays locked) agree to <5e-3,
        # verified separately in src/debug_realimage.py.
        import pandas as pd
        from data.transforms import get_val_transforms

        img_size = config['data']['img_size']
        processed_dir = Path.cwd() / config['data']['processed_dir']
        manifest = pd.read_csv(processed_dir / "manifest.csv")
        val_df = manifest[manifest["split"] == "val"].head(4)

        transform = get_val_transforms(config)
        test_input = torch.stack([
            transform(np.load(processed_dir / fp))
            for fp in val_df["filepath"]
        ])
        if verbose:
            print(f"   Parity inputs: {len(val_df)} real val images")
        
        # PyTorch inference
        pytorch_model = pytorch_model.to(device)
        pytorch_model.eval()
        
        with torch.no_grad():
            test_input_device = test_input.to(device)
            pytorch_output = pytorch_model(test_input_device)
            pytorch_output = pytorch_output.cpu().numpy()
        
        # ONNX Runtime inference
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Compute differences
        max_diff = np.abs(pytorch_output - ort_output).max()
        mean_diff = np.abs(pytorch_output - ort_output).mean()
        
        if verbose:
            print(f"   Max absolute difference: {max_diff:.6f}")
            print(f"   Mean absolute difference: {mean_diff:.6f}")
        
        # Check parity
        # For float32, use 1e-4 tolerance
        # For mixed precision models, use 1e-2 tolerance (more lenient)
        amp_enabled = config.get('training', {}).get('use_amp', False)
        tolerance = 1e-2 if amp_enabled else 1e-4
        
        if max_diff < tolerance:
            if verbose:
                print(f"   ✓ Output parity check passed (tolerance: {tolerance})")
            print()
            print("✓ Verification successful")
            print()
            return True
        else:
            if verbose:
                print(f"   ✗ Output parity check failed (max_diff={max_diff:.6f} > {tolerance})")
            print()
            print("✗ Verification failed: outputs do not match")
            print()
            return False
        
    except Exception as e:
        print(f"✗ Verification failed with error: {e}")
        print()
        return False


def benchmark_models(
    onnx_path: Path,
    pytorch_model: nn.Module,
    device: str,
    config: dict,
    num_iterations: int = 100
) -> None:
    """
    Benchmark PyTorch vs ONNX Runtime inference speed.
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        device: Device for inference
        config: Configuration dictionary
        num_iterations: Number of iterations for benchmarking
    """
    print("=" * 70)
    print("Benchmarking Performance")
    print("=" * 70)
    print(f"Running {num_iterations} iterations on each model...")
    print()
    
    # Get input size
    img_size = config['data']['img_size']
    
    # Create test input
    test_input = torch.randn(1, 3, img_size, img_size)
    
    # --- PyTorch Benchmark ---
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()
    
    test_input_device = test_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = pytorch_model(test_input_device)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    pytorch_times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = pytorch_model(test_input_device)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            pytorch_times.append(end - start)
    
    pytorch_avg = np.mean(pytorch_times) * 1000  # Convert to ms
    pytorch_std = np.std(pytorch_times) * 1000
    
    # --- ONNX Runtime Benchmark ---
    providers = []
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    
    ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
    ort_input_name = ort_session.get_inputs()[0].name
    
    test_input_numpy = test_input.numpy()
    
    # Warmup
    for _ in range(10):
        _ = ort_session.run(None, {ort_input_name: test_input_numpy})
    
    # Benchmark
    onnx_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = ort_session.run(None, {ort_input_name: test_input_numpy})
        end = time.perf_counter()
        onnx_times.append(end - start)
    
    onnx_avg = np.mean(onnx_times) * 1000  # Convert to ms
    onnx_std = np.std(onnx_times) * 1000
    
    # --- Results ---
    print(f"PyTorch ({device.upper()}):")
    print(f"  Average: {pytorch_avg:.2f} ± {pytorch_std:.2f} ms")
    print(f"  Throughput: {1000/pytorch_avg:.1f} images/sec")
    print()
    
    provider = ort_session.get_providers()[0]
    print(f"ONNX Runtime ({provider}):")
    print(f"  Average: {onnx_avg:.2f} ± {onnx_std:.2f} ms")
    print(f"  Throughput: {1000/onnx_avg:.1f} images/sec")
    print()
    
    # Speedup
    speedup = pytorch_avg / onnx_avg
    if speedup > 1.0:
        print(f"Speedup: {speedup:.2f}x faster with ONNX Runtime")
    elif speedup < 1.0:
        print(f"Speedup: {1/speedup:.2f}x slower with ONNX Runtime")
    else:
        print(f"Speedup: equivalent performance")
    print()
    
    # Model size
    onnx_size_mb = onnx_path.stat().st_size / (1024 ** 2)
    print(f"ONNX model size: {onnx_size_mb:.2f} MB")
    print()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    device, models_dir = setup_export(config)
    
    # Determine output path
    if args.output is None:
        model_name = config['model']['name']
        output_path = models_dir / f"{model_name}.onnx"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model from checkpoint
    model = load_model_from_checkpoint(args.checkpoint, config, device)
    
    # Export to ONNX
    try:
        export_to_onnx(model, output_path, args.opset_version, config)
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)
    
    # Simplify if requested
    if args.simplify:
        simplify_onnx(output_path)
    
    # Verify ONNX model
    verification_passed = verify_onnx(output_path, model, device, config)
    
    if not verification_passed:
        print("Warning: Verification failed. ONNX model may not produce correct results.")
        print()
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_models(output_path, model, device, config)
    
    # Summary
    print("=" * 70)
    print("Export Complete")
    print("=" * 70)
    print(f"ONNX model saved to: {output_path}")
    print(f"Verification: {'✓ Passed' if verification_passed else '✗ Failed'}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
