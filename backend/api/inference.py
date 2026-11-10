"""
ONNX Runtime inference engine with GPU acceleration.
Matches exact preprocessing pipeline from training.
"""

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

# Import the shared config loader (supports `_base_` inheritance). The API is
# normally started from `backend/`, but resolve `src` relative to this file so
# `uvicorn api.main:app` also works from the repository root.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from src.config import configure_utf8_console, load_config  # noqa: E402

configure_utf8_console()

# ---------------------------------------------------------------------------
# Windows DLL resolution for the ONNX Runtime CUDA provider.
# The CUDA EP needs cuDNN/cuBLAS DLLs, which are bundled with the PyTorch
# wheel (torch/lib) but NOT with onnxruntime. When torch was imported
# earlier in the same process (e.g. export scripts) this works by chance;
# in the API process no torch import exists, so register the directories
# explicitly. Must run BEFORE importing onnxruntime.
# ---------------------------------------------------------------------------
import importlib.util

_candidate_dirs = []
try:
    _spec = importlib.util.find_spec("torch")
    if _spec and _spec.origin:
        _candidate_dirs.append(Path(_spec.origin).parent / "lib")
except (ImportError, AttributeError, ValueError):
    pass

try:
    import onnxruntime as _ort_probe

    _candidate_dirs.append(
        Path(_ort_probe.__file__).parent / "capi"
    )
except (ImportError, AttributeError):
    pass

_registered_dirs = []
for _dir in _candidate_dirs:
    if _dir.is_dir():
        # Both mechanisms: add_dll_directory covers DLLs loaded by absolute
        # path within the default-search audit; PATH prepending covers the
        # cuDNN 9 facade, which locates cudnn_graph64_9.dll etc. by plain
        # name via the full search chain (add_dll_directory alone is not
        # sufficient — "Invalid handle. Cannot load symbol cudnnCreate").
        os.add_dll_directory(str(_dir))
        os.environ["PATH"] = str(_dir) + os.pathsep + os.environ["PATH"]
        _registered_dirs.append(str(_dir))

if not _registered_dirs:
    logging.getLogger(__name__).warning(
        "No torch/lib or onnxruntime/capi DLL directories found — "
        "CUDA EP may fail to load cudnn64_9.dll"
    )

import onnxruntime as ort  # noqa: E402  (DLL dirs must be registered first)

logger = logging.getLogger(__name__)


# ImageNet normalization constants (must match training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Grayscale gate (project objective).
# Brain MRI scans are single-channel (grayscale) images. Any color (RGB)
# upload is treated as potential garbage / non-MRI input and rejected.
# The check is content-based rather than mode-based: JPEG and PNG often
# store genuine grayscale scans as three identical channels, so an image
# is accepted as long as its channels are (near-)identical.
# ---------------------------------------------------------------------------
GRAYSCALE_ACCEPTED_MODES = frozenset({'L', '1', 'I', 'I;16', 'F', 'LA', 'La'})
_CHROMA_THRESHOLD = 15        # max channel spread tolerated (JPEG/PNG compression noise)
_MAX_COLORED_FRACTION = 0.01  # tolerate <=1% of sampled pixels with visible color
COLOR_IMAGE_REJECTION_MESSAGE = (
    "Rejected: this appears to be a color (RGB) image. The classifier only "
    "accepts grayscale brain MRI scans — color images are treated as "
    "potentially garbage input. Please upload a grayscale MRI scan (JPEG or PNG)."
)


def is_grayscale_image(image: Image.Image, max_side: int = 256) -> bool:
    """
    Check whether an image is effectively grayscale.

    Args:
        image: PIL Image to inspect.
        max_side: Downscale sample size for fast channel checking on
            large uploads.

    Returns:
        True if the image content is grayscale (single channel, or RGB
        with near-identical channels), False if it carries real color.
    """
    if image.mode in GRAYSCALE_ACCEPTED_MODES:
        return True

    rgb = image.convert('RGB')
    if max_side and max(rgb.size) > max_side:
        rgb.thumbnail((max_side, max_side), Image.BILINEAR)

    arr = np.asarray(rgb, dtype=np.int16)
    if arr.ndim != 3 or arr.shape[2] == 1:
        return True

    # Per-pixel channel spread: 0 for grayscale content stored as RGB.
    chroma = arr.max(axis=2) - arr.min(axis=2)
    colored_fraction = float((chroma > _CHROMA_THRESHOLD).mean())
    return colored_fraction <= _MAX_COLORED_FRACTION


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine with preprocessing pipeline
    matching the training data pipeline exactly.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        use_gpu: bool = True
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to ONNX model file
            config_path: Path to model config YAML
            use_gpu: Whether to use CUDA GPU acceleration
            
        Raises:
            FileNotFoundError: If model or config not found
            RuntimeError: If ONNX session creation fails
        """
        model_path = Path(model_path)
        config_path = Path(config_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        logger.info(f"Loading ONNX model: {model_path}")
        logger.info(f"Loading config: {config_path}")
        
        # Load merged config (supports `_base_` inheritance, same as training).
        self.config = load_config(str(config_path))
        
        # Extract model info
        self.model_name = self.config.get('model', {}).get('name', 'unknown')
        self.num_classes = int(self.config.get('data', {}).get('num_classes', 4))
        self.class_names = list(self.config.get('data', {}).get('classes', [
            'glioma', 'meningioma', 'notumor', 'pituitary'
        ]))
        self.img_size = int(self.config.get('data', {}).get('img_size', 224))
        
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"Config mismatch: {self.num_classes} classes declared but "
                f"{len(self.class_names)} class names provided"
            )
        
        # Normalization parameters (should match training exactly)
        aug_config = self.config.get('augmentation', {})
        self.mean = np.array(aug_config.get('mean', IMAGENET_MEAN.tolist()), dtype=np.float32)
        self.std = np.array(aug_config.get('std', IMAGENET_STD.tolist()), dtype=np.float32)
        
        if self.mean.shape != (3,) or self.std.shape != (3,):
            raise ValueError("Augmentation mean/std must contain exactly 3 values")
        if np.any(self.std == 0):
            raise ValueError("Augmentation std must not contain zeros")
        
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Image size: {self.img_size}x{self.img_size}")
        logger.info(f"Normalization - Mean: {self.mean.tolist()}, Std: {self.std.tolist()}")
        
        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set execution providers
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            # Log active provider
            active_provider = self.session.get_providers()[0]
            logger.info(f"ONNX Runtime provider: {active_provider}")
            
            if use_gpu and active_provider != 'CUDAExecutionProvider':
                logger.warning(
                    "CUDA provider requested but not available. "
                    "Falling back to CPU. Install onnxruntime-gpu for GPU support."
                )
            
            self.device = active_provider
            
        except Exception as e:
            raise RuntimeError(f"Failed to create ONNX session: {e}")
        
        # Get model input/output info
        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]
        self.input_name = input_meta.name
        self.output_name = output_meta.name
        
        # Fail fast on a model/config mismatch instead of returning silently
        # wrong probabilities at request time.
        input_shape = input_meta.shape
        if len(input_shape) == 4:
            expected = (3, self.img_size, self.img_size)
            for axis, want in zip(input_shape[1:], expected):
                if isinstance(axis, int) and axis != want:
                    raise ValueError(
                        f"Config says img_size={self.img_size} with 3 channels, "
                        f"but the ONNX input is {input_shape}"
                    )
        
        output_shape = output_meta.shape
        if len(output_shape) == 2 and isinstance(output_shape[1], int):
            if output_shape[1] != self.num_classes:
                raise ValueError(
                    f"Config says num_classes={self.num_classes}, but the ONNX "
                    f"model outputs {output_shape[1]} logits"
                )
        
        logger.info(f"Input name: {self.input_name} {input_shape}")
        logger.info(f"Output name: {self.output_name} {output_shape}")
        logger.info("Inference engine initialized successfully")
    
    def _find_brain_contour(
        self,
        image: np.ndarray,
        threshold: int = 10
    ) -> tuple[int, int, int, int]:
        """
        Find bounding box of brain region by removing black margins.
        Must match training preprocessing exactly.
        
        Args:
            image: Input image as numpy array (H, W, C)
            threshold: Pixel value threshold for non-black
            
        Returns:
            Tuple of (x, y, width, height) for cropping
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Threshold to binary mask
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            # No contours found, return full image bounds
            return 0, 0, image.shape[1], image.shape[0]
        
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Guard against degenerate boxes (single-pixel contours, etc.)
        if w < 2 or h < 2:
            return 0, 0, image.shape[1], image.shape[0]
        
        return x, y, w, h
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess PIL image to model input tensor.
        Must match training preprocessing pipeline exactly:
        1. Convert to RGB
        2. Crop black margins (brain region detection)
        3. Resize to target size
        4. Convert to float32 [0, 1]
        5. Normalize with ImageNet stats
        6. Transpose to CHW format
        7. Add batch dimension
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed numpy array of shape (1, 3, H, W)
        """
        # Convert PIL to numpy (RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)  # (H, W, 3), uint8, RGB
        
        # Find brain region and crop margins
        x, y, w, h = self._find_brain_contour(img_array)
        cropped = img_array[y:y+h, x:x+w]
        
        # Resize to target size (matching cv2.INTER_AREA from training)
        resized = cv2.resize(
            cropped,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA
        )
        
        # Convert to float32 and scale to [0, 1]
        img_float = resized.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        # PyTorch transforms normalize per-channel: (x - mean) / std
        img_normalized = (img_float - self.mean) / self.std
        
        # Transpose from HWC to CHW
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension: (1, C, H, W)
        img_batch = np.expand_dims(img_chw, axis=0)
        
        return img_batch
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with:
                - class_name: predicted class
                - confidence: prediction confidence
                - probabilities: dict of class -> probability
                - inference_time_ms: inference time in milliseconds
                
        Raises:
            ValueError: If prediction fails
        """
        try:
            # Preprocess
            input_array = self.preprocess(image)
            
            # Run inference with timing
            start_time = time.perf_counter()
            
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_array}
            )
            
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000
            
            # Get logits
            logits = np.asarray(outputs[0][0]).reshape(-1)  # Remove batch dimension
            if logits.shape[0] != self.num_classes:
                raise ValueError(
                    f"Model returned {logits.shape[0]} logits, expected "
                    f"{self.num_classes}"
                )
            
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
            probabilities_array = exp_logits / np.sum(exp_logits)
            
            # Get predicted class
            predicted_idx = int(np.argmax(probabilities_array))
            predicted_class = self.class_names[predicted_idx]
            confidence = float(probabilities_array[predicted_idx])
            
            # Build probabilities dict in the training class order
            probabilities = {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probabilities_array)
            }
            
            return {
                'class_name': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'inference_time_ms': inference_time_ms
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}")
    
    def warmup(self, num_iterations: int = 10) -> None:
        """
        Warm up the inference engine with dummy inputs.
        Important for GPU to allocate memory and compile kernels.
        
        Args:
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Warming up inference engine ({num_iterations} iterations)...")
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, self.img_size, self.img_size).astype(np.float32)
        
        # Run warmup iterations
        for i in range(num_iterations):
            self.session.run(
                [self.output_name],
                {self.input_name: dummy_input}
            )
        
        logger.info("Warmup complete")
    
    def get_info(self) -> Dict:
        """
        Get inference engine information.
        
        Returns:
            Dictionary with engine info
        """
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'img_size': self.img_size,
            'device': self.device,
            'input_name': self.input_name,
            'output_name': self.output_name
        }
