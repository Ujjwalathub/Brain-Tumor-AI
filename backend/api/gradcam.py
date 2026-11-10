"""
Grad-CAM explainability for the deployed brain-tumour classifier.

The REST service serves predictions through ONNX Runtime, which does not
expose gradients. This module therefore loads the *matching PyTorch
checkpoint* lazily (only when the first explanation is requested), runs a real
Grad-CAM pass, and returns a PNG heatmap overlay.

If PyTorch or the checkpoint is unavailable, callers receive a clear
``ExplanationUnavailable`` error instead of a fabricated heatmap.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

# Resolve the backend package root so `src.*` imports work regardless of cwd.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from src.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)


class ExplanationUnavailable(RuntimeError):
    """Raised when a real Grad-CAM explanation cannot be produced."""


class GradCAMEngine:
    """
    Lazy Grad-CAM engine backed by the training checkpoint.

    One instance is shared by the API; ``explain`` is thread-safe because the
    model forward/backward and hook buffers are guarded by a lock.
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: Optional[str] = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self._lock = threading.Lock()
        self._model = None
        self._torch = None
        self._device = device
        self._target_layer_name = ""

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return

        if not self.checkpoint_path.exists():
            raise ExplanationUnavailable(
                f"PyTorch checkpoint not found: {self.checkpoint_path}"
            )

        try:
            import torch
            import torch.nn as nn
            from src.models import create_model
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ExplanationUnavailable(
                f"PyTorch is required for Grad-CAM explanations: {exc}"
            ) from exc

        config = load_config(str(self.config_path))
        checkpoint = torch.load(
            str(self.checkpoint_path), map_location="cpu", weights_only=False
        )
        config = checkpoint.get("config", config)

        # Never download ImageNet weights here: the checkpoint overwrites them.
        config.setdefault("model", {})["pretrained"] = False
        model = create_model(config)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        model.eval()

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Last Conv2d is the deepest convolutional feature map (conv_head for
        # EfficientNet, layer4.2.conv3 for ResNet). That is the canonical
        # Grad-CAM target for these GAP architectures.
        target_name, target = "", None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_name, target = name, module
        if target is None:
            raise ExplanationUnavailable("No convolutional layer found in model")

        self._torch = torch
        self._model = model.to(self._device)
        if self._device == "cuda":
            self._model = self._model.to(memory_format=torch.channels_last)
        self._target = target
        self._target_layer_name = target_name
        self._class_names = list(config.get("data", {}).get("classes", []))
        self._activations = None
        self._gradients = None

        target.register_forward_hook(self._forward_hook)
        target.register_full_backward_hook(self._backward_hook)

        logger.info(
            "Grad-CAM engine ready (checkpoint=%s, device=%s, layer=%s)",
            self.checkpoint_path.name, self._device, target_name,
        )

    def _forward_hook(self, _module, _inputs, output):
        self._activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def explain(
        self,
        image: Image.Image,
        preprocess_fn: Callable[[Image.Image], np.ndarray],
        target_class: Optional[int] = None,
    ) -> dict:
        """
        Compute a Grad-CAM heatmap for ``image``.

        Args:
            image: PIL image (already validated / EXIF-corrected by caller).
            preprocess_fn: The ONNX engine's ``preprocess`` method, so the
                PyTorch pass sees byte-identical tensors to the ONNX pass.
            target_class: Class index to explain; defaults to the argmax.

        Returns:
            Dict with ``heatmap_png_base64``, ``class_name``, ``confidence``,
            ``method`` and ``target_layer``.
        """
        self._load()
        torch = self._torch

        with self._lock:
            tensor = torch.from_numpy(preprocess_fn(image)).to(self._device)
            if self._device == "cuda":
                tensor = tensor.to(memory_format=torch.channels_last)

            self._model.zero_grad(set_to_none=True)
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            class_idx = int(probabilities.argmax()) if target_class is None else int(target_class)
            confidence = float(probabilities[class_idx])

            logits[0, class_idx].backward()

            if self._activations is None or self._gradients is None:
                raise ExplanationUnavailable("Grad-CAM hooks did not capture activations")

            weights = self._gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self._activations).sum(dim=1))[0]
            cam = cam - cam.min()
            max_val = cam.max()
            if float(max_val) > 0:
                cam = cam / max_val
            else:
                raise ExplanationUnavailable("Grad-CAM produced an empty map")

            cam = torch.nn.functional.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            cam_np = cam.detach().cpu().numpy()

            self._activations = None
            self._gradients = None
            self._model.zero_grad(set_to_none=True)

        return {
            "heatmap_png_base64": _overlay_png(cam_np),
            "class_name": self._class_name(class_idx),
            "class_index": class_idx,
            "confidence": confidence,
            "method": "grad-cam",
            "target_layer": self._target_layer_name,
        }

    def _class_name(self, index: int) -> str:
        if 0 <= index < len(self._class_names):
            return self._class_names[index]
        return f"class_{index}"


def _overlay_png(cam: np.ndarray, alpha_scale: float = 210.0) -> str:
    """Convert a [0, 1] heatmap to a transparent JET-coloured PNG (base64)."""
    import cv2

    cam_u8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    rgba = np.zeros((cam.shape[0], cam.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = colored_bgr[..., ::-1]

    # Fade out low-activation regions so the underlying scan stays visible.
    alpha = np.clip((cam - 0.15) / 0.85, 0.0, 1.0) * alpha_scale
    rgba[..., 3] = alpha.astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
