"""
FastAPI inference server for Brain Tumor MRI Classification.

Provides a REST API for ONNX model inference with optional GPU acceleration,
plus a real Grad-CAM explainability endpoint backed by the matching PyTorch
checkpoint.

Configuration is environment-driven (see ``backend/.env.example``):
    MODEL_PATH          ONNX model to serve (default: models/onnx/efficientnet_b0.onnx)
    CONFIG_PATH         YAML config (default: configs/<model>.yaml)
    GRADCAM_CHECKPOINT  PyTorch checkpoint used for explanations
    MAX_UPLOAD_SIZE     Maximum upload size in bytes (default: 10 MiB)
    CORS_ORIGINS        Comma-separated allowed origins
"""

import logging
import os
import sys
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .gradcam import ExplanationUnavailable, GradCAMEngine
from .inference import COLOR_IMAGE_REJECTION_MESSAGE, ONNXInferenceEngine, is_grayscale_image
from .schemas import ExplanationResponse, HealthResponse, PredictionResponse

# ---------------------------------------------------------------------------
# Console / logging setup
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError, OSError):  # detached streams
    pass

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths & limits
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parent.parent

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(50_000_000)))
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/pjpeg",
    "image/png",
    "image/webp",
    "application/octet-stream",
}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# dev-server ports used by Vite/TanStack Start and common static servers
_DEFAULT_ORIGINS = [
    f"http://{host}:{port}"
    for host in ("localhost", "127.0.0.1")
    for port in (3000, 4173, 5173, 8000, 8080)
]


def _env_path(name: str) -> Optional[Path]:
    raw = os.getenv(name)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (BACKEND_ROOT / path)


def resolve_model_path() -> Optional[Path]:
    """Pick the ONNX model deterministically (env override wins)."""
    explicit = _env_path("MODEL_PATH")
    if explicit is not None:
        return explicit if explicit.exists() else None

    onnx_dir = BACKEND_ROOT / "models" / "onnx"
    default = onnx_dir / "efficientnet_b0.onnx"
    if default.exists():
        return default

    candidates = sorted(onnx_dir.glob("*.onnx"))
    return candidates[0] if candidates else None


def resolve_config_path(model_path: Path) -> Optional[Path]:
    """Find the config that describes ``model_path``."""
    explicit = _env_path("CONFIG_PATH")
    if explicit is not None:
        return explicit if explicit.exists() else None

    configs_dir = BACKEND_ROOT / "configs"
    stem = model_path.stem.replace("_optimized", "")
    for name in (f"{stem}.yaml", "base.yaml"):
        candidate = configs_dir / name
        if candidate.exists():
            return candidate
    return None


def resolve_gradcam_checkpoint() -> Optional[Path]:
    """
    Checkpoint used for explanations.

    Only enabled when it provably matches the deployed weights: the default
    is the verified augaffine epoch-13 checkpoint (bit-identical logits to the
    deployed ONNX). Override with GRADCAM_CHECKPOINT when swapping models.
    """
    explicit = _env_path("GRADCAM_CHECKPOINT")
    if explicit is not None:
        return explicit if explicit.exists() else None

    default = (
        BACKEND_ROOT
        / "models"
        / "run_augaffine"
        / "best_epoch013_macro-recall0.9988.pth"
    )
    return default if default.exists() else None


# Global engines, populated during lifespan
inference_engine: Optional[ONNXInferenceEngine] = None
gradcam_engine: Optional[GradCAMEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup and release it on shutdown."""
    global inference_engine, gradcam_engine

    logger.info("=" * 70)
    logger.info("Brain Tumor MRI Classifier - Inference API")
    logger.info("=" * 70)

    model_path = resolve_model_path()
    config_path = resolve_config_path(model_path) if model_path else None

    if model_path and config_path:
        try:
            logger.info(f"Loading model: {model_path}")
            logger.info(f"Loading config: {config_path}")

            inference_engine = ONNXInferenceEngine(
                model_path=str(model_path),
                config_path=str(config_path),
                use_gpu=os.getenv("USE_GPU", "1") not in {"0", "false", "False"},
            )
            inference_engine.warmup(num_iterations=10)
            logger.info("[OK] Inference engine ready")
        except Exception as exc:
            logger.error(f"Failed to load inference engine: {exc}", exc_info=True)
            logger.warning("API will start but /predict will return 503")
            inference_engine = None
    else:
        logger.warning("No ONNX model / config found")
        logger.warning("API will start but /predict will return 503")
        logger.info(f"Expected model location: {BACKEND_ROOT / 'models' / 'onnx'}")

    checkpoint = resolve_gradcam_checkpoint()
    if checkpoint and inference_engine is not None and config_path is not None:
        gradcam_engine = GradCAMEngine(str(checkpoint), str(config_path))
        logger.info(f"Grad-CAM checkpoint: {checkpoint}")
    else:
        gradcam_engine = None
        logger.warning("Grad-CAM explanations disabled (no matching checkpoint)")

    logger.info("=" * 70)
    logger.info("API server started")
    logger.info("=" * 70)

    yield

    logger.info("Shutting down inference API")
    inference_engine = None
    gradcam_engine = None


app = FastAPI(
    title="Brain Tumor MRI Classifier API",
    description=(
        "REST API for brain tumor classification from grayscale MRI images. "
        "Classifies into 4 categories: glioma, meningioma, notumor, pituitary. "
        "Color images are rejected as potentially garbage input."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# CORS for local development. Override with CORS_ORIGINS="https://a.example,...".
_origins_env = os.getenv("CORS_ORIGINS", "").strip()
allowed_origins = (
    [o.strip() for o in _origins_env.split(",") if o.strip()]
    if _origins_env
    else _DEFAULT_ORIGINS
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Return structured errors under the ``error`` key used by the client."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "detail": ""},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------
def _read_image(contents: bytes, filename: Optional[str]) -> Image.Image:
    """
    Decode, orientation-correct and validate an uploaded image.

    Raises HTTPException with a user-facing message for every reject path.
    """
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(BytesIO(contents))
            image.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Failed to process image: '{filename or 'upload'}' is not a "
                "valid JPEG, PNG or WebP file"
            ),
        )
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Failed to process image: {exc}"
        )

    image_format = (image.format or "").upper()
    if image_format not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid image format: {image_format or 'unknown'}. "
                "Supported: JPEG, PNG, WebP"
            ),
        )

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit",
        )

    # Respect EXIF orientation (phone photos) before cropping/thresholding.
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    return image


async def _read_upload(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type: {file.content_type}. "
                "Supported: JPEG, PNG, WebP"
            ),
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(contents)} bytes). Maximum is "
                f"{MAX_UPLOAD_BYTES / (1024 * 1024):.1f} MB"
            ),
        )
    return contents


def _validate_grayscale(image: Image.Image, filename: Optional[str]) -> None:
    if is_grayscale_image(image):
        return
    logger.warning(
        f"Rejected color image upload: {filename} (mode={image.mode})"
    )
    raise HTTPException(status_code=422, detail=COLOR_IMAGE_REJECTION_MESSAGE)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """API information and usage instructions."""
    return {
        "name": "Brain Tumor MRI Classifier API",
        "version": "1.1.0",
        "description": "ONNX-based inference API for brain tumor classification",
        "endpoints": {
            "/": "API information",
            "/health": "Health check and model status",
            "/predict": "POST multipart/form-data with 'file' field containing MRI image",
            "/explain": "POST multipart/form-data; returns a real Grad-CAM heatmap",
        },
        "supported_formats": ["JPEG", "PNG", "WebP"],
        "notes": "Grayscale MRI scans only — color images are rejected.",
        "classes": ["glioma", "meningioma", "notumor", "pituitary"],
        "max_upload_mb": round(MAX_UPLOAD_BYTES / (1024 * 1024), 1),
        "model_loaded": inference_engine is not None,
        "gradcam_available": gradcam_engine is not None,
        "usage": {
            "curl": (
                "curl -X POST -F 'file=@image.jpg' "
                "http://localhost:8000/predict"
            ),
            "python": (
                "import requests\n"
                "with open('image.jpg', 'rb') as f:\n"
                "    response = requests.post(\n"
                "        'http://localhost:8000/predict',\n"
                "        files={'file': f}\n"
                "    )\n"
                "print(response.json())"
            ),
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Service status and model information."""
    if inference_engine is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            device="none",
            model_name="none",
            gradcam_available=False,
        )

    info = inference_engine.get_info()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=info["device"],
        model_name=info["model_name"],
        gradcam_available=gradcam_engine is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict the tumor class from an uploaded MRI image.

    Only grayscale (MRI) content is accepted; colour images are rejected with
    HTTP 422. Inference runs in a worker thread so concurrent uploads do not
    block the event loop.
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for details.",
        )

    contents = await _read_upload(file)
    image = _read_image(contents, file.filename)
    _validate_grayscale(image, file.filename)

    try:
        result = await run_in_threadpool(inference_engine.predict, image)
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=422, detail=f"Failed to process image: {exc}"
        )

    logger.info(
        f"Prediction: {result['class_name']} "
        f"(confidence: {result['confidence']:.4f}, "
        f"time: {result['inference_time_ms']:.2f}ms)"
    )
    return PredictionResponse(**result)


@app.post("/explain", response_model=ExplanationResponse)
async def explain(file: UploadFile = File(...)):
    """
    Return a real Grad-CAM heatmap (PNG, base64) for an uploaded MRI image.

    Requires the matching PyTorch checkpoint; when it is missing the endpoint
    returns 503 rather than inventing a heatmap.
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for details.",
        )
    if gradcam_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Grad-CAM is unavailable: no matching PyTorch checkpoint was "
                "found. Set GRADCAM_CHECKPOINT to enable explanations."
            ),
        )

    contents = await _read_upload(file)
    image = _read_image(contents, file.filename)
    _validate_grayscale(image, file.filename)

    try:
        result = await run_in_threadpool(
            gradcam_engine.explain, image, inference_engine.preprocess
        )
    except ExplanationUnavailable as exc:
        logger.warning(f"Grad-CAM unavailable: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"Grad-CAM failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=422, detail=f"Failed to explain image: {exc}"
        )

    return ExplanationResponse(**result)


# For local development/testing
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
