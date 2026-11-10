"""
Pydantic models for request/response validation.
"""

from typing import Dict
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    
    class_name: str = Field(
        ...,
        description="Predicted tumor class",
        examples=["glioma"]
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for predicted class",
        examples=[0.9523]
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all classes",
        examples=[{
            "glioma": 0.9523,
            "meningioma": 0.0234,
            "notumor": 0.0123,
            "pituitary": 0.0120
        }]
    )
    inference_time_ms: float = Field(
        ...,
        gt=0.0,
        description="Inference time in milliseconds",
        examples=[12.34]
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(
        ...,
        description="Service status",
        examples=["healthy"]
    )
    model_loaded: bool = Field(
        ...,
        description="Whether ONNX model is loaded",
        examples=[True]
    )
    device: str = Field(
        ...,
        description="Execution provider (CUDA or CPU)",
        examples=["CUDAExecutionProvider"]
    )
    model_name: str = Field(
        ...,
        description="Name of loaded model",
        examples=["efficientnet_b0"]
    )
    gradcam_available: bool = Field(
        default=False,
        description="Whether real Grad-CAM explanations are available",
        examples=[True]
    )


class ExplanationResponse(BaseModel):
    """Response model for the Grad-CAM explanation endpoint."""

    heatmap_png_base64: str = Field(
        ...,
        description=(
            "Base64-encoded RGBA PNG heatmap at the uploaded image's "
            "resolution, ready to overlay on the scan"
        )
    )
    class_name: str = Field(
        ...,
        description="Class index that was explained",
        examples=["glioma"]
    )
    class_index: int = Field(
        ...,
        ge=0,
        description="Explained class index",
        examples=[0]
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model probability for the explained class",
        examples=[0.9523]
    )
    method: str = Field(
        default="grad-cam",
        description="Explainability method",
        examples=["grad-cam"]
    )
    target_layer: str = Field(
        default="",
        description="Name of the convolutional layer used for Grad-CAM",
        examples=["conv_head"]
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(
        ...,
        description="Error message",
        examples=["Invalid image format"]
    )
    detail: str = Field(
        default="",
        description="Additional error details",
        examples=["Supported formats: JPEG, PNG"]
    )
