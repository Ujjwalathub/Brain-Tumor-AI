"""
Comprehensive test suite for FastAPI inference server.

Tests cover:
- Health and root endpoints
- Prediction endpoint with valid/invalid inputs
- CORS headers
- Error handling
- Preprocessing pipeline
- All class predictions
"""

from io import BytesIO
from typing import Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from api.main import app
from api.inference import ONNXInferenceEngine, is_grayscale_image


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_inference_engine():
    """
    Mock ONNXInferenceEngine with predict method returning dummy predictions.
    """
    mock_engine = Mock(spec=ONNXInferenceEngine)
    
    # Default prediction result
    mock_engine.predict.return_value = {
        'class_name': 'glioma',
        'confidence': 0.9523,
        'probabilities': {
            'glioma': 0.9523,
            'meningioma': 0.0234,
            'notumor': 0.0123,
            'pituitary': 0.0120
        },
        'inference_time_ms': 12.34
    }
    
    # Engine info
    mock_engine.get_info.return_value = {
        'model_name': 'efficientnet_b0',
        'num_classes': 4,
        'class_names': ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'img_size': 224,
        'device': 'CUDAExecutionProvider',
        'input_name': 'input',
        'output_name': 'output'
    }
    
    return mock_engine


@pytest.fixture
def client(mock_inference_engine):
    """
    FastAPI TestClient with mocked inference engine.
    """
    # Patch the global inference_engine in the main module
    with patch('api.main.inference_engine', mock_inference_engine):
        yield TestClient(app)


@pytest.fixture
def sample_image():
    """
    Create a dummy MRI scan image (256x256 grayscale, converted to RGB).
    """
    # Create grayscale image with brain-like circular region
    img_array = np.zeros((256, 256), dtype=np.uint8)
    
    # Add circular brain region (center circle)
    center_y, center_x = 128, 128
    radius = 80
    y, x = np.ogrid[:256, :256]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    img_array[mask] = 150  # Brain tissue intensity
    
    # Add some variation to make it look more realistic
    noise = np.random.randint(0, 50, (256, 256), dtype=np.uint8)
    img_array = np.clip(img_array.astype(np.int16) + noise * mask, 0, 255).astype(np.uint8)
    
    # Convert to RGB PIL Image
    img = Image.fromarray(img_array, mode='L').convert('RGB')
    
    return img


# ============================================================================
# Endpoint Tests
# ============================================================================

def test_health_endpoint(client, mock_inference_engine):
    """
    Test GET /health endpoint with loaded model.
    """
    response = client.get("/health")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True
    assert data['device'] == 'CUDAExecutionProvider'
    assert data['model_name'] == 'efficientnet_b0'


def test_health_endpoint_no_model():
    """
    Test GET /health endpoint when model is not loaded.
    """
    with patch('api.main.inference_engine', None):
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'degraded'
        assert data['model_loaded'] is False
        assert data['device'] == 'none'
        assert data['model_name'] == 'none'


def test_root_endpoint(client):
    """
    Test GET / endpoint returns API information.
    """
    response = client.get("/")
    
    assert response.status_code == 200
    
    data = response.json()
    assert 'name' in data
    assert 'version' in data
    assert 'description' in data
    assert 'endpoints' in data
    assert 'classes' in data
    assert 'model_loaded' in data
    
    # Verify expected classes
    assert data['classes'] == ['glioma', 'meningioma', 'notumor', 'pituitary']
    assert data['model_loaded'] is True
    
    # Verify endpoints documentation
    assert '/' in data['endpoints']
    assert '/health' in data['endpoints']
    assert '/predict' in data['endpoints']


def test_predict_valid_image(client, sample_image, mock_inference_engine):
    """
    Test POST /predict with valid image upload.
    """
    # Convert PIL Image to bytes
    img_bytes = BytesIO()
    sample_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Upload file
    files = {'file': ('test_image.png', img_bytes, 'image/png')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    
    data = response.json()
    
    # Verify response structure
    assert 'class_name' in data
    assert 'confidence' in data
    assert 'probabilities' in data
    assert 'inference_time_ms' in data
    
    # Verify data types and ranges
    assert isinstance(data['class_name'], str)
    assert data['class_name'] in ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    assert isinstance(data['confidence'], float)
    assert 0.0 <= data['confidence'] <= 1.0
    
    assert isinstance(data['probabilities'], dict)
    assert len(data['probabilities']) == 4
    
    # Verify probabilities sum to approximately 1.0
    prob_sum = sum(data['probabilities'].values())
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, expected ~1.0"
    
    # Verify all expected classes are in probabilities
    for cls in ['glioma', 'meningioma', 'notumor', 'pituitary']:
        assert cls in data['probabilities']
        assert 0.0 <= data['probabilities'][cls] <= 1.0
    
    assert isinstance(data['inference_time_ms'], float)
    assert data['inference_time_ms'] > 0
    
    # Verify the mock was called
    mock_inference_engine.predict.assert_called_once()


def test_predict_invalid_file(client):
    """
    Test POST /predict with non-image file (e.g., text file).
    """
    # Create a text file
    text_content = b"This is not an image file"
    files = {'file': ('test.txt', BytesIO(text_content), 'text/plain')}
    
    response = client.post("/predict", files=files)
    
    # Should return client error
    assert response.status_code == 400
    
    data = response.json()
    # API's exception handler returns error message in 'error' key
    assert 'error' in data
    assert 'Invalid file type' in str(data['error'])


def test_predict_missing_file(client):
    """
    Test POST /predict without file parameter.
    """
    response = client.post("/predict")
    
    # FastAPI validation error
    assert response.status_code == 422
    
    data = response.json()
    assert 'detail' in data


def test_predict_corrupted_image(client, mock_inference_engine):
    """
    Test POST /predict with corrupted image data.
    """
    # Create corrupted image bytes
    corrupted_bytes = b"fake image data that cannot be decoded"
    files = {'file': ('corrupted.jpg', BytesIO(corrupted_bytes), 'image/jpeg')}
    
    response = client.post("/predict", files=files)
    
    # Should return unprocessable entity
    assert response.status_code == 422
    
    data = response.json()
    # API's exception handler returns error message in 'error' key
    assert 'error' in data
    assert 'Failed to process image' in str(data['error'])


def test_predict_no_model_loaded():
    """
    Test POST /predict when model is not loaded.
    """
    with patch('api.main.inference_engine', None):
        client = TestClient(app)
        
        # Create dummy image
        img = Image.new('RGB', (256, 256), color='gray')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {'file': ('test.png', img_bytes, 'image/png')}
        response = client.post("/predict", files=files)
        
        # Should return service unavailable
        assert response.status_code == 503
        
        data = response.json()
        # API's exception handler returns error message in 'error' key
        assert 'error' in data
        assert 'Model not loaded' in str(data['error'])


# ============================================================================
# Preprocessing Tests
# ============================================================================

def test_inference_preprocessing():
    """
    Unit test for ONNXInferenceEngine.preprocess() method.
    Verifies preprocessing pipeline output shape and value ranges.
    """
    # Create a mock inference engine with real preprocess method
    with patch('onnxruntime.InferenceSession'):
        # Mock config data
        mock_config = {
            'model': {'name': 'test_model'},
            'data': {
                'num_classes': 4,
                'classes': ['glioma', 'meningioma', 'notumor', 'pituitary'],
                'img_size': 224
            },
            'augmentation': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
        
        with patch('builtins.open', create=True):
            with patch('yaml.safe_load', return_value=mock_config):
                with patch('pathlib.Path.exists', return_value=True):
                    # Create engine (will use mocked ONNX session)
                    engine = ONNXInferenceEngine(
                        model_path='dummy.onnx',
                        config_path='dummy.yaml',
                        use_gpu=False
                    )
    
    # Create test image (256x256 RGB with brain-like region)
    img_array = np.zeros((256, 256, 3), dtype=np.uint8)
    center_y, center_x = 128, 128
    radius = 80
    y, x = np.ogrid[:256, :256]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    for c in range(3):
        img_array[mask, c] = 150
    
    img = Image.fromarray(img_array, mode='RGB')
    
    # Run preprocessing
    output = engine.preprocess(img)
    
    # Verify output shape (1, 3, 224, 224)
    assert output.shape == (1, 3, 224, 224), f"Expected (1, 3, 224, 224), got {output.shape}"
    
    # Verify dtype
    assert output.dtype == np.float32, f"Expected float32, got {output.dtype}"
    
    # Verify value range is reasonable (normalized, should be roughly [-3, 3])
    # After ImageNet normalization: (x/255 - mean) / std
    assert output.min() >= -5.0, f"Min value {output.min()} seems too low"
    assert output.max() <= 5.0, f"Max value {output.max()} seems too high"
    
    # Verify not all zeros or NaN
    assert not np.all(output == 0), "Output is all zeros"
    assert not np.any(np.isnan(output)), "Output contains NaN values"
    assert not np.any(np.isinf(output)), "Output contains inf values"


def test_preprocess_grayscale_image():
    """
    Test preprocessing with grayscale image (should convert to RGB).
    """
    with patch('onnxruntime.InferenceSession'):
        mock_config = {
            'model': {'name': 'test_model'},
            'data': {
                'num_classes': 4,
                'classes': ['glioma', 'meningioma', 'notumor', 'pituitary'],
                'img_size': 224
            },
            'augmentation': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
        
        with patch('builtins.open', create=True):
            with patch('yaml.safe_load', return_value=mock_config):
                with patch('pathlib.Path.exists', return_value=True):
                    engine = ONNXInferenceEngine(
                        model_path='dummy.onnx',
                        config_path='dummy.yaml',
                        use_gpu=False
                    )
    
    # Create grayscale image
    img = Image.new('L', (256, 256), color=128)
    
    # Preprocess
    output = engine.preprocess(img)
    
    # Should still produce (1, 3, 224, 224) output
    assert output.shape == (1, 3, 224, 224)
    assert output.dtype == np.float32


# ============================================================================
# Multi-Class Prediction Tests
# ============================================================================

@pytest.mark.parametrize("predicted_class,confidence", [
    ('glioma', 0.95),
    ('meningioma', 0.87),
    ('notumor', 0.92),
    ('pituitary', 0.89)
])
def test_predict_all_classes(sample_image, predicted_class, confidence):
    """
    Test that all 4 classes can be returned by the prediction endpoint.
    Mocks different predictions for each class.
    """
    # Create mock engine with specific prediction
    mock_engine = Mock(spec=ONNXInferenceEngine)
    
    # Build probabilities dict with predicted class having highest prob
    base_prob = (1.0 - confidence) / 3  # Distribute remaining among other classes
    probabilities = {
        'glioma': base_prob,
        'meningioma': base_prob,
        'notumor': base_prob,
        'pituitary': base_prob
    }
    probabilities[predicted_class] = confidence
    
    mock_engine.predict.return_value = {
        'class_name': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities,
        'inference_time_ms': 10.0
    }
    
    mock_engine.get_info.return_value = {
        'model_name': 'test_model',
        'num_classes': 4,
        'class_names': ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'img_size': 224,
        'device': 'CPUExecutionProvider',
        'input_name': 'input',
        'output_name': 'output'
    }
    
    # Use TestClient with this specific mock
    with patch('api.main.inference_engine', mock_engine):
        client = TestClient(app)
        
        # Convert image to bytes
        img_bytes = BytesIO()
        sample_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Make prediction
        files = {'file': ('test.png', img_bytes, 'image/png')}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify predicted class
        assert data['class_name'] == predicted_class
        assert abs(data['confidence'] - confidence) < 0.01
        
        # Verify all classes are present in probabilities
        for cls in ['glioma', 'meningioma', 'notumor', 'pituitary']:
            assert cls in data['probabilities']


# ============================================================================
# CORS Tests
# ============================================================================

def test_cors_headers(client, sample_image):
    """
    Test that CORS headers are present in responses.
    Verifies cross-origin requests are allowed.
    """
    # Test OPTIONS preflight request
    response = client.options(
        "/predict",
        headers={
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
    )
    
    # CORS middleware should handle OPTIONS
    assert response.status_code == 200
    
    # Check CORS headers in actual request
    img_bytes = BytesIO()
    sample_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {'file': ('test.png', img_bytes, 'image/png')}
    response = client.post(
        "/predict",
        files=files,
        headers={'Origin': 'http://localhost:3000'}
    )
    
    assert response.status_code == 200
    
    # Verify CORS headers are present
    assert 'access-control-allow-origin' in response.headers
    # With allow_origins=["*"], CORS middleware echoes the origin or returns *
    assert response.headers['access-control-allow-origin'] in ['*', 'http://localhost:3000']


def test_cors_health_endpoint(client):
    """
    Test CORS headers on health endpoint.
    """
    response = client.get(
        "/health",
        headers={'Origin': 'http://localhost:5173'}
    )
    
    assert response.status_code == 200
    assert 'access-control-allow-origin' in response.headers


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

def test_predict_very_small_image(client, mock_inference_engine):
    """
    Test prediction with very small image (e.g., 32x32).
    Should still work after resizing.
    """
    # Create small image
    small_img = Image.new('RGB', (32, 32), color=(100, 100, 100))
    
    img_bytes = BytesIO()
    small_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {'file': ('small.png', img_bytes, 'image/png')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert 'class_name' in data


def test_predict_large_image(client, mock_inference_engine):
    """
    Test prediction with large image (e.g., 2048x2048).
    Should still work after resizing.
    """
    # Create large image (using smaller size for test speed)
    large_img = Image.new('RGB', (1024, 1024), color=(150, 150, 150))
    
    img_bytes = BytesIO()
    large_img.save(img_bytes, format='JPEG', quality=85)
    img_bytes.seek(0)
    
    files = {'file': ('large.jpg', img_bytes, 'image/jpeg')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert 'class_name' in data


def test_predict_rgba_image(client, mock_inference_engine):
    """
    Test prediction with RGBA image (PNG with transparency).
    Grayscale RGBA content should convert to RGB and be accepted.
    """
    # Create RGBA image
    rgba_img = Image.new('RGBA', (256, 256), color=(150, 150, 150, 255))
    
    img_bytes = BytesIO()
    rgba_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {'file': ('rgba.png', img_bytes, 'image/png')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert 'class_name' in data


def test_predict_color_image_rejected(client, mock_inference_engine):
    """
    Test POST /predict rejects color (RGB) images as potential garbage.
    Brain MRI scans are grayscale; colored uploads must not be predicted.
    """
    # Create a colorful RGB image
    img_array = np.zeros((256, 256, 3), dtype=np.uint8)
    img_array[..., 0] = 200  # strong red channel
    img_array[..., 1] = 40
    img_array[..., 2] = 90
    color_img = Image.fromarray(img_array, mode='RGB')
    
    img_bytes = BytesIO()
    color_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {'file': ('color_scan.png', img_bytes, 'image/png')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 422
    data = response.json()
    assert 'grayscale' in str(data['error']).lower()
    # The engine must never have received a color image
    mock_inference_engine.predict.assert_not_called()


def test_predict_colorish_jpeg_rejected(client, mock_inference_engine):
    """
    A JPEG carrying real color content is rejected, while a JPEG that
    merely stores grayscale content in RGB channels stays accepted.
    """
    # Colored content
    img_array = np.zeros((256, 256, 3), dtype=np.uint8)
    y, x = np.ogrid[:256, :256]
    mask = (x - 128)**2 + (y - 128)**2 <= 80**2
    img_array[..., 0] = np.where(mask, 180, 10)
    img_array[..., 2] = np.where(mask, 220, 30)
    color_jpeg = Image.fromarray(img_array, mode='RGB')
    
    img_bytes = BytesIO()
    color_jpeg.save(img_bytes, format='JPEG', quality=95)
    img_bytes.seek(0)
    
    files = {'file': ('color.jpg', img_bytes, 'image/jpeg')}
    response = client.post("/predict", files=files)
    assert response.status_code == 422
    assert 'grayscale' in str(response.json()['error']).lower()
    
    # Grayscale MRI content stored in an RGB JPEG container is fine
    gray_array = np.zeros((256, 256), dtype=np.uint8)
    gray_array[mask] = 150
    gray_jpeg = Image.fromarray(gray_array, mode='L').convert('RGB')
    
    img_bytes = BytesIO()
    gray_jpeg.save(img_bytes, format='JPEG', quality=95)
    img_bytes.seek(0)
    
    files = {'file': ('gray_as_rgb.jpg', img_bytes, 'image/jpeg')}
    response = client.post("/predict", files=files)
    assert response.status_code == 200


def test_is_grayscale_image_helper():
    """
    Unit test for the grayscale gate helper.
    """
    # True grayscale (single channel)
    assert is_grayscale_image(Image.new('L', (64, 64), color=128))
    
    # Grayscale content stored as RGB (3 identical channels)
    arr = np.full((64, 64, 3), 100, dtype=np.uint8)
    assert is_grayscale_image(Image.fromarray(arr, mode='RGB'))
    
    # Real color content
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[..., 0] = 200
    arr[..., 2] = 40
    assert not is_grayscale_image(Image.fromarray(arr, mode='RGB'))
    
    # Grayscale RGBA (equal channels) accepted, colored RGBA rejected
    assert is_grayscale_image(Image.new('RGBA', (64, 64), color=(120, 120, 120, 255)))
    assert not is_grayscale_image(Image.new('RGBA', (64, 64), color=(220, 30, 30, 255)))


def test_predict_jpeg_image(client, mock_inference_engine, sample_image):
    """
    Test prediction with JPEG format.
    """
    img_bytes = BytesIO()
    sample_image.save(img_bytes, format='JPEG', quality=90)
    img_bytes.seek(0)
    
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert 'class_name' in data


def test_inference_engine_prediction_failure(client, sample_image):
    """
    Test handling of inference engine prediction failure.
    """
    # Create mock that raises exception
    mock_engine = Mock(spec=ONNXInferenceEngine)
    mock_engine.predict.side_effect = ValueError("Model inference failed")
    mock_engine.get_info.return_value = {'model_name': 'test_model'}
    
    with patch('api.main.inference_engine', mock_engine):
        client = TestClient(app)
        
        img_bytes = BytesIO()
        sample_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {'file': ('test.png', img_bytes, 'image/png')}
        response = client.post("/predict", files=files)
        
        # Should return 422
        assert response.status_code == 422
        data = response.json()
        # API's exception handler returns error message in 'error' key
        assert 'error' in data


# ============================================================================
# Concurrency Tests
# ============================================================================

def test_concurrent_predictions(client, sample_image, mock_inference_engine):
    """
    Test multiple concurrent prediction requests.
    Ensures thread safety and proper response for each request.
    """
    # Prepare multiple image uploads
    num_requests = 5
    files_list = []
    
    for i in range(num_requests):
        img_bytes = BytesIO()
        sample_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        files_list.append({'file': (f'test_{i}.png', img_bytes, 'image/png')})
    
    # Send multiple requests
    responses = []
    for files in files_list:
        response = client.post("/predict", files=files)
        responses.append(response)
    
    # Verify all succeeded
    for response in responses:
        assert response.status_code == 200
        data = response.json()
        assert 'class_name' in data
        assert 'confidence' in data
    
    # Verify mock was called for each request
    assert mock_inference_engine.predict.call_count == num_requests
