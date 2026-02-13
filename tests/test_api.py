import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import io
from PIL import Image
from src.main import app

# Initialize the TestClient
client = TestClient(app)

def test_health_check_endpoint():
    """Test the /health endpoint for a 200 OK status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is healthy and model is loaded."}

@patch('src.model.predict_image')
@patch('src.model.preprocess_image')
def test_predict_success_with_mocked_model(mock_preprocess_image, mock_predict_image):
    """Test /predict using mocked model logic to ensure API flow works."""
    # Setup mocks
    mock_preprocess_image.return_value = "mock_preprocessed_array"
    mock_predict_image.return_value = {
        "class_label": "dog", 
        "probabilities": [0.05, 0.95]
    }

    # Create a dummy image in memory
    dummy_image = Image.new('RGB', (64, 64), color='blue')
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Send request
    response = client.post(
        "/predict", 
        files={"file": ("test_image.png", img_byte_arr, "image/png")}
    )

    # Verify results
    assert response.status_code == 200
    assert response.json()["class_label"] == "dog"
    mock_preprocess_image.assert_called_once()
    mock_predict_image.assert_called_once()

def test_predict_invalid_file_type():
    """Test that the API rejects non-image files."""
    response = client.post(
        "/predict", 
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only image files" in response.json()["detail"]

def test_predict_missing_file():
    """Test that the API handles empty requests."""
    response = client.post("/predict")
    # FastAPI returns 422 Unprocessable Entity for missing required fields
    assert response.status_code == 422