import io
from fastapi.testclient import TestClient
from PIL import Image
from src.main import app

client = TestClient(app)

def test_health_check_endpoint():
    """Test the /health endpoint to match the actual API response."""
    response = client.get("/health")
    assert response.status_code == 200
    # Updated to match your API's actual return: {'status': 'healthy'}
    assert response.json() == {"status": "healthy"}

def test_predict_success():
    """Test /predict with a real dummy image."""
    dummy_image = Image.new('RGB', (64, 64), color='blue')
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")}
    )
    assert response.status_code == 200
    assert "class_label" in response.json()
    assert response.json()["status"] == "success"

def test_predict_invalid_file_type():
    """Test that the API rejects non-image files."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    # Updated to match your API's actual error message
    assert "File provided is not an image" in response.json()["detail"]