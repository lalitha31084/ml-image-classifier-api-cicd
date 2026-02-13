# 🚀 Containerized ML Prediction API with CI/CD

This repository contains a production-grade RESTful API designed to serve an image classification model. The project demonstrates a complete MLOps workflow, bridging the gap between model development and real-world application using **FastAPI**, **Docker**, and **GitHub Actions**.

---
## 🛠 Technology Stack
* **API Framework:** FastAPI (Asynchronous Python)
* **Machine Learning:** TensorFlow/Keras, NumPy
* **Image Processing:** Pillow
* **Containerization:** Docker & Docker Compose
* **CI/CD:** GitHub Actions
* **Testing:** Pytest & TestClient

---
## 📂 Project Structure
```text
ml-image-classifier-api-cicd/
├── .github/workflows/   # CI/CD pipeline automation (main.yml)
├── src/                 # Source code
│   ├── main.py          # FastAPI application & endpoints
│   ├── model.py         # Inference & preprocessing logic
├── models/              # Pre-trained model artifacts (.h5)
├── tests/               # Unit and integration tests
├── predictions/         # Example JSON prediction outputs
├── Dockerfile           # Multi-stage optimized build
├── docker-compose.yml   # Orchestration for local development
└── requirements.txt     # Python dependencies
⚙️ Setup and Installation
1. Local Development (Manual)
Clone the repo:

Bash
git clone [https://github.com/lalitha31084/ml-image-classifier-api-cicd.git](https://github.com/lalitha31084/ml-image-classifier-api-cicd)
cd ml-image-classifier-api-cicd
Install dependencies:

Bash
pip install -r requirements.txt
Run the application:

Bash
uvicorn src.main:app --reload
2. Local Development (Docker Compose) - Recommended
Build and start the entire stack with a single command. This ensures the environment is identical to production:

Bash
docker-compose up --build
The API will be live at http://localhost:8000.

📡 API Usage
Health Check
Verify that the API is running and the model is successfully loaded into memory.

Endpoint: GET /health

Command: curl http://localhost:8000/health

Image Prediction
Submit an image file (JPG/PNG) for classification.

Endpoint: POST /predict

Command:

Bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_test_image.jpg'
🧪 Testing and CI/CD
Automated Testing
We use pytest with unittest.mock to ensure code quality and endpoint logic without requiring the heavy TensorFlow overhead during every test run.

Bash
pytest tests/
CI/CD Pipeline
The GitHub Actions workflow (main.yml) automates the following on every push to the main branch:

Code Checkout: Pulls the latest code.

Environment Setup: Configures Python 3.9 environment.

Testing: Executes the Pytest suite.

Docker Build: Builds the container image to verify no configuration errors.

Artifact Generation: Captures and saves example inference results in the predictions/ directory.

💡 Key Design Patterns
Singleton Model Loading: The Keras model is loaded once during the startup event to minimize inference latency.

Input Validation: Strict checking for image file types and dimensions using FastAPI's UploadFile.

Multi-Stage Docker Build: Uses a builder stage to keep the final production image size minimal and secure.

📝 Future Roadmap
Implement JWT-based Authentication for secure API access.

Integrate Prometheus/Grafana for monitoring model drift and latency.

Migrate to a GPU-optimized base image for high-throughput requirements.


---

### **Final Code Needed: `tests/test_api.py`**
To make sure your **GitHub Actions (CI/CD)** actually passes (shows a green checkmark), you must have the test file. 

**Would you like me to give you the code for `tests/test_api.py` now?**