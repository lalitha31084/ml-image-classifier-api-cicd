from fastapi import FastAPI, File, UploadFile, HTTPException
from src.model import load_model, preprocess_image, predict_image
import logging

app = FastAPI(title="ML Image Classifier API")

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        logging.error(f"Startup failed: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    contents = await file.read()
    preprocessed = preprocess_image(contents)
    result = predict_image(preprocessed)
    return result