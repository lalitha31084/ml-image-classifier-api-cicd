import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None
IMAGE_SIZE = (64, 64) 
CLASS_LABELS = [f'class_{i}' for i in range(10)]

def load_model():
    global MODEL
    if MODEL is None:
        try:
            # Rebuild architecture manually to bypass Keras 3 metadata errors
            MODEL = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            weights_path = "models/model_weights.weights.h5"
            if os.path.exists(weights_path):
                MODEL.load_weights(weights_path)
                logger.info("✅ SUCCESS: Model architecture rebuilt and weights loaded.")
            else:
                logger.warning("Weights file not found, trying full model load...")
                MODEL = tf.keras.models.load_model("models/my_classifier_model.h5", compile=False)
        except Exception as e:
            logger.error(f"❌ CRITICAL LOAD ERROR: {e}")
            raise e
    return MODEL

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0).astype(np.float32)
    except Exception as e:
        logger.error(f"PREPROCESSING ERROR: {e}")
        raise ValueError(f"Invalid image format: {e}")

def predict_image(preprocessed_image: np.ndarray):
    model = load_model()
    predictions = model.predict(preprocessed_image)
    predicted_class_idx = int(np.argmax(predictions, axis=1)[0])
    return {
        "class_label": CLASS_LABELS[predicted_class_idx],
        "probabilities": [round(p, 4) for p in predictions[0].tolist()],
        "status": "success"
    }