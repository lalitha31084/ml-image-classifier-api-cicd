import tensorflow as tf
import os

# Create directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Build a simple CNN architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Save the weights separately (This is the key fix!)
model.save_weights('models/model_weights.weights.h5')
# Save full model as fallback
model.save('models/my_classifier_model.h5')

print("✅ Files generated: models/model_weights.weights.h5 and models/my_classifier_model.h5")