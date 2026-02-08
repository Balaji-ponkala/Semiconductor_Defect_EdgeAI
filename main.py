import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tf2onnx

def build_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(8, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    print("Initializing Edge-AI Training Pipeline...")
    model = build_model()
    
    # Dummy data for demonstration
    X_train = np.random.rand(10, 128, 128, 1).astype(np.float32)
    y_train = np.random.randint(0, 8, 10)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1)
    
    # Save ONNX
    spec = (tf.TensorSpec((None, 128, 128, 1), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path="model.onnx")
    print("SUCCESS: model.onnx generated.")
