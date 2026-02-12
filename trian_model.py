import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def create_and_train_model():
    # 1. Build CNN Architecture
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 Classes: Standing, Walking, Running
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 2. Create Synthetic Training Data (Simulating frames)
    # In a real project, you'd load images here.
    X_train = np.random.rand(300, 64, 64, 1)
    y_train = np.array([0] * 100 + [1] * 100 + [2] * 100)  # 0: Standing, 1: Walking, 2: Running

    print("--- Training CNN Model ---")
    model.fit(X_train, y_train, epochs=5, verbose=1)

    # 3. Save the result
    model.save('human_activity_model.h5')
    print("Model saved as human_activity_model.h5")


if __name__ == "__main__":
    create_and_train_model()