import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Define dataset path
data_dir = "dataset"
categories = os.listdir(data_dir)  # Auto-detect categories

# Image settings
img_size = 128

# Load images & labels
def load_data():
    images, labels = [], []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_idx = categories.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array, (img_size, img_size))
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading image {img}: {e}")

    return np.array(images), np.array(labels)

X, y = load_data()
X = X / 255.0  # Normalize

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')  # Output layer for 10+ categories
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save("coffee_leaf_model.h5")
np.save("categories.npy", categories)  # Save category names
