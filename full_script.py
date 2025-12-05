import os  # For finding files
import cv2  # For reading images
import numpy as np  # For math with arrays
from sklearn.model_selection import train_test_split  # For splitting data
import tensorflow as tf  # AI library
from tensorflow.keras.models import Sequential  # For building model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Layer types
from tensorflow.keras.applications import ResNet50  # Pre-trained model

# Step 1: Preprocess Data
real_path = 'Real/'  # Path to real images folder
fake_path = 'Fake/'  # Path to fake images folder

# Function to load images
def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(label)
    return images, labels

# Load real and fake images
print("Loading images...")
real_images, real_labels = load_images(real_path, 0)
fake_images, fake_labels = load_images(fake_path, 1)

# Combine and normalize
all_images = np.array(real_images + fake_images)
all_labels = np.array(real_labels + fake_labels)
all_images = all_images / 255.0  # Normalize pixel values to 0-1

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.3, random_state=42)
print("Data ready! Train images:", len(X_train), "Test images:", len(X_test))

# Step 2: Build Model
print("Building model...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train Model
print("Training model... This may take a while.")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Step 4: Save Model
model.save('deepfake_model.h5')
print("Model trained and saved as deepfake_model.h5!")