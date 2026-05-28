import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Set paths to your folders
real_path = 'Real/'  # Update if your real images are in a different folder
fake_path = 'Fake/'  # Update if your fake images are in a different folder

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
real_images, real_labels = load_images(real_path, 0)
fake_images, fake_labels = load_images(fake_path, 1)

# Combine and normalize
all_images = np.array(real_images + fake_images)
all_labels = np.array(real_labels + fake_labels)
all_images = all_images / 255.0

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.3, random_state=42)
print("Data ready! Train images:", len(X_train), "Test images:", len(X_test))

# Save test data
import pickle
with open('test_data.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)
print("Test data saved as test_data.pkl")