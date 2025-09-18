import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load test data
try:
    with open('test_data.pkl', 'rb') as f:
        X_test, y_test = pickle.load(f)
    print("Loaded test data:", len(X_test), "images")
except:
    print("Error: Run preprocess.py first to save test_data.pkl")
    exit()

# Load model
model = load_model('deepfake_model.h5')

# Predict
print("Testing model...")
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Show results
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
print("Accuracy:", accuracy_score(y_test, y_pred))