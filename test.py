import numpy as np
import os
# Set Keras backend to TensorFlow before importing
os.environ['KERAS_BACKEND'] = 'tensorflow'
# Import Keras 3.x directly (it will use TensorFlow as backend)
try:
    import keras
    from keras.models import load_model
    print(f"Using Keras {keras.__version__} with TensorFlow backend")
except ImportError:
    try:
        from tensorflow.keras.models import load_model
        print("Using TensorFlow Keras")
    except Exception as e:
        print(f"Error: Could not import keras or tensorflow.keras: {e}")
        print("This script requires TensorFlow/Keras to load the model.")
        exit(1)
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load test data
try:
    with open('test_data.pkl', 'rb') as f:
        X_test, y_test = pickle.load(f)
    print("Loaded test data:", len(X_test), "images")
except Exception as e:
    print(f"Error loading test_data.pkl: {e}")
    print("Run preprocess.py first to save test_data.pkl")
    exit(1)

# Load model
try:
    # Load model with Keras 3.x (which can load models saved with Keras 3.x)
    model = load_model('deepfake_model.h5', compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure deepfake_model.h5 exists in the current directory")
    print("2. Check that Keras 3.x is installed and using TensorFlow backend")
    exit(1)

# Predict
print("Testing model...")
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Show results
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
print("Accuracy:", accuracy_score(y_test, y_pred))