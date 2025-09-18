import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Initialize model as None
model = None

# Try to load model
try:
    model = load_model('deepfake_model.h5')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Web app
st.title("Deepfake Detector")
uploaded_file = st.file_uploader("Upload image", type=['jpg', 'png'])

if uploaded_file:
    try:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
        if img is None:
            st.error("Error: Could not read the image. Ensure it's a valid JPG or PNG.")
            st.stop()
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img / 255.0, axis=0)
        pred = model.predict(img)[0][0]
        if pred > 0.5:
            st.write("This is FAKE! Confidence:", round(pred, 2))
        else:
            st.write("This is REAL! Confidence:", round(1 - pred, 2))
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")