import os

import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="centered"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_deepfake_model():
    """Load the deepfake detection model from local file (no cloud, no download)."""
    model_path = 'deepfake_model.h5'

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Train the model locally using 'python full_script.py' or 'python model.py' to create deepfake_model.h5.")
        return None

    try:
        model = load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully from local file.")
        return model
    except Exception as e:
        st.error(f"Error loading local model: {str(e)}")
        return None

# Load model
model = load_deepfake_model()

# Web app UI
st.title("🔍 Deepfake Detector")
st.markdown("Upload an image to detect if it's a deepfake or real image.")
st.markdown("---")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses a deep learning model (ResNet50) to detect deepfake images.
    
    **How it works:**
    1. Upload an image (JPG or PNG)
    2. The model analyzes the image
    3. Get instant results with confidence score
    """)
    st.markdown("---")
    # Threshold slider for decision boundary
    threshold = st.slider(
        "Decision threshold (higher = more strict for FAKE)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )
    st.caption("Predictions above this threshold are labeled as FAKE.")
    st.markdown("---")
    if model is not None:
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Model not loaded")

# Main content
if model is None:
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload a JPG or PNG image to analyze"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Uploaded Image")
        # Convert to RGB for display (OpenCV uses BGR)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is not None:
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Your uploaded image", use_container_width=True)
        else:
            st.error("Error: Could not read the image. Please ensure it's a valid image file.")
            st.stop()
    
    with col2:
        st.subheader("🔍 Analysis Result")
        
        # Process image for prediction
        try:
            # Resize and normalize
            img_resized = cv2.resize(img, (224, 224))
            img_normalized = np.expand_dims(img_resized / 255.0, axis=0)
            
            # Show loading spinner
            with st.spinner("Analyzing image..."):
                pred = model.predict(img_normalized, verbose=0)[0][0]
            
            # Display results with adjustable threshold
            if pred > threshold:
                st.error("🚨 **FAKE DETECTED**")
                confidence = round(pred * 100, 2)
                st.metric("Fake Confidence", f"{confidence}%")
                
                # Progress bar
                st.progress(float(pred))
                
                st.warning("⚠️ This image appears to be a deepfake or manipulated image.")
            else:
                st.success("✅ **REAL IMAGE**")
                confidence = round((1 - pred) * 100, 2)
                st.metric("Real Confidence", f"{confidence}%")
                
                # Progress bar
                st.progress(float(1 - pred))
                
                st.info("ℹ️ This image appears to be authentic.")
            
            st.markdown("---")
            st.caption(f"Prediction score: {pred:.4f} (threshold: {threshold:.2f})")
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)
else:
    st.info("👆 Please upload an image to get started!")