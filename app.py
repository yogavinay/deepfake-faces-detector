import os
import time

import streamlit as st
# Set Keras backend to TensorFlow before importing
os.environ['KERAS_BACKEND'] = 'tensorflow'
# Import Keras 3.x directly (it will use TensorFlow as backend)
try:
    import keras
    from keras.models import load_model
    import tensorflow as tf
except ImportError:
    try:
        from tensorflow.keras.models import load_model
        import tensorflow as tf
    except Exception as e:
        st.error(f"Error: Could not import keras or tensorflow.keras: {e}")
        st.stop()
import cv2
import numpy as np

# Page configuration
st.set_page_config(
    page_title="True Vision Detector",
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

def image_basic_stats(img_rgb):
    """Return simple image stats to support forensic cues."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation_mean = float(np.mean(hsv[:, :, 1]))
    brightness_mean = float(np.mean(hsv[:, :, 2]))
    return {
        "sharpness": sharpness,
        "saturation_mean": saturation_mean,
        "brightness_mean": brightness_mean,
    }


def _find_last_conv_layer(model):
    """Pick the last convolution-like layer for Grad-CAM."""
    for layer in reversed(model.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except Exception:
            continue
    return None


def summarize_heatmap_regions(heatmap):
    """Return simple quadrant scores to describe attention areas."""
    h, w = heatmap.shape
    regions = {
        "top-left": float(np.mean(heatmap[0:h // 2, 0:w // 2])),
        "top-right": float(np.mean(heatmap[0:h // 2, w // 2:])),
        "bottom-left": float(np.mean(heatmap[h // 2:, 0:w // 2])),
        "bottom-right": float(np.mean(heatmap[h // 2:, w // 2:])),
    }
    return sorted(regions.items(), key=lambda x: x[1], reverse=True)


def generate_grad_cam(model, img_array, layer_name=None):
    """Return a Grad-CAM heatmap (values 0-1) for the given image batch (1, h, w, 3).
    Falls back to input-gradient saliency if Grad-CAM fails."""
    img_array = tf.cast(img_array, tf.float32)
    target_layer = layer_name or _find_last_conv_layer(model)

    # Try Grad-CAM on the last conv layer
    if target_layer is not None:
        try:
            # Get layer output - works with both Keras 2.x and 3.x
            try:
                layer = model.get_layer(target_layer)
            except:
                layer = None
            if layer is not None:
                grad_model = tf.keras.models.Model(
                    [model.inputs],
                    [layer.output, model.output],
                )
            else:
                raise Exception("Layer not found")

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, 0]

            grads = tape.gradient(loss, conv_outputs)
            if grads is not None:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]
                heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
                heatmap = np.maximum(heatmap, 0)
                if np.max(heatmap) > 0:
                    heatmap /= np.max(heatmap)
                    return heatmap.numpy()
        except Exception:
            pass

    # Fallback: input-gradient saliency
    try:
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            preds = model(img_array, training=False)
            loss = preds[:, 0]
        grads = tape.gradient(loss, img_array)
        saliency = tf.reduce_mean(tf.abs(grads), axis=-1)[0]
        saliency = saliency.numpy()
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        return saliency
    except Exception:
        return None


def overlay_heatmap_on_image(img_rgb, heatmap, alpha=0.45):
    """Overlay a heatmap onto the original RGB image."""
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return heatmap_resized, overlay


def forensic_notes(pred_score, threshold, stats):
    """Return short textual cues explaining the decision with image stats."""
    cues = []
    verdict_fake = pred_score > threshold

    if verdict_fake:
        cues.append("Model score is above the fake threshold.")
        cues.append("Look for blending artifacts near edges and facial landmarks.")
        if stats["sharpness"] < 50:
            cues.append("Image is soft/blurred; low sharpness may hide artifacts.")
        if stats["saturation_mean"] < 40:
            cues.append("Unusually low saturation — possible color flattening.")
    else:
        cues.append("Model score is below the fake threshold.")
        cues.append("Texture and lighting appear consistent across the face.")
        if stats["sharpness"] > 120:
            cues.append("High sharpness; details look consistent.")
    return cues


# Load model
model = load_deepfake_model()

# Web app UI
st.title("🔍 True Vision Detector")
st.markdown("Upload an image to detect if it's a deepfake or real image.")
st.markdown("---")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses a deep learning model (ResNet50) to detect deepfake images.

    **How it works:**
    1) Upload an image (JPG or PNG)
    2) The model analyzes the image + creates an attention heatmap
    3) You get decision, confidence, attention map, and forensic cues
    """)
    st.markdown("---")
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
            st.image(img_rgb, caption="Your uploaded image")
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
                start = time.time()
                pred = model.predict(img_normalized, verbose=0)[0][0]
                duration = (time.time() - start) * 1000  # ms

            # Display results with adjustable threshold
            verdict_fake = pred > threshold
            if verdict_fake:
                st.error("🚨 **FAKE DETECTED**")
                confidence = round(pred * 100, 2)
                st.metric("Fake Confidence", f"{confidence}%", delta=f"{(pred-threshold)*100:.1f}% over threshold")
                st.progress(float(pred))
                st.warning("⚠️ This image appears to be a deepfake or manipulated image.")
            else:
                st.success("✅ **REAL IMAGE**")
                confidence = round((1 - pred) * 100, 2)
                st.metric("Real Confidence", f"{confidence}%", delta=f"{(threshold-pred)*100:.1f}% margin")
                st.progress(float(1 - pred))
                st.info("ℹ️ This image appears to be authentic.")

            st.caption(f"Prediction score: {pred:.4f} (threshold: {threshold:.2f}) · Inference: {duration:.1f} ms")

            # Always-on forensic cues and stats (no heatmap dependency)
            stats = image_basic_stats(img_rgb)

            st.markdown("### 🕵️ Forensic cues")
            cues = forensic_notes(pred, threshold, stats)
            for c in cues:
                st.markdown(f"- {c}")
            st.caption("Quick checks: edges, facial landmarks, lighting, and color balance.")

            st.markdown("### 📊 Image stats")
            stat_cols = st.columns(3)
            stat_cols[0].metric("Sharpness (Laplacian var)", f"{stats['sharpness']:.1f}")
            stat_cols[1].metric("Saturation (mean)", f"{stats['saturation_mean']:.1f}")
            stat_cols[2].metric("Brightness (mean)", f"{stats['brightness_mean']:.1f}")

            st.info("All cues and stats are shown together for quick review.")

            st.markdown("### 🔍 Detailing (attention overlay)")
            heatmap = generate_grad_cam(model, img_normalized)
            if heatmap is None:
                st.warning("Detailing unavailable for this model architecture.")
            else:
                heatmap_resized, overlay = overlay_heatmap_on_image(img_rgb, heatmap)
                colh1, colh2 = st.columns(2)
                with colh1:
                    st.image(overlay, caption="Detected areas (overlay)")
                with colh2:
                    st.image(heatmap_resized, caption="Attention intensity (0-1)")

                st.markdown("**Details spotted:**")
                for name, score in summarize_heatmap_regions(heatmap)[:3]:
                    st.markdown(f"- {name}: attention {score:.3f} — inspect boundaries/texture here.")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)
else:
    st.info("👆 Please upload an image to get started!")