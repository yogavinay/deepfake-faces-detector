# Deepfake Detector

A machine learning project for detecting deepfake images using computer vision and deep learning techniques.

## Project Structure

- `app.py` - Main Streamlit web application
- `model.py` - Model architecture and training code
- `preprocess.py` - Data preprocessing utilities
- `full_script.py` - Complete training pipeline
- `test.py` - Model testing and evaluation
- `dataset.csv` - Dataset metadata
- `.gitignore` - Git ignore file to exclude large files

## Features

- Deepfake image detection using CNN models
- Web interface built with Streamlit
- Data preprocessing and augmentation
- Model training and evaluation pipeline

## Setup

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install tensorflow streamlit opencv-python numpy pandas
   ```
3. Run the web application:
   ```bash
   streamlit run app.py
   ```

## Dataset

The project uses a dataset of real and fake images. The dataset folders (`Real/` and `Fake/`) are excluded from the repository due to size constraints.

## Model

The trained model (`deepfake_model.h5`) is excluded from the repository due to size constraints. You can train your own model using the provided scripts.

## Usage

1. **Training**: Run `python full_script.py` to train the model
2. **Testing**: Run `python test.py` to evaluate the model
3. **Web App**: Run `streamlit run app.py` to use the web interface

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- OpenCV
- NumPy
- Pandas

## Note

Large files (model weights, datasets) are excluded from this repository. You'll need to:
- Train your own model or obtain the trained model separately
- Download or prepare your own dataset of real and fake images
