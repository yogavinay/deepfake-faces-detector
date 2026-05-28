import tensorflow as tf  # AI library
from tensorflow.keras.models import Sequential  # To build model layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Layer types
from tensorflow.keras.applications import ResNet50  # Pre-trained model

# Assuming you ran preprocess.py first, or copy those variables here if combining files

# Build model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Load smart pre-trained base
base_model.trainable = False  # Don't change it yet

model = Sequential([  # Stack layers
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),  # Summarize features
    Dense(128, activation='relu'),  # Thinking layer
    Dropout(0.5),  # Prevent overfitting (randomly ignore some)
    Dense(1, activation='sigmoid')  # Output: 0-1 probability (fake if >0.5)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Set up learning

# Train
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)  # Train for 5 rounds

# Save model
model.save('deepfake_model.h5')
print("Model trained and saved!")