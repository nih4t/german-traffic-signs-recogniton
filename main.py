import os
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import cv2

import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = 'archive/'
TRAIN_CSV = os.path.join(DATASET_PATH, 'Train.csv')
TEST_CSV = os.path.join(DATASET_PATH, 'Test.csv')
IMG_SIZE = 32
NUM_CLASSES = 43

def load_data(csv_file, dataset_path):
    df = pd.read_csv(csv_file)
    images = []
    labels = []

    for index, row in df.iterrows():
        img_path = os.path.join(dataset_path, row['Path'])
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            images.append(img)
            labels.append(row['ClassId'])
        else:
            print(f'Warning: Could not load image {img_path}')
    return np.array(images), np.array(labels)

print('Loading training data...')
X_train, y_train = load_data(TRAIN_CSV, DATASET_PATH)
print('Loading test data...')
X_test, y_test = load_data(TEST_CSV, DATASET_PATH)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data augmentation to improve generalization
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model with data augmentation
print("Training model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=15,  # Increased epochs for better training
    validation_data=(X_val, y_val)
)

# Evaluate the model on test data
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the model
model.save('traffic_sign_cnn.h5')
print("Model saved as traffic_sign_cnn.h5")

# Example: Predict on a single test image
sample_img = X_test[0:1]  # Take the first test image
prediction = model.predict(sample_img)
predicted_class = np.argmax(prediction, axis=1)[0]
print(f"Predicted class for sample image: {predicted_class}")

# Optional: Load class names from Meta.csv or define manually
# Example manual class names (subset, complete list available in GTSRB documentation)
class_names = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    # Add more from Meta.csv or GTSRB documentation
}
if predicted_class in class_names:
    print(f"Predicted sign: {class_names[predicted_class]}")
