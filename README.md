
# 🚦 German Traffic Sign Recognition

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify German traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. Built with TensorFlow/Keras in Python, the model uses data augmentation to improve generalization and includes scripts for data preprocessing, training, evaluation, and prediction. The CNN accurately classifies 43 unique traffic sign categories, handling real-world variations such as lighting changes and occlusions.

## Table of Contents

- Features  
- Dataset  
- Requirements  
- Installation  
- Usage  
- Model Architecture  
- Data Augmentation  
- Training and Evaluation  
- Results  
- Contributing  
- License  
- Acknowledgments

## Features

- CNN Model: Custom CNN for classifying 43 traffic sign classes.  
- Data Augmentation: Rotation, zoom, width/height shift, shear.  
- Preprocessing: Resizes images to 32x32, normalizes pixel values, splits data.  
- Evaluation: Reports test accuracy and supports single-image predictions.  
- Class Mapping: Partial class name mapping for predictions, extendable via Meta.csv.

## Dataset

The GTSRB dataset contains over 50,000 RGB images of German traffic signs across 43 classes, with variations in size, lighting, and orientation.

- Training Set: ~39,209 images (Train.csv)  
- Test Set: ~12,630 images (Test.csv)  
- Metadata: Partial class mapping included; full mapping in Meta.csv.

Download the dataset from the INI Benchmark Website and place it in the `archive/` directory.

Download the dataset:
- Official site: [https://benchmark.ini.rub.de/gtsrb_news.html](https://benchmark.ini.rub.de/gtsrb_news.html)

## Requirements

- Python 3.8+  
- TensorFlow 2.10+  
- Keras  
- NumPy  
- Pandas  
- OpenCV  
- Scikit-learn

## Installation

```bash
git clone https://github.com/nih4t/german-traffic-signs-recogniton
cd german-traffic-sign-recognition

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

Download and extract the GTSRB dataset into the `archive/` folder, ensuring `Train.csv` and `Test.csv` are inside.

## Usage

Run the main script to preprocess data, train, evaluate, and predict:

```bash
python main.py
```

This will:

- Load and preprocess images from Train.csv and Test.csv  
- Train the CNN with data augmentation for 15 epochs  
- Evaluate the model on the test set and print accuracy  
- Save the model as `traffic_sign_cnn.h5`  
- Predict the class of a sample test image

## Model Architecture

- Input: 32x32x3 RGB images  
- Conv2D(32 filters, 3x3) + ReLU → MaxPooling(2x2)  
- Conv2D(64 filters, 3x3) + ReLU → MaxPooling(2x2)  
- Conv2D(128 filters, 3x3) + ReLU → MaxPooling(2x2)  
- Flatten → Dense(256) + ReLU → Dropout(0.5) → Dense(43) + Softmax  
- Optimizer: Adam  
- Loss: Categorical Cross-entropy  
- Metrics: Accuracy

## Data Augmentation

Applied via `ImageDataGenerator` with:

- Rotation ±10°  
- Zoom ±10%  
- Width/Height Shift ±10%  
- Shear 0.1 radians  
- Fill Mode: nearest

## Training and Evaluation

- Epochs: 15  
- Batch Size: 32  
- Optimizer: Adam  
- Validation Split: 20%

Final validation accuracy ~99.31%, test accuracy ~95.65%.

## Results

- Test Accuracy: 95.65%  
- Validation Accuracy: 99.31%  
- Model saved as `traffic_sign_cnn.h5`  
- Sample prediction with class mapping (e.g., "Speed limit (20km/h)")

## Contributing

Contributions are welcome!

- Fork the repo  
- Create a branch  
- Commit your changes  
- Push the branch  
- Open a pull request

Please follow PEP8 and include tests.

## Acknowledgments

- GTSRB dataset creators: Institute of Neural Information Processing, Ulm University  
- TensorFlow & Keras  
- OpenCV, NumPy, Pandas libraries
