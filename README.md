# Handwritten Digit Recognition System using CNN

## 📌 Overview
This project implements a **Handwritten Digit Recognition System** using **Convolutional Neural Networks (CNNs)**. The system is trained on the **MNIST dataset**, which consists of grayscale images of handwritten digits (0-9). The model achieves an impressive **98% accuracy** on the test dataset.

## ✨ Features
- Utilizes a **CNN architecture** for feature extraction and classification.
- Trained on the **MNIST dataset** consisting of 60,000 training and 10,000 test images.
- Achieves **98% accuracy**, making it highly efficient for digit recognition tasks.
- Implements **Keras & TensorFlow** for deep learning.
- Uses **OpenCV & Matplotlib** for image processing and visualization.
- Supports real-time handwritten digit prediction.

## 📂 Dataset
The project uses the **MNIST dataset**, which consists of:
- **60,000 training images** and **10,000 test images**.
- Grayscale images of **28x28 pixels**.
- Labels from **0 to 9** representing handwritten digits.

## 🏗️ Model Architecture
The CNN model follows this structure:
1. **Convolutional Layer (Conv2D - 32 filters, 3x3, ReLU, padding=same)** - Extracts important spatial features.
2. **Max-Pooling Layer (2x2)** - Reduces dimensionality while preserving essential features.
3. **Convolutional Layer (Conv2D - 64 filters, 3x3, ReLU, padding=same)**.
4. **Max-Pooling Layer (2x2)**.
5. **Convolutional Layer (Conv2D - 128 filters, 3x3, ReLU, padding=same)**.
6. **Flatten Layer** - Converts 2D feature maps into a 1D vector.
7. **Fully Connected (Dense) Layer - 128 neurons, ReLU activation**.
8. **Fully Connected (Dense) Layer - 64 neurons, ReLU activation**.
9. **Output Layer (Dense - 10 neurons, Softmax activation)** - Predicts the digit (0-9).

### 📌 Model Summary
- **Input Layer**: 28x28 grayscale image.
- **Conv2D (ReLU) + MaxPooling** layers.
- **Fully Connected (Dense) Layers with Dropout** to prevent overfitting.
- **Softmax Activation** for multi-class classification.
- **Optimizer**: RMSprop (Root Mean Square Propagation for non-stationary objectives).
- **Loss Function**: Categorical Crossentropy.
- **Training**: 5 epochs with a batch size of 64.

## 🚀 Installation & Usage
### Prerequisites
Ensure you have the following installed:
- Python (>=3.7)
- TensorFlow & Keras
- NumPy, OpenCV, Matplotlib


## 📊 Results
- The trained CNN model achieves **98% accuracy** on the MNIST test dataset.
- The model can accurately classify handwritten digits with minimal error.
- Below is a sample prediction visualization:


## 🤖 Applications
- Digit recognition in **banking and finance (check processing)**.
- Optical Character Recognition (**OCR**) for digitized documents.
- Automated form processing.
- Educational tools for learning handwriting recognition.

