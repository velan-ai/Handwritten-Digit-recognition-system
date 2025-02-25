import os
import cv2
import numpy as np
import tensorflow as tf
import keras

class DigitClassifier:
    def __init__(self, model_path: str):
        # Initializing classifier with the trained model
        self.model = tf.keras.models.load_model(model_path)

    def grayscale(self, file_path: str) -> np.ndarray:
        # Grayscaling the pre-processed image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Reshape for model input
        return img

    def predict_digit(self, file_path: str) -> int:
        """
        Predicts the digit in the given image file.
        :param file_path: Path to the image file.
        :return: Predicted digit label.
        """
        img = self.grayscale(file_path)
        prediction = self.model.predict(img)
        predicted_label = prediction.argmax()
        return predicted_label


if __name__ == "__main__":
    model_path = "model_001.h5"
    file_path = input("Enter the image file path: ").replace('/', '\\')

    classifier = DigitClassifier(model_path)
    predicted_label = classifier.predict_digit(file_path)
    print("This is:", predicted_label)
