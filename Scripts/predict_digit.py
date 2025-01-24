import cv2
import numpy as np
import tensorflow as tf

class Digit_prediction:

    # Initializing the classifier using the pre-trained model
    def __init__(self):
        self.model = tf.keras.models.load_model(model_path)

    def pre_process(self,image): # image = filepath of the image
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(image, (28, 28))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self,image):
        img = self.pre_process(image)
        prediction = self.model.predict(img)
        predicted_label = np.argmax(prediction)
        return predicted_label

    def

