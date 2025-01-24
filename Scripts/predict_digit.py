import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

class Digit_prediction:

    # Initializing the classifier using the pre-trained model
    def __init__(self, model_path):
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

    def draw_contour(self, image_path, out_dir, predict):
        # This function is to draw the contours

        # Load the image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply thresholding to create a binary image
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Morphological cleaning to refine binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned_image = cv2.dilate(cleaned_image, kernel, iterations=5)

        # Copy the original image to draw rectangles
        contoured_image = image.copy()

        for idx, contour in enumerate(contours):
            # Get the bounding box coordinates for each contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw the rectangle around each digit
            cv2.rectangle(contoured_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop and resize the digit region
            crop_img = image[y:y+h, x:x+w]
            rs_crop_img = cv2.resize(crop_img, (28, 28))

            # Save the cropped image
            digit_path = os.path.join(out_dir, f'digit_{idx}.png')
            cv2.imwrite(digit_path, rs_crop_img)

            # Predict the digit and display the result
            predicted_label = self.predict(rs_crop_img)
            print(f"Digit {idx}: Predicted Label = {predicted_label}")

        # Save and display the contoured image
        contoured_output_path = os.path.join(out_dir, 'Contoured.png')
        cv2.imwrite(contoured_output_path, contoured_image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Contoured Image")
        plt.imshow(cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


