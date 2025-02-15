import os
import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, video_file, frames_dir):
        self.video_file = video_file
        self.frames_dir = frames_dir
        self.cap = None
        self.frame_count = 0
        os.makedirs(self.frames_dir, exist_ok=True)

    def open_video(self):
        self.cap = cv2.VideoCapture(self.video_file)

        if not self.cap.isOpened():
            print('Cannot open video file')
            exit(0)

        print('Video file opened')
        print('Video Dimensions:', self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('Video FPS:', self.cap.get(cv2.CAP_PROP_FPS))

    def save_frames(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_in = frame[:, :, (2, 1, 0)]  # Reorder frame to RGB
            else:
                break

            frame_filename = os.path.join(self.frames_dir, f'frame_{self.frame_count}.png')
            cv2.imwrite(frame_filename, frame)
            self.frame_count += 1

            cv2.imshow('frame', frame_in)
            cv2.waitKey(300)
            cv2.destroyAllWindows()

    def close_video(self):
        self.cap.release()


class ImageProcessor:
    def __init__(self, image_name, image_dir, output_dir):
        self.image_name = image_name
        self.image_path = os.path.join(image_dir, f"{self.image_name}.png")
        self.output_dir = output_dir

    def preprocess_image(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Inverting the image
        inverted = cv2.bitwise_not(image)

        # Creating a kernel for detecting horizontal lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (110, 1))

        # Detecting white lines
        detected_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=2)

        # Removing detected lines
        lines_removed = cv2.subtract(inverted, detected_lines)

        # Invert back to original colors
        result = cv2.bitwise_not(lines_removed)

        # Save the processed image
        output_path = os.path.join(self.output_dir, 'image_after_shadow_removal.png')
        cv2.imwrite(output_path, result)

        print("Preprocessed image saved at:", output_path)


class VideoToImageProcessor:
    def __init__(self, video_file, frames_dir, image_name, image_dir, output_dir):
        self.video_processor = VideoProcessor(video_file, frames_dir)
        self.image_processor = ImageProcessor(image_name, image_dir, output_dir)

    def process_video(self):
        self.video_processor.open_video()
        self.video_processor.save_frames()
        self.video_processor.close_video()

    def process_image(self):
        self.image_processor.preprocess_image()
