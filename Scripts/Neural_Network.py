from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical


class NN_Train:
    def __init__(self):
        self.model = None
        self.load_data()
        self.preprocess_data()
        self.build_model()

    def load_data(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()

    def preprocess_loaded_data(self):
        self.train_images = self.train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
        self.test_images = self.test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)

    def NN_design(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=5, batch_size=64):
        self.model.fit(self.train_images, self.train_labels, epochs=epochs, batch_size=batch_size)

    def evaluate(self):
        return self.model.evaluate(self.test_images, self.test_labels)

    def save_model(self, filename='model_001.h5'):
        self.model.save(filename)


if __name__ == "__main__":
    classifier = NN_Train()
    classifier.train()
    classifier.evaluate()
    classifier.save_model()
