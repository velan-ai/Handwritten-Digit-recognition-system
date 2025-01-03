from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical



(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32')/255


train_labels = to_categorical(train_labels) # one-hot encoding (creates binary matrices 'helps in simplifying the compuataion of ')
test_labels = to_categorical(test_labels)  #  cross entropy loss entropy

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', padding = 'same', input_shape = (28,28,1)))  # padding creates a border stricture
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu', padding = 'same'))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation = 'relu'))
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10, activation= 'softmax'))
model.compile(optimizer = 'rmsprop',                  # used for non-stationary noisy objectives (Root mean square propagation)
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size = 64)

model.evaluate(test_images, test_labels)

model.save('model_001.h5')