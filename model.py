from tensorflow import keras
from keras.datasets import mnist
from tensorflow.keras import layers
from keras.utils import to_categorical

import os

model2 = keras.Sequential()
model2.add(layers.Dense(256, activation='sigmoid', input_shape=(784,)))
model2.add(layers.Dense(128, activation='sigmoid'))
model2.add(layers.Dense(10, activation='softmax'))
model2.summary()

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the input data from 28x28 to 784
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalize the input data to a range of [0, 1]
X_train = X_train / 255
X_test = X_test / 255

# One-hot encode the output labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

model2.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose=2)

# save_path = "C:/Users/user/Desktop/Sem 8/Big Data Lab/Assignment 6"
# model2.save(os.path.join(save_path, "my_model.h5"))
model2.save('mnist_model.keras')