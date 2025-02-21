import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from mnist_classifier_interface import MnistClassifierInterface


class CnnClassifierMnist(MnistClassifierInterface):
    """
    Convolutional Neural Network (CNN) for MNIST classification.
    """

    def __init__(self):
        """
        CNN model initialization.
        """
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),  # Add regularization

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer=Adam(),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self):
        """
        CNN training on the MNIST dataset.
        """
        # Loading MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Convert the data to (28,28,1) format for CNN
        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

        # One-hot encoding of tags
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

        # EarlyStopping: stop training after 2 epochs without improving `val_loss`
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        # Training the model
        history = self.model.fit(X_train, y_train,
                                 epochs=10, batch_size=32,
                                 validation_data=(X_test, y_test),
                                 callbacks=[early_stopping])

        # Accuracy assessment on test data
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=2)
        print(f"✅ Accuracy: {test_acc:.4f}")

    def predict(self, image):
        """
        Predict a digit for a single MNIST image.

        :param image: 2D array (28x28) – handwritten digit
        :return: predicted class (0-9)
        """
        image = np.array(image).reshape(1, 28, 28, 1) / 255.0  # Normalizing
        prediction = np.argmax(self.model.predict(image))
        return prediction


if __name__ == "__main__":
    classifier = CnnClassifierMnist()
    classifier.train()