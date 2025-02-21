import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from mnist_classifier_interface import MnistClassifierInterface


class NeuralNetworkClassifierMnist(MnistClassifierInterface):
    """
    Feed-Forward Neural Network (MLP) for MNIST classification.
    """

    def __init__(self):
        """
        Initializing the neural network model.
        """
        self.model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dropout(0.2),  # Add Dropout for regularization
            Dense(10, activation='softmax')  # Output layer for 10 classes (0-9)
        ])
        self.model.compile(optimizer=Adam(),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self):
        """
        Training the model on the MNIST dataset.
        """
        # Load MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Normalizing the image
        X_train, X_test = X_train / 255.0, X_test / 255.0

        # One-hot encoding of tags
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

        # EarlyStopping: if `val_loss` does not decrease for 3 epochs – stop
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Training the model
        history = self.model.fit(X_train, y_train,
                                 epochs=20, batch_size=32,
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
        image = np.array(image).reshape(1, 28, 28) / 255.0  # Normalizing
        prediction = np.argmax(self.model.predict(image))
        return prediction


if __name__ == "__main__":
    classifier = NeuralNetworkClassifierMnist()
    classifier.train()