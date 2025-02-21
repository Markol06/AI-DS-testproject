import sys
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from mnist_classifier_interface import MnistClassifierInterface


# Add the path to the project root folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

print("Updated sys.path:", sys.path)


class RandomForestClassifierMnist(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST.
    """

    def __init__(self):
        """
        Initialization of the Random Forest model.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        """
        Training the model on MNIST.
        """
        # Loading MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Convert the image to vectors (28x28 → 784)
        X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
        X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

        # Training the model
        self.model.fit(X_train, y_train)

        # Evaluate accuracy on test data
        accuracy = self.model.score(X_test, y_test)
        print(f"✅ Accuracy: {accuracy:.4f}")

    def predict(self, image):
        """
        Predict a digit for a single MNIST image.

        :param image: 2D array (28x28) – handwritten digit
        :return: predicted class (0-9)
        """
        image = np.array(image).reshape(1, -1) / 255.0  # Convert to vector
        return self.model.predict(image)[0]


if __name__ == "__main__":
    print("✅ RandomForestClassifierMnist loaded successfully!")

    # Initialize and train the model
    rf_classifier = RandomForestClassifierMnist()
    rf_classifier.train()

    print("✅ The Random Forest model has been successfully trained!")
