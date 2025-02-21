import sys
import os

models_path = os.path.abspath(os.path.dirname(__file__))
if models_path not in sys.path:
    sys.path.append(models_path)

from random_forest_classifier import RandomForestClassifierMnist
from neural_network_classifier import NeuralNetworkClassifierMnist
from cnn_classifier import CnnClassifierMnist


class MnistClassifier:
    """
    Manager for selecting and running MNIST models (CNN, NN, RF).
    """

    def __init__(self, model_type):
        """
        Initialize the selected model.

        :param model_type: Model name ('cnn', 'nn', 'rf')
        """
        self.model = None

        if model_type == "cnn":
            self.model = CnnClassifierMnist()
        elif model_type == "nn":
            self.model = NeuralNetworkClassifierMnist()
        elif model_type == "rf":
            self.model = RandomForestClassifierMnist()
        else:
            raise ValueError("❌ Incorrect model type! Use: 'cnn', 'nn' або 'rf'.")

        print(f"✅ Model selected: {model_type.upper()}")

    def train(self):
        """
        Starts training the selected model.
        """
        if self.model:
            self.model.train()
        else:
            print("❌ The model is not initialized.")

    def predict(self, image):
        """
        Predicts a class for an MNIST image.

        :param image: 2D array (28x28) of images
        :return: Predicted class (0-9)
        """
        if self.model:
            return self.model.predict(image)
        else:
            print("❌ The model is not initialized.")
            return None


if __name__ == "__main__":
    model_type = input("🔹 Enter model type ('cnn', 'nn', 'rf'): ").strip().lower()

    try:
        classifier = MnistClassifier(model_type)
        classifier.train()
    except ValueError as e:
        print(e)