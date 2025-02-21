from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    Interface for MNIST classification models.
    Each model should implement `train` and `predict` methods.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the classifier on the given dataset.
        :param X_train: Training images (numpy array).
        :param y_train: Corresponding labels (numpy array).
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Predict the class of input images.
        :param X_test: Test images (numpy array).
        :return: Predicted labels (numpy array).
        """
        pass
