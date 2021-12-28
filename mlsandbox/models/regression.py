import numpy as np
from typing import List

from mlsandbox.models.loss import MSELoss, CrossEntropyLoss


class Regression:
    def __init__(
        self,
        iterations: int,
        learning_rate: float,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n: int) -> np.ndarray:
        """Initialize model weights as zeros
        TODO: allow different types of initializations

        Args:
            n (int): number of weights to initialize (ie. the number of features in model)
        """
        weights = np.zeros(n)
        return weights

    def fit(self) -> None:
        """Placeholder method - should be implemented in subclass"""
        raise NotImplementedError("Subclasses of Regression must implement fit method")

    def predict(self) -> None:
        """Placeholder method - should be implemented in subclass"""
        raise NotImplementedError("Subclasses of Regression must implement fit method")


class LinearRegression(Regression):
    def __init__(self, iterations: int, learning_rate: float) -> None:
        super().__init__(iterations, learning_rate)

    def fit(self):
        pass

    def predict(self):
        pass


class LogisticRegression(Regression):
    def __init__(self, iterations: int, learning_rate: float) -> None:
        super().__init__(iterations, learning_rate)

    def fit(self):
        pass

    def predict(self):
        pass
