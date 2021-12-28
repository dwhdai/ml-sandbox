import numpy as np
from typing import List

from mlsandbox.models.loss import MSELoss, CrossEntropyLoss


class Regression:
    def __init__(
        self,
        iterations: int,
        learning_rate: float,
        loss,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.loss = loss

    def initialize_weights(self, n: int) -> np.ndarray:
        """Initialize model weights as zeros
        TODO: allow different types of initializations

        Args:
            n (int): number of weights to initialize (ie. the number of features in model)
        """
        weights = np.zeros(n)
        return weights

    def fit(self, x, y) -> float:
        """Fits the model
        TODO: Implement regularization

        Returns:
            float: The model log-likelihood
        """
        x = np.insert(x, 0, 1, axis=1)
        self.weights = self.initialize_weights(x.shape[1])
        for _ in range(self.iterations):
            loss = self.loss(x, y, self.weights)
            grad = self.loss.gradient(x, y, self.weights)
            self.weights -= self.learning_rate * grad
        return loss

    def predict(self, x) -> List[float]:
        """Get predictions from model

        Returns:
            float: Returns the predictions from the model, matching the dimensions of
                the input data.
        """

        y_pred = np.matmul(np.transpose(self.weights), x)

        return y_pred


class LinearRegression(Regression):
    def __init__(self, iterations: int, learning_rate: float) -> None:
        super().__init__(iterations, learning_rate, MSELoss)


class LogisticRegression(Regression):
    def __init__(self, iterations: int, learning_rate: float) -> None:
        super().__init__(iterations, learning_rate, CrossEntropyLoss)
