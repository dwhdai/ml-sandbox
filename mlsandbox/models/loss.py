from typing import Any


class _Loss:
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class MSELoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args: Any, **kwds: Any) -> float:
        """Calculate and return loss

        Returns:
            float: The mean squared error loss
        """
        loss = float()
        return loss

    def gradient(self, *args: Any) -> float:
        """Calculate and return the gradient of the loss w.r.t to the model parameters

        Returns:
            float: The gradient of the loss w.r.t to the model parameters
        """

    def hessian(self, *args: Any) -> float:
        """Calculate and return the Hessian of the loss w.r.t to the model parameters

        Returns:
            float: The Hessian of the loss w.r.t to the model parameters
        """


class CrossEntropyLoss(_Loss):
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> float:
        """Calculate and return loss

        Returns:
            float: The mean squared error loss
        """
        loss = float()
        return loss

    def gradient(self, *args: Any) -> float:
        """Calculate and return the gradient of the loss w.r.t to the model parameters

        Returns:
            float: The gradient of the loss w.r.t to the model parameters
        """

        return float()

    def hessian(self, *args: Any) -> float:
        """Calculate and return the Hessian of the loss w.r.t to the model parameters

        Returns:
            float: The Hessian of the loss w.r.t to the model parameters
        """

        return float()
