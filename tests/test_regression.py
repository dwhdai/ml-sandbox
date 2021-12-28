from mlsandbox.models import Regression, LinearRegression
import pytest


class TestRegression:
    def test_abstract_methods_fail(self):
        regression = Regression(iterations=1, learning_rate=0.01)
        assert hasattr(regression, "fit")
        assert hasattr(regression, "predict")

        with pytest.raises(NotImplementedError) as execinfo:
            regression.fit()
        assert execinfo.value.args[0] == "Subclasses of Regression must implement fit method"

        with pytest.raises(NotImplementedError) as execinfo:
            regression.predict()
        assert execinfo.value.args[0] == "Subclasses of Regression must implement fit method"


class TestLinearRegression:
    lin_reg = LinearRegression(iterations=1, learning_rate=0.01)

    def test_fit(self):
        pass
