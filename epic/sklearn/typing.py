from typing import Protocol, TypeVar
from numpy._typing import NDArray

T = TypeVar('T')


class Estimator(Protocol):
    """
    A protocol for an Estimator.
    For this purpose, whatever defines a `fit` method is considered an estimator.
    """
    def fit(self: T, X, y=None, **fit_params) -> T: ...


class Classifier(Estimator):
    """A protocol for a Classifier."""
    classes_: NDArray

    def predict_proba(self, X): ...
    def predict(self, X): ...


class LinearClassifier(Classifier):
    """A protocol for a Linear Classifier."""
    coef_: NDArray
    intercept_: NDArray
