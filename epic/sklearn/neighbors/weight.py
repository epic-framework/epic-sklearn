import numpy as np

from typing import Literal
from numpy.typing import ArrayLike, NDArray


class ThresholdedWeight:
    """
    A collection of distance weighing schemes, allowing for a threshold to be applied.
    Wherever the distance is above the threshold, the weight will be zero.

    Can be used in KNN estimators which take a `weights` argument that can be a callable.

    Parameters
    ----------
    scheme : {'uniform', 'reciprocal'}, default 'uniform'
        The weighing scheme to use.
        - "uniform": All weights are set to 1.
        - "reciprocal": Weights are set to 1 / distance.

    threshold : float, optional
        The distance above which the weights are set to zero.
        If not provided, no threshold is applied.
    """
    def __init__(
            self,
            scheme: Literal['uniform', 'reciprocal'] = 'uniform',
            threshold: float | None = None,
    ):
        if scheme not in ('uniform', 'reciprocal'):
            raise ValueError(f"Invalid value for `scheme`: {scheme!r}.")
        self.scheme = scheme
        self.threshold = threshold

    def __call__(self, dist: ArrayLike) -> NDArray:
        weight = getattr(self, f'{self.scheme}_weight')(dist)
        if self.threshold is not None:
            weight[dist > self.threshold] = 0.
        return weight

    @staticmethod
    def uniform_weight(dist: ArrayLike) -> NDArray:
        return np.ones_like(dist)

    @staticmethod
    def reciprocal_weight(dist: ArrayLike) -> NDArray:
        with np.errstate(divide='ignore'):
            weight = 1. / np.asarray(dist)
        inf_mask = np.isinf(weight)
        inf_row = np.any(inf_mask, axis=1)
        weight[inf_row] = inf_mask[inf_row]
        return weight
