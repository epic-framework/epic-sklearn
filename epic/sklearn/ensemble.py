from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor

from .general import FitPredictMixin


class RandomForestRegressor(FitPredictMixin, _RandomForestRegressor):
    """
    A RandomForestRegressor which also includes a `fit_predict` method.

    See Also
    --------
    sklearn.ensemble.RandomForestRegressor
    """
