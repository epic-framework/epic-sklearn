import warnings
import numpy as np
import pandas as pd

from copy import deepcopy
from typing import TypeVar
from numpy.typing import NDArray
from collections.abc import Collection
from abc import ABCMeta, abstractmethod

try:
    from sklearn.utils import get_tags
except ImportError:
    get_tags = None
    from sklearn.utils._tags import _safe_tags
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin as _MetaEstimatorMixin, clone

from .typing import Classifier
from .utils import check_array_permissive

BC = TypeVar('BC', bound='BaseClassifier')
RC = TypeVar('RC', bound='RandomClassifier')


class FitPredictMixin:
    """Mixin class to add a naive implementation of `fit_predict`."""
    def fit_predict(self: Classifier, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).predict(X)


class BaseClassifier(ClassifierMixin, FitPredictMixin, BaseEstimator, metaclass=ABCMeta):
    """
    An alternative to using `BaseEstimator` for classifiers.

    Any subclass must define two methods:
    - _predict_proba: Predicts raw probabilities from features.
    - classes_: A property which returns an array of classes after training.
                The order of the classes should match the second dimension of
                the probabilities returned by `_predict_proba`.

    From these two methods everything else is calculated automatically.
    """
    def fit(self: BC, X, y=None) -> BC:
        return self

    def _predict(self, X) -> NDArray:
        return self.classes_[self._predict_proba(X).argmax(axis=1)]

    def predict(self, X) -> NDArray | pd.Series:
        pred = self._predict(X)
        if isinstance(X, pd.DataFrame):
            return pd.Series(pred, index=X.index)
        return pred

    @abstractmethod
    def _predict_proba(self, X) -> NDArray:
        pass

    def predict_proba(self, X) -> NDArray | pd.DataFrame:
        probs = self._predict_proba(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(probs, index=X.index, columns=self.classes_)
        return probs

    def predict_log_proba(self, X) -> NDArray | pd.DataFrame:
        return np.log(self.predict_proba(X))

    @property
    @abstractmethod
    def classes_(self) -> NDArray:
        pass


class ConstantClassifier(BaseClassifier):
    """
    A classifier which always predicts a constant.

    Parameters
    ----------
    constant : object, default 0
        Constant to predict.

    .. deprecated:: 1.1.2
        Use `sklearn.dummy.DummyClassifier` instead.
    """
    def __init__(self, constant=0):
        warnings.warn(
            "`ConstantClassifier` is deprecated and will be removed in a future version. "
            "Use `sklearn.dummy.DummyClassifier` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.constant = constant

    def _predict_proba(self, X):
        return np.ones((check_array_permissive(X).shape[0], 1), dtype=float)

    @property
    def classes_(self):
        return np.array([self.constant])

    def _more_tags(self):
        return {"stateless": True}


class RandomClassifier(BaseClassifier):
    """
    A classifier which predicts one of its classes at random.

    Parameters
    ----------
    classes : collection
        The possible classes for the estimator.

    random_state : int or RandomState, optional
        The seed of the pseudo random number generator.

    .. deprecated:: 1.1.2
        Use `sklearn.dummy.DummyClassifier` instead.
    """
    def __init__(self, classes: Collection = (0, 1), random_state: int | np.random.RandomState | None = None):
        warnings.warn(
            "`RandomClassifier` is deprecated and will be removed in a future version. "
            "Use `sklearn.dummy.DummyClassifier` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.classes = classes
        self.random_state = random_state

    def fit(self: RC, X, y=None) -> RC:
        self.random_ = check_random_state(self.random_state)
        return self

    def _predict_proba(self, X):
        probs = self.random_.random((check_array_permissive(X).shape[0], len(self.classes) - 1))
        return np.hstack((probs, 1 - probs.sum(axis=1).reshape((-1, 1))))

    @property
    def classes_(self):
        return np.asarray(self.classes)

    def _more_tags(self):
        return {"non_deterministic": True}


class AlwaysPandasEstimatorMixin:
    """
    A mixin for estimators which return pandas objects even when the input
    is a non-pandas object. On older versions of scikir-learn, some tests
    are expected to fail in those cases.
    """

    def _xfail_checks(self, old_version=False) -> dict[str, str]:
        xfail_checks = dict()
        if old_version:
            xfail_checks |= dict(
                check_methods_sample_order_invariance=(
                    "This test uses numeric indexing, not compatible with pandas objects."
                ),
                check_fit_idempotent=(
                    "This test tries to access `.dtype`, not expecting a pandas object."
                ),
            )
        return xfail_checks

    def _more_tags(self):
        return dict(
            _xfail_checks=self._xfail_checks(old_version=True),
        )

    def __sklearn_tags__(self):
        # Must be defined, since we define `_more_tags`
        return super().__sklearn_tags__()


class MetaEstimatorMixin(_MetaEstimatorMixin):
    """
    Mixin class for meta estimators.
    """
    @property
    def _estimator(self):
        for x in ('estimator', 'classifier'):
            if hasattr(self, x):
                return getattr(self, x)
        raise AttributeError("Cannot find underlying estimator")
    
    def _clone_estimator(self):
        return clone(self._estimator)

    def _more_tags(self):
        tags = _safe_tags(self._estimator)
        if hasattr(self, '_xfail_checks'):
            xfail_checks = self._xfail_checks(old_version=True)
            if isinstance((xfc := tags.get('_xfail_checks')), dict):
                xfc.update(xfail_checks)
            else:
                tags['_xfail_checks'] = xfail_checks
        return tags

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        estimator_tags = get_tags(self._estimator)
        tags.estimator_type = estimator_tags.estimator_type
        for t in ('input', 'target', 'classifier', 'regressor', 'transformer'):
            tt = f"{t}_tags"
            setattr(tags, tt, deepcopy(getattr(estimator_tags, tt)))
        return tags

    def _validation_kwargs(self):
        kwargs = dict()
        try:
            tags = self.__sklearn_tags__()
        except Exception:
            try:
                tags = _safe_tags(self)
            except Exception:
                pass
            else:
                kwargs['accept_sparse'] = tags.get('sparse', False)
        else:
            kwargs['accept_sparse'] = tags.input_tags.sparse
        return kwargs
