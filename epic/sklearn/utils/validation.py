import pandas as pd

from types import EllipsisType
from typing import Literal, overload

from scipy import sparse as sp
from pandas.core.generic import NDFrame
from numpy._typing import ArrayLike, DTypeLike, NDArray

from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_X_y
try:
    from sklearn.utils.validation import validate_data
except ImportError:
    validate_data = None

from ..typing import Estimator


def assert_categorical(array: ArrayLike, name: str = "array") -> None:
    """
    Verifies than an array contains discrete values.

    Parameters
    ----------
    array : array-like
        Input array.

    name : str, default "array"
        Displayed in error message.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `array` is not categorical.
    """
    if type_of_target(array) not in ('binary', 'multiclass') and array.dtype != 'category':
        raise ValueError(f"{name} must be categorical.")


@overload
def check_pandas(X, y: None = ..., **kwargs) -> tuple[pd.DataFrame, bool]: ...
@overload
def check_pandas(X, y, **kwargs) -> tuple[pd.DataFrame, pd.Series, bool]: ...

def check_pandas(X, y=None, **kwargs):
    """
    Converts features and possibly targets to pandas objects, if needed.

    Parameters
    ----------
    X : object
        Features matrix.

    y : object, optional
        Target vector.

    **kwargs :
        If `X` is neither a DataFrame nor a Series and `kwargs` are provided,
        also validate `X` using `sklearn.utils.check_array` with the `kwargs`
        before converting it to a DataFrame.
        If `X` is a DataFrame or a Series, these arguments are ignored.

    Returns
    -------
    DataFrame
        `X` converted to a DataFrame.
        If `X` is already a DaraFrame, the same object is returned.

    Series
        `y` converted to a Series.
        If `y` is already a Series, the same object is returned.
        Only returned if `y` was provided.

    bool
        A flag indicating whether `X` was already a DataFrame to begin with.
    """
    if isinstance(X, pd.DataFrame):
        given_df = True
    else:
        given_df = False
        if kwargs and not isinstance(X, pd.Series):
            if y is None:
                X = check_array(X, **kwargs)
            else:
                X, y = check_X_y(X, y, **kwargs)
        if sp.issparse(X):
            X = pd.DataFrame.sparse.from_spmatrix(X)
        else:
            X = pd.DataFrame(X)
    if y is None:
        return X, given_df
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    if len(X) != len(y):
        raise ValueError(f"Mismatch in lengths between `X` ({len(X)}) and `y` ({len(y)}).")
    # Would raise if indices aren't matching
    y = y.loc[X.index]
    return X, y, given_df


def check_array_permissive(
        array: ArrayLike | sp.spmatrix | NDFrame,
        accept_sparse: str | bool | list[str] | tuple[str] = ('csr', 'csc', 'coo', 'dok', 'bsr', 'lil', 'dia'),
        dtype: Literal['numeric'] | DTypeLike | list[DTypeLike] | tuple[DTypeLike] | None = None,
        force_all_finite: bool | Literal['allow-nan'] = False,
        **kwargs,
) -> NDArray | sp.spmatrix:
    """
    A thin wrap around `sklearn.utils.check_array` with very permissive defaults.
    By default, allows any king of sparse matrix, non-numeric arrays and arrays containing NaNs.

    Parameters
    ----------
    array : array-like, sparse matrix or pandas object
        Input array.

    accept_sparse : str, bool or list/tuple of str, default ('csr', 'csc', 'coo', 'dok', 'bsr', 'lil', 'dia')
        Sparse formats to allow, if any.

    dtype : 'numeric', type or list/tuple of types, optional
        Dtypes to allow.

    force_all_finite : bool or 'allow-nan', default False
        Whether to raise an error on np.inf, np.nan and pd.NA in `array`.

    **kwargs :
        Sent to `sklearn.utils.check_array` as is.

    Returns
    -------
    ndarray or sparse matrix
    """
    return check_array(array, accept_sparse=accept_sparse, dtype=dtype, force_all_finite=force_all_finite, **kwargs)


def validate_pandas(
        estimator: Estimator,
        /,
        X: ArrayLike | sp.spmatrix | pd.DataFrame,
        y: ArrayLike | pd.Series | EllipsisType = ...,
        *,
        reset: bool,
        ensure_df: bool = False,
        **check_params,
) -> (
        tuple[NDArray | sp.spmatrix | pd.DataFrame, bool] |
        tuple[NDArray | sp.spmatrix | pd.DataFrame, NDArray | pd.Series, bool]
):
    """
    Validate input data, but pass along DataFrame and Series objects.

    Parameters
    ----------
    estimator : Estimator
        The estimator inside which the validation is performed.

    X : array-like, sparse matrix of DataFrame
        The data to validate.

    y : array-like or Series, optional
        The target to validate.

    reset : bool
        Whether to reset estimator attributes (fit mode) or not (transform / predict mode).

    ensure_df : bool, default False
        If true, will convert the input data to a DataFrame if it is not already one.

    **check_params :
        Additional kwargs to be passed along to `check_array`.

    Returns
    -------
    DataFrame, ndarray or sparse matrix
        Validated data.

    bool
        A flag indicating whether `X` was already a DataFrame to begin with.
    """
    if ensure_df:
        if y is ...:
            X, given_df = check_pandas(X, estimator=estimator, **check_params)
        else:
            X, y, given_df = check_pandas(X, y, estimator=estimator, **check_params)
    elif not (given_df := isinstance(X, pd.DataFrame)):
        if y is ...:
            X = check_array(X, estimator=estimator, **check_params)
        else:
            X, y = check_X_y(X, y, estimator=estimator, **check_params)
    if validate_data is not None:
        if y is ...:
            X = validate_data(estimator, X, reset=reset, skip_check_array=True)
        else:
            X, y = validate_data(estimator, X, y, reset=reset, skip_check_array=True)
    elif isinstance(estimator, BaseEstimator):
        estimator._check_feature_names(X, reset=reset)
        estimator._check_n_features(X, reset=reset)
    if y is ...:
        return X, given_df
    return X, y, given_df
