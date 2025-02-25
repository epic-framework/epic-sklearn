import numpy as np

from collections.abc import Iterable
from numpy.typing import ArrayLike, NDArray


def rebalance_precision(precision: float, pnr_in: float, pnr_out: float = 1) -> float:
    """
    Translate precision between datasets which have different positives-to-negatives ratios.

    Parameters
    ----------
    precision : float
        Precision on input dataset.

    pnr_in : float
        Positive-to-Negative ratio on input dataset.

    pnr_out : float
        Positive-to-Negative ratio on output dataset.

    Returns
    -------
    float
        Precision on output dataset.
    """
    return 1 / ((1 / precision - 1) * pnr_in / pnr_out + 1)


def boolean_selector(arg: bool | ArrayLike | Iterable, /, length: int) -> NDArray[np.bool_]:
    """
    Create a boolean mask for selecting indices from a 1-D array.

    Parameters
    ----------
    arg : bool, array-like or iterable
        - bool:
            Mask will be filled with this value.
        - array-like or iterable of booleans:
            Must be 1-D, of length `length`.
            Just converted into an array and returned.
        - array-like or iterable of integers:
            Must be 1-D.
            Indices of the output to be filled with True; rest are False.

    length : int
        Length of returned array.

    Returns
    -------
    ndarray, shape (length,), dtype bool

    Examples
    --------
    >>> boolean_selector(True, 3)
    array([True, True, True])

    >>> boolean_selector([True, False, True], 3)
    array([True, False, True])

    >>> boolean_selector([1, 3, 4], 6)
    array([False, True, False, True, True, False])
    """
    if isinstance(arg, bool):
        return np.full(length, arg)
    if isinstance(arg, Iterable) and not isinstance(arg, np.ndarray):
        arg = list(arg)
    arg = np.asarray(arg)
    if arg.ndim != 1:
        raise ValueError(f"Expected a 1-D array; got {arg.ndim}-D instead.")
    if np.issubdtype(arg.dtype, np.integer):
        mask = np.zeros(length, dtype=np.bool_)
        mask[arg] = True
        return mask
    if arg.size != length:
        raise ValueError(f"Length of argument should be {length}; got {len(arg)} instead.")
    if not np.issubdtype(arg.dtype, np.bool_):
        raise ValueError(f"Expected a boolean dtype; got {arg.dtype} instead.")
    return arg
