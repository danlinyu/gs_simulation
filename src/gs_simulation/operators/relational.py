"""Grey relational operators D₁ / D₂ / D₃ (Liu §5.2, Deng 1985).

These operators normalize a behavioural sequence into a non-dimensional form
suitable for grey relational analysis. Each operator emphasizes a different
aspect of the data:

- ``D₁`` *initialing*: divide by the first value; preserves growth shape
  relative to the starting position.
- ``D₂`` *averaging*: divide by the mean; preserves deviation from the
  average.
- ``D₃`` *interval*: min-max rescaling to ``[0, 1]``; preserves rank-and-range
  pattern.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis: Methods, Models and Applications*,
2nd Ed., Springer Singapore. §5.2 Definitions 5.2.1-5.2.3.
Deng, Julong (1985). Original three-operator formulation.
"""
from __future__ import annotations

import numpy as np


def apply_d1_initialing(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply the D₁ initialing operator along the given axis.

    For a sequence ``X = (x(1), x(2), …, x(n))`` with ``x(1) ≠ 0``::

        x(k)^{D₁} = x(k) / x(1)

    Parameters
    ----------
    x : array-like
        Input sequence(s); any shape with at least one element along ``axis``.
    axis : int, default -1
        Axis along which to apply the operator.

    Returns
    -------
    numpy.ndarray
        D₁-normalized sequence(s); the first element along ``axis`` is 1.

    Raises
    ------
    ValueError
        If any first value along ``axis`` is zero (operator undefined).

    References
    ----------
    Liu (2024) §5.2 Definition 5.2.1.

    Examples
    --------
    >>> apply_d1_initialing(np.array([3.2, 3.7, 4.5, 4.9, 5.6]))
    array([1.     , 1.15625, 1.40625, 1.53125, 1.75   ])
    """
    x_arr = np.asarray(x, dtype=float)
    first = np.take(x_arr, indices=0, axis=axis)
    if np.any(first == 0):
        raise ValueError(
            "D1 initialing operator undefined: first value along axis is zero."
        )
    return x_arr / np.expand_dims(first, axis=axis)


def apply_d2_averaging(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply the D₂ averaging operator along the given axis.

    For a sequence with mean ``X̄ ≠ 0``::

        x(k)^{D₂} = x(k) / X̄

    Parameters
    ----------
    x : array-like
        Input sequence(s); any shape with at least one element along ``axis``.
    axis : int, default -1
        Axis along which to apply the operator.

    Returns
    -------
    numpy.ndarray
        D₂-normalized sequence(s); the resulting mean along ``axis`` is 1.

    Raises
    ------
    ValueError
        If the sequence mean along ``axis`` is zero (operator undefined).

    References
    ----------
    Liu (2024) §5.2 Definition 5.2.2.

    Examples
    --------
    >>> X = np.array([3.2, 3.7, 4.5, 4.9, 5.6])
    >>> result = apply_d2_averaging(X)
    >>> bool(np.isclose(result.mean(), 1.0))
    True
    """
    x_arr = np.asarray(x, dtype=float)
    mean = np.mean(x_arr, axis=axis)
    if np.any(mean == 0):
        raise ValueError(
            "D2 averaging operator undefined: sequence mean along axis is zero."
        )
    return x_arr / np.expand_dims(mean, axis=axis)


def apply_d3_interval(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply the D₃ interval (min-max) operator along the given axis.

    For a sequence with ``max ≠ min``::

        x(k)^{D₃} = (x(k) − min) / (max − min)

    The result is bounded in ``[0, 1]`` with both endpoints attained.

    Parameters
    ----------
    x : array-like
        Input sequence(s); any shape with at least one element along ``axis``.
    axis : int, default -1
        Axis along which to apply the operator.

    Returns
    -------
    numpy.ndarray
        D₃-normalized sequence(s); each element along ``axis`` is in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the sequence is constant along ``axis`` (``max == min``).

    References
    ----------
    Liu (2024) §5.2 Definition 5.2.3.

    Examples
    --------
    >>> X = np.array([3.2, 3.7, 4.5, 4.9, 5.6])
    >>> apply_d3_interval(X)  # doctest: +NORMALIZE_WHITESPACE
    array([0.        , 0.20833333, 0.54166667, 0.70833333, 1.        ])
    """
    x_arr = np.asarray(x, dtype=float)
    x_max = np.max(x_arr, axis=axis)
    x_min = np.min(x_arr, axis=axis)
    span = x_max - x_min
    if np.any(span == 0):
        raise ValueError(
            "D3 interval operator undefined: sequence is constant along axis."
        )
    return (x_arr - np.expand_dims(x_min, axis=axis)) / np.expand_dims(
        span, axis=axis
    )


__all__ = [
    "apply_d1_initialing",
    "apply_d2_averaging",
    "apply_d3_interval",
]
