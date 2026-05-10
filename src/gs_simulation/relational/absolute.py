"""Grey Absolute Relational Degree — Liu §5.4.

Captures *shape* similarity between two sequences via the signed area of
their zero-starting-point images. Translation-invariant: shifting either
sequence by a constant leaves ε unchanged. Uncorrelated with the relative
relational degree (Liu §5.5.1, Theorem 5.5.1) — high ε does not imply
high r and vice versa.

Mathematics
-----------
For two equal-length, equal-interval sequences ``X_i, X_j`` of length ``n``:

1. Zero-starting-point transformation::

       X_i⁰(k) = X_i(k) - X_i(1),    k = 1, …, n

2. Signed area of the zigzag curve via trapezoidal rule::

       s_i = Σ_{k=1}^{n-1} (X_i⁰(k) + X_i⁰(k+1)) / 2

3. Absolute relational degree (Liu §5.4 Definition 5.4.3, Liu 1992)::

       ε_{ij} = (1 + |s_i| + |s_j|) / (1 + |s_i| + |s_j| + |s_i - s_j|)

Properties (Liu §5.4 Theorem 5.4.2)
-----------------------------------
- ``0 < ε_{ij} ≤ 1``.
- Translation-invariant: ``ε(X+c, Y+d) = ε(X, Y)``.
- ``ε_{ii} = 1``; symmetric: ``ε_{ij} = ε_{ji}``.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§5.4 Definitions 5.4.1-5.4.3; Theorem 5.4.2; Lemma 5.4.2 (trapezoidal
approximation); Example 5.4.1.
Liu, Sifeng (1992). Original absolute-degree formulation.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "absolute_relational_degree",
    "zero_starting_point_image",
    "trapezoidal_signed_area",
]


def zero_starting_point_image(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply the zero-starting-point transformation: ``X⁰(k) = X(k) − X(1)``.

    Parameters
    ----------
    x : array-like
    axis : int, default -1

    Returns
    -------
    numpy.ndarray
        Same shape as input; first slice along ``axis`` is zero.

    References
    ----------
    Liu (2024) §5.4 Definition 5.4.1.
    """
    arr = np.asarray(x, dtype=float)
    first = np.take(arr, indices=[0], axis=axis)
    return arr - first


def trapezoidal_signed_area(x: np.ndarray) -> float:
    """Signed area of a 1-D zero-shifted zigzag curve via trapezoidal rule.

    Liu §5.4 Lemma 5.4.2: ``s = ∫_1^n X⁰ dt ≈ Σ (X⁰(k) + X⁰(k+1)) / 2``.

    Parameters
    ----------
    x : array-like, 1-D
        Zero-starting-point image (i.e., ``x[0] == 0`` typically).

    Returns
    -------
    float
        Signed area.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D sequence; got shape {arr.shape}")
    return float(np.sum((arr[:-1] + arr[1:]) / 2.0))


def absolute_relational_degree(
    x_i: np.ndarray, x_j: np.ndarray,
) -> float:
    """Compute the absolute grey relational degree ``ε_{ij}``.

    Captures shape similarity. Translation-invariant.

    Parameters
    ----------
    x_i, x_j : array-like, 1-D, equal length
        Two sequences sampled at equal intervals.

    Returns
    -------
    float
        ``ε_{ij} ∈ (0, 1]``. Larger ⟺ more similar geometric shape.

    Raises
    ------
    ValueError
        If either input is non-1-D or the sequences have different lengths.

    References
    ----------
    Liu (2024) §5.4 Definition 5.4.3.

    Examples
    --------
    Identity yields 1:

    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 4.0, 8.0])
    >>> bool(np.isclose(absolute_relational_degree(x, x), 1.0))
    True

    Translation-invariant:

    >>> x = np.array([1.0, 2.0, 4.0, 8.0])
    >>> y = np.array([3.0, 4.0, 6.0, 10.0])  # = x + 2
    >>> bool(np.isclose(absolute_relational_degree(x, y), 1.0))
    True
    """
    x_i_arr = np.asarray(x_i, dtype=float)
    x_j_arr = np.asarray(x_j, dtype=float)
    if x_i_arr.ndim != 1 or x_j_arr.ndim != 1:
        raise ValueError(
            "absolute_relational_degree requires 1-D sequences; got shapes "
            f"{x_i_arr.shape} and {x_j_arr.shape}"
        )
    if x_i_arr.size != x_j_arr.size:
        raise ValueError(
            "Sequences must be the same length; got "
            f"{x_i_arr.size} and {x_j_arr.size}"
        )

    x_i_zero = zero_starting_point_image(x_i_arr)
    x_j_zero = zero_starting_point_image(x_j_arr)

    s_i = trapezoidal_signed_area(x_i_zero)
    s_j = trapezoidal_signed_area(x_j_zero)
    s_ij = trapezoidal_signed_area(x_i_zero - x_j_zero)

    abs_si = abs(s_i)
    abs_sj = abs(s_j)
    abs_diff = abs(s_ij)

    numer = 1.0 + abs_si + abs_sj
    denom = 1.0 + abs_si + abs_sj + abs_diff
    return float(numer / denom)
