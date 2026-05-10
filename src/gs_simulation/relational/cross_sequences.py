"""Cross-Sequence Correction for GRA — Liu §5.8 (Liu et al. 2024).

When two zero-shifted sequences *cross* (intersect at one or more interior
points), their signed-area difference can cancel positive and negative
regions, falsely suggesting high similarity. The cross-sequence corrected
degree subtracts a *spatial divergence* term ``Δ_{ij}`` that uses
*absolute-value* differences before integration::

    Δ_{ij} = ||s_i − s_j|| / (||s_j|| + 1 + ||s_i|| + ||s_i − s_j||)
    ε_{ij}^EC = ε_{ij} − Δ_{ij}

where::

    ||s_i|| = ∫ |X_i⁰| dt   (trapezoidal absolute integral)

Use case
--------
Substrate-validity diagnostic for oscillating systems (Sterman bathtub-
mode dynamics, Forrester multi-loop systems). Without the EC correction,
two anti-phase oscillators may be flagged as "highly similar" by
absolute or similitude degree alone.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§5.8 Definitions 5.8.1-5.8.3.
Liu et al. (2024). Original cross-sequence correction.
"""
from __future__ import annotations

import numpy as np

from gs_simulation.relational.absolute import (
    absolute_relational_degree,
    zero_starting_point_image,
)

__all__ = [
    "trapezoidal_absolute_area",
    "degree_of_difference",
    "absolute_relational_degree_corrected",
]


def trapezoidal_absolute_area(x: np.ndarray) -> float:
    """Trapezoidal integral of ``|x|`` (absolute area, no sign cancellation).

    Liu §5.8 ``||s||`` notation. Used in the cross-sequence correction
    so that intersecting curves don't spuriously cancel.

    Parameters
    ----------
    x : array-like, 1-D

    Returns
    -------
    float
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D sequence; got shape {arr.shape}")
    abs_arr = np.abs(arr)
    return float(np.sum((abs_arr[:-1] + abs_arr[1:]) / 2.0))


def degree_of_difference(
    x_i: np.ndarray, x_j: np.ndarray,
) -> float:
    """Compute the spatial divergence ``Δ_{ij}`` between zero-shifted sequences.

    Larger Δ ⟺ greater spatial separation despite possibly equal signed
    areas (i.e., the sequences cross).

    Parameters
    ----------
    x_i, x_j : array-like, 1-D, equal length

    Returns
    -------
    float
        ``Δ_{ij} ∈ [0, 1)``.

    References
    ----------
    Liu (2024) §5.8 Definition 5.8.3.
    """
    x_i_arr = np.asarray(x_i, dtype=float)
    x_j_arr = np.asarray(x_j, dtype=float)
    if x_i_arr.size != x_j_arr.size:
        raise ValueError(
            f"Sequences must be the same length; got {x_i_arr.size} and {x_j_arr.size}"
        )
    z_i = zero_starting_point_image(x_i_arr)
    z_j = zero_starting_point_image(x_j_arr)
    norm_i = trapezoidal_absolute_area(z_i)
    norm_j = trapezoidal_absolute_area(z_j)
    norm_diff = trapezoidal_absolute_area(z_i - z_j)
    denom = norm_j + 1.0 + norm_i + norm_diff
    return float(norm_diff / denom)


def absolute_relational_degree_corrected(
    x_i: np.ndarray, x_j: np.ndarray,
) -> float:
    """Cross-sequence-corrected absolute relational degree ``ε_{ij}^EC``.

    Equals the standard absolute degree minus the spatial-divergence
    penalty ``Δ_{ij}``. Use when curves are known or suspected to cross.

    Parameters
    ----------
    x_i, x_j : array-like, 1-D, equal length

    Returns
    -------
    float
        ``ε_{ij}^EC = ε_{ij} − Δ_{ij}``.

    References
    ----------
    Liu (2024) §5.8.
    """
    eps = absolute_relational_degree(x_i, x_j)
    delta = degree_of_difference(x_i, x_j)
    return float(eps - delta)
