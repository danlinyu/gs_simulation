"""Group of GM(1,1) models — Liu §7.5.

Four data-window strategies, each fitting a basic GM(1,1) on a different
subset of the input sequence to capture regime shifts and recency effects:

============      ====================================================
Variant           Training data
============      ====================================================
**all_data**      Full sequence ``x⁽⁰⁾(1) … x⁽⁰⁾(n)``
**partial_data**  Tail starting at ``k₀``: ``x⁽⁰⁾(k₀) … x⁽⁰⁾(n)``
**new_info**      Append a new observation: ``x⁽⁰⁾(1) … x⁽⁰⁾(n+1)``
**metabolic**     Roll the window: drop ``x⁽⁰⁾(1)``, append ``x⁽⁰⁾(n+1)``
============      ====================================================

Per Liu §7.5 Example 7.5.1, the metabolic variant typically produces lower
prediction error on the most recent values than the all-data fit because
newer data better reflects current system dynamics.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§7.5 Definitions 7.5.1-7.5.3; Example 7.5.1.
"""
from __future__ import annotations

import numpy as np

from gs_simulation.gm.base import GMFit
from gs_simulation.gm.gm11 import fit_egm11

__all__ = [
    "fit_all_data_gm11",
    "fit_partial_data_gm11",
    "fit_new_information_gm11",
    "fit_metabolic_gm11",
]


def fit_all_data_gm11(x: np.ndarray) -> GMFit:
    """All-data GM(1,1) — fit EGM on the full sequence.

    Equivalent to :func:`fit_egm11` directly; retained as part of the group
    API for symmetry.

    Parameters
    ----------
    x : array-like
        1-D training sequence.

    Returns
    -------
    GMFit
    """
    return fit_egm11(np.asarray(x, dtype=float))


def fit_partial_data_gm11(x: np.ndarray, k0: int) -> GMFit:
    """Partial-data GM(1,1) — fit EGM on the tail ``x[k0-1:]`` (1-based ``k0``).

    Use when an early-period regime is no longer representative of current
    dynamics. ``k0`` must be small enough that the tail has ≥ 4 elements.

    Parameters
    ----------
    x : array-like
        1-D training sequence.
    k0 : int
        1-based start index of the tail to fit. ``k0=1`` is equivalent to
        :func:`fit_all_data_gm11`.

    Returns
    -------
    GMFit

    Raises
    ------
    ValueError
        If ``k0 < 1`` or the tail has fewer than 4 elements.
    """
    arr = np.asarray(x, dtype=float)
    if k0 < 1:
        raise ValueError(f"k0 must be >= 1 (1-based); got k0={k0}")
    tail = arr[k0 - 1:]
    if tail.size < 4:
        raise ValueError(
            f"Partial-data tail has {tail.size} elements; need ≥ 4."
        )
    return fit_egm11(tail)


def fit_new_information_gm11(
    x: np.ndarray, x_new: float | np.ndarray,
) -> GMFit:
    """New-information GM(1,1) — append new observations and refit on the union.

    Used when fresh data arrives and the original fit should be updated
    without dropping any history.

    Parameters
    ----------
    x : array-like
        Original training sequence.
    x_new : float | array-like
        One or more new observations to append.

    Returns
    -------
    GMFit
    """
    arr = np.asarray(x, dtype=float)
    new_arr = np.atleast_1d(np.asarray(x_new, dtype=float))
    extended = np.concatenate([arr, new_arr])
    return fit_egm11(extended)


def fit_metabolic_gm11(
    x: np.ndarray, x_new: float | np.ndarray,
) -> GMFit:
    """Metabolic GM(1,1) — roll the window: drop oldest, append newest.

    Liu §7.5 Example 7.5.1 reports the metabolic fit outperforms all-data
    on recent values because newer observations better reflect current
    dynamics. The window length is preserved.

    Parameters
    ----------
    x : array-like
        Original training sequence (length ``n``).
    x_new : float | array-like
        New observations to append. Same number of oldest observations are
        dropped to keep the window length at ``n``.

    Returns
    -------
    GMFit

    Raises
    ------
    ValueError
        If the resulting window has fewer than 4 elements.
    """
    arr = np.asarray(x, dtype=float)
    new_arr = np.atleast_1d(np.asarray(x_new, dtype=float))
    n_drop = new_arr.size
    if arr.size - n_drop < 4:
        raise ValueError(
            f"Metabolic window after dropping {n_drop} oldest leaves "
            f"{arr.size - n_drop} elements; need ≥ 4."
        )
    rolled = np.concatenate([arr[n_drop:], new_arr])
    return fit_egm11(rolled)
