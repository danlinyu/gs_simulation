"""Base infrastructure for the GM(1,1) family — Liu §7.

This module defines the :class:`GMFit` immutable dataclass that captures a
fitted GM(1,1) model and its metadata. The four basic forms (EGM, ODGM,
DGM, EDGM per Liu §7.2) all populate the same dataclass; downstream
simulate / forecast routines dispatch on the ``form`` field.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore. §7.1.
"""
from __future__ import annotations

from dataclasses import dataclass

# Forbidden-zone threshold for the development coefficient |a| (Liu §7.3
# Proposition 7.3.2). Above this, GM(1,1) predictions oscillate or diverge.
GM11_FORBIDDEN_ABS_A: float = 2.0

# Suitable-range advisory thresholds for -a (Liu §7.3 Table 7.9).
GM11_SUITABLE_RANGES: dict[str, tuple[float, float]] = {
    "long_term":  (0.0, 0.3),   # > 98 % accuracy at step 1
    "mid_term":   (0.3, 0.5),   # > 90 % accuracy at steps 1-2
    "short_term": (0.5, 0.8),   # short-term only; remnant model recommended
    "remnant":    (0.8, 1.0),   # remnant model required
    "unsuitable": (1.0, 2.0),   # GM(1,1) not recommended
}


@dataclass(frozen=True)
class GMFit:
    """Immutable container for a fitted GM(1,1) model.

    Attributes
    ----------
    a : float
        Development coefficient.
    b : float
        Grey input. The pair ``(a, b)`` parameterises the whitenization ODE
        ``dx⁽¹⁾/dt + a·x⁽¹⁾ = b`` (Liu §7.2 Eq 7.4 for EGM).
    x0_first : float
        ``x⁽⁰⁾(1)`` — initial value of the training sequence; used in the
        time-response formula.
    n_train : int
        Number of training observations the model was fit on.
    form : str
        Which basic form was fit: ``"EGM"`` (even form, mean-background),
        ``"ODGM"`` (original-difference), ``"DGM"`` (discrete), or
        ``"EDGM"`` (even-difference).

    References
    ----------
    Liu (2024) §7.2 Definitions 7.2.1, 7.2.3-7.2.4, 7.2.7-7.2.8.
    """

    a: float
    b: float
    x0_first: float
    n_train: int
    form: str

    def suitable_range(self) -> str:
        """Return the regime label describing this fit's development coefficient.

        Liu §7.3 Table 7.9 maps ``-a`` magnitude to recommended use.
        """
        neg_a = -self.a
        for label, (lo, hi) in GM11_SUITABLE_RANGES.items():
            if lo <= neg_a < hi:
                return label
        return "forbidden"  # |a| >= 2; in practice fit should have raised
