"""Tests for cross-sequence correction (Liu §5.8)."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.relational.absolute import absolute_relational_degree
from gs_simulation.relational.cross_sequences import (
    absolute_relational_degree_corrected,
    degree_of_difference,
    trapezoidal_absolute_area,
)


def test_trapezoidal_absolute_zero_for_zero_sequence() -> None:
    z = np.zeros(5)
    assert trapezoidal_absolute_area(z) == 0.0


def test_trapezoidal_absolute_no_signed_cancellation() -> None:
    """For x = (-2, 0, 2): trapezoidal of |x| = (|-2|+|0|)/2 + (|0|+|2|)/2 = 2."""
    x = np.array([-2.0, 0.0, 2.0])
    np.testing.assert_allclose(trapezoidal_absolute_area(x), 2.0, atol=1e-12)


def test_degree_of_difference_zero_for_identical() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert degree_of_difference(x, x) == 0.0


def test_degree_of_difference_in_unit_range() -> None:
    rng = np.random.default_rng(1)
    for _ in range(15):
        x = rng.standard_normal(6)
        y = rng.standard_normal(6)
        delta = degree_of_difference(x, y)
        assert 0.0 <= delta < 1.0


def test_corrected_degree_lower_than_uncorrected() -> None:
    """For crossing curves, the EC correction reduces ε."""
    # Two oscillating curves whose signed areas cancel but whose absolute
    # divergences are large (intersect at multiple interior points).
    X = np.array([1.0, 1.2, 0.8, 1.2, 0.8, 1.0])
    Y = np.array([1.5, 1.3, 1.7, 1.3, 1.7, 1.5])
    eps = absolute_relational_degree(X, Y)
    eps_ec = absolute_relational_degree_corrected(X, Y)
    assert eps_ec < eps


def test_corrected_equal_to_uncorrected_for_identical() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    np.testing.assert_allclose(
        absolute_relational_degree_corrected(x, x),
        absolute_relational_degree(x, x),
        atol=1e-12,
    )


def test_cross_sequence_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        degree_of_difference(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
