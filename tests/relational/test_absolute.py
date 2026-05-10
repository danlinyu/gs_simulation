"""Tests for Grey Absolute Relational Degree (Liu §5.4)."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.relational.absolute import (
    absolute_relational_degree,
    trapezoidal_signed_area,
    zero_starting_point_image,
)


# ---------------------------------------------------------------------------
# Property-based correctness (Liu §5.4 Theorem 5.4.2)
# ---------------------------------------------------------------------------


def test_identity_yields_unity() -> None:
    """ε_{ii} = 1 (Theorem 5.4.2 reflexivity)."""
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert absolute_relational_degree(x, x) == 1.0


def test_translation_invariance() -> None:
    """ε(X, Y) = ε(X + c, Y + d) for any constants c, d (Theorem 5.4.2)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8)
    y = rng.standard_normal(8)
    eps_orig = absolute_relational_degree(x, y)
    eps_shifted = absolute_relational_degree(x + 100.0, y - 50.0)
    np.testing.assert_allclose(eps_shifted, eps_orig, atol=1e-12)


def test_symmetry() -> None:
    """ε_{ij} = ε_{ji}."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(6)
    y = rng.standard_normal(6)
    eps_xy = absolute_relational_degree(x, y)
    eps_yx = absolute_relational_degree(y, x)
    np.testing.assert_allclose(eps_xy, eps_yx, atol=1e-12)


def test_in_unit_interval() -> None:
    """0 < ε ≤ 1 (Theorem 5.4.2 normality)."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        x = rng.standard_normal(7)
        y = rng.standard_normal(7)
        eps = absolute_relational_degree(x, y)
        assert 0.0 < eps <= 1.0


def test_parallel_sequences_yield_unity() -> None:
    """Parallel curves (X − X(1) = Y − Y(1) at every point) yield ε = 1."""
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = np.array([10.0, 11.0, 13.0, 17.0])  # y - y(1) = (0, 1, 3, 7) = x - x(1)
    np.testing.assert_allclose(
        absolute_relational_degree(x, y), 1.0, atol=1e-12,
    )


def test_dissimilar_shapes_have_lower_degree() -> None:
    """An increasing-then-decreasing curve has lower ε with a monotonic one."""
    monotonic = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    triangle  = np.array([1.0, 3.0, 5.0, 3.0, 1.0])  # peak in the middle
    eps_id = absolute_relational_degree(monotonic, monotonic)
    eps_tri = absolute_relational_degree(monotonic, triangle)
    assert eps_tri < eps_id


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_zero_starting_point_image_starts_at_zero() -> None:
    x = np.array([10.0, 12.0, 14.0, 16.0])
    z = zero_starting_point_image(x)
    assert z[0] == 0.0
    np.testing.assert_array_equal(z, np.array([0.0, 2.0, 4.0, 6.0]))


def test_zero_starting_point_image_preserves_2d_axis() -> None:
    X = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    Z = zero_starting_point_image(X, axis=-1)
    np.testing.assert_array_equal(Z[:, 0], 0.0)
    np.testing.assert_array_equal(Z[0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(Z[1], np.array([0.0, 10.0, 20.0]))


def test_trapezoidal_area_zero_for_zero_sequence() -> None:
    z = np.zeros(5)
    assert trapezoidal_signed_area(z) == 0.0


def test_trapezoidal_area_matches_analytic() -> None:
    """For x = (0, 1, 2, 3), the trapezoidal sum is (0+1)/2 + (1+2)/2 + (2+3)/2 = 4.5."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(trapezoidal_signed_area(x), 4.5, atol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        absolute_relational_degree(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_2d_input_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        absolute_relational_degree(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        )


def test_constant_sequence_yields_unity() -> None:
    """Constant sequence has zero zero-starting-point image; both s_i = s_j = 0 → ε = 1."""
    x = np.array([5.0, 5.0, 5.0, 5.0])
    y = np.array([7.0, 7.0, 7.0, 7.0])
    assert absolute_relational_degree(x, y) == 1.0
