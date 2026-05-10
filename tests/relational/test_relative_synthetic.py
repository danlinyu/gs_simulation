"""Tests for relative + synthetic relational degrees (Liu §5.5)."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.relational.absolute import absolute_relational_degree
from gs_simulation.relational.relative import relative_relational_degree
from gs_simulation.relational.synthetic import synthetic_relational_degree


# ---------------------------------------------------------------------------
# Relative degree
# ---------------------------------------------------------------------------


def test_relative_identity_yields_unity() -> None:
    x = np.array([2.0, 4.0, 6.0, 8.0])
    assert relative_relational_degree(x, x) == 1.0


def test_relative_scale_invariance() -> None:
    """Liu §5.5.1 Theorem 5.5.1: r(aX, bY) = r(X, Y) for positive a, b."""
    rng = np.random.default_rng(0)
    x = np.abs(rng.standard_normal(7)) + 1.0
    y = np.abs(rng.standard_normal(7)) + 1.0
    r_orig = relative_relational_degree(x, y)
    r_scaled = relative_relational_degree(3.0 * x, 7.5 * y)
    np.testing.assert_allclose(r_scaled, r_orig, atol=1e-12)


def test_relative_in_unit_interval() -> None:
    rng = np.random.default_rng(42)
    for _ in range(15):
        x = np.abs(rng.standard_normal(8)) + 1.0
        y = np.abs(rng.standard_normal(8)) + 1.0
        r = relative_relational_degree(x, y)
        assert 0.0 < r <= 1.0


def test_relative_proportional_yields_unity() -> None:
    """If Y = c X for c > 0, then r = 1 (proportional growth)."""
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = 2.5 * x
    np.testing.assert_allclose(
        relative_relational_degree(x, y), 1.0, atol=1e-12,
    )


def test_relative_zero_first_element_raises() -> None:
    """D₁ initialing requires non-zero first; relative inherits this."""
    with pytest.raises(ValueError, match="first value"):
        relative_relational_degree(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0]),
        )


# ---------------------------------------------------------------------------
# Synthetic degree
# ---------------------------------------------------------------------------


def test_synthetic_default_theta_is_half() -> None:
    """Default θ = 0.5 averages absolute + relative equally (Liu §5.5.2)."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.5, 3.0, 4.5, 6.0])
    rho = synthetic_relational_degree(x, y)
    explicit = synthetic_relational_degree(x, y, theta=0.5)
    assert rho == explicit


def test_synthetic_theta_one_recovers_absolute() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = np.array([1.5, 2.5, 5.0, 9.0])
    rho_1 = synthetic_relational_degree(x, y, theta=1.0)
    eps = absolute_relational_degree(x, y)
    np.testing.assert_allclose(rho_1, eps, atol=1e-12)


def test_synthetic_theta_zero_recovers_relative() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = np.array([1.5, 2.5, 5.0, 9.0])
    rho_0 = synthetic_relational_degree(x, y, theta=0.0)
    r = relative_relational_degree(x, y)
    np.testing.assert_allclose(rho_0, r, atol=1e-12)


def test_synthetic_theta_out_of_range_raises() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="theta must be"):
        synthetic_relational_degree(x, x, theta=-0.1)
    with pytest.raises(ValueError, match="theta must be"):
        synthetic_relational_degree(x, x, theta=1.5)


def test_synthetic_in_unit_interval() -> None:
    rng = np.random.default_rng(99)
    for _ in range(15):
        x = np.abs(rng.standard_normal(6)) + 1.0
        y = np.abs(rng.standard_normal(6)) + 1.0
        for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
            rho = synthetic_relational_degree(x, y, theta=theta)
            assert 0.0 < rho <= 1.0


def test_synthetic_identity_yields_unity_at_any_theta() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    for theta in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(
            synthetic_relational_degree(x, x, theta=theta), 1.0, atol=1e-12,
        )
