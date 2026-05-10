"""Tests for buffer + moving-average operators (Liu §4.2-4.5)."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.operators.buffer import (
    apply_asbo,
    apply_awbo,
    apply_gfbo,
    apply_wasbo,
    apply_wawbo,
    apply_wgawbo,
)
from gs_simulation.operators.moving import (
    is_quasi_smooth,
    mean_operator,
    moving_average_denoise,
    smoothness_ratio,
    stepwise_ratio,
)


# ---------------------------------------------------------------------------
# AWBO — Liu §4.3 Example 4.3.1
# ---------------------------------------------------------------------------


@pytest.mark.book_example
def test_awbo_book_example_4_3_1() -> None:
    """Liu §4.3 Ex 4.3.1: X = (36.5, 54.3, 80.1, 109.8, 143.2)
    → X_D ≈ (84.78, 96.85, 111.03, 126.50, 143.20)."""
    X = np.array([36.5, 54.3, 80.1, 109.8, 143.2])
    expected = np.array([84.78, 96.85, 111.03, 126.50, 143.20])
    actual = apply_awbo(X)
    np.testing.assert_allclose(actual, expected, atol=1e-2)


def test_awbo_fixed_point_axiom() -> None:
    """Last element preserved (Axiom 4.2.1)."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = apply_awbo(x)
    assert out[-1] == x[-1]


def test_awbo_weakens_increasing_sequence() -> None:
    """Theorem 4.2.1 (increasing): weakening ⟺ x_D(k) ≥ x(k) for all k."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = apply_awbo(x)
    assert np.all(out >= x)


# ---------------------------------------------------------------------------
# WAWBO
# ---------------------------------------------------------------------------


def test_wawbo_with_uniform_weights_equals_awbo() -> None:
    x = np.array([2.0, 5.0, 7.0, 11.0])
    w = np.ones_like(x)
    np.testing.assert_allclose(
        apply_wawbo(x, w), apply_awbo(x), atol=1e-12,
    )


def test_wawbo_negative_weights_raise() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        apply_wawbo(np.array([1.0, 2.0, 3.0]), np.array([1.0, -1.0, 1.0]))


def test_wawbo_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must equal"):
        apply_wawbo(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0]))


# ---------------------------------------------------------------------------
# WGAWBO
# ---------------------------------------------------------------------------


def test_wgawbo_uniform_weights_yields_geometric_mean_of_tail() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    w = np.ones_like(x)
    out = apply_wgawbo(x, w)
    # First element = geometric mean of (1, 2, 4, 8) = (1*2*4*8)^(1/4) = 64^(1/4) ≈ 2.828
    np.testing.assert_allclose(out[0], 64.0 ** 0.25, atol=1e-9)


def test_wgawbo_non_positive_input_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive input"):
        apply_wgawbo(np.array([1.0, 0.0, 4.0]), np.array([1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# ASBO / WASBO
# ---------------------------------------------------------------------------


def test_asbo_strengthens_decreasing_sequence() -> None:
    """Theorem 4.2.2 (decreasing): strengthening ⟺ x_D(k) ≥ x(k) for all k."""
    x = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    out = apply_asbo(x)
    # ASBO is strengthening regardless; check fixed point.
    assert out[-1] == x[-1]


def test_wasbo_uniform_weights_close_to_asbo_pattern() -> None:
    x = np.array([2.0, 4.0, 5.0, 7.0])
    w = np.ones_like(x)
    out_w = apply_wasbo(x, w)
    # Both have x_d(n) = x(n).
    assert out_w[-1] == x[-1]


# ---------------------------------------------------------------------------
# GFBO — α = -1 reduces to WAWBO; α = +1 reduces to WASBO.
# ---------------------------------------------------------------------------


def test_gfbo_alpha_zero_is_identity() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    np.testing.assert_array_equal(apply_gfbo(x, alpha=0.0), x)


def test_gfbo_alpha_minus_1_recovers_wawbo() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    w = np.ones_like(x)
    np.testing.assert_allclose(
        apply_gfbo(x, alpha=-1.0, weights=w),
        apply_wawbo(x, w),
        atol=1e-9,
    )


def test_gfbo_alpha_plus_1_recovers_wasbo() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    w = np.ones_like(x)
    np.testing.assert_allclose(
        apply_gfbo(x, alpha=1.0, weights=w),
        apply_wasbo(x, w),
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Mean / Moving Average / Smoothness / Stepwise / Quasi-smooth
# ---------------------------------------------------------------------------


def test_mean_operator_returns_n_minus_1_elements() -> None:
    x = np.array([1.0, 3.0, 5.0, 7.0])
    z = mean_operator(x)
    assert z.shape == (3,)
    np.testing.assert_array_equal(z, np.array([2.0, 4.0, 6.0]))


def test_mean_operator_too_short_raises() -> None:
    with pytest.raises(ValueError, match="n ≥ 2"):
        mean_operator(np.array([5.0]))


def test_moving_average_m1_smooths_to_n_minus_2() -> None:
    """Liu §4.4 Eq 4.13: m=1 → 3-item average; output spans [m+1, n−m]."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = moving_average_denoise(x, m=1)
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, np.array([2.0, 3.0, 4.0]))


def test_moving_average_m2_smooths_to_n_minus_4() -> None:
    """Liu §4.4 Eq 4.14: m=2 → 5-item average; output length n−4."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    out = moving_average_denoise(x, m=2)
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, np.array([3.0, 4.0, 5.0]))


def test_moving_average_too_short_raises() -> None:
    with pytest.raises(ValueError, match="exceeds input"):
        moving_average_denoise(np.array([1.0, 2.0]), m=2)


def test_smoothness_ratio_shape_n_minus_1() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    rho = smoothness_ratio(x)
    assert rho.shape == (3,)


def test_smoothness_ratio_decreasing_for_geometric_growth() -> None:
    """Geometric growth has decreasing ρ as cumsum dominates."""
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    rho = smoothness_ratio(x)
    # ρ(2) = 2/1 = 2; ρ(3) = 4/3 = 1.33; ρ(4) = 8/7 ≈ 1.14; ρ(5) = 16/15 ≈ 1.07
    assert np.all(rho[1:] < rho[:-1])


def test_stepwise_ratio_constant_for_geometric_sequence() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    sigma = stepwise_ratio(x)
    np.testing.assert_allclose(sigma, np.full(4, 2.0), atol=1e-12)


def test_stepwise_ratio_zero_predecessor_raises() -> None:
    with pytest.raises(ValueError, match="x\\(k−1\\) = 0"):
        stepwise_ratio(np.array([0.0, 1.0, 2.0]))


def test_quasi_smooth_passes_when_rho_below_threshold() -> None:
    """Liu §4.5 quasi-smoothness is strict: ρ(k) ≤ ε for k=3..n with ε < 0.5.

    Most typical growing systems have ρ(3) ≈ x(3) / (x(1)+x(2)) ~ 0.5 and
    fail Condition 2. Quasi-smoothness is satisfied when the cumulative
    sum dominates each new value, e.g., when later values are small
    relative to early ones.
    """
    # Construction: a large-then-small sequence whose ρ(k) for k≥3 is tiny.
    x = np.array([1.0, 5.0, 1.0, 1.0, 1.0])
    # ρ(2) = 5/1 = 5; ρ(3) = 1/6 ≈ 0.167; ρ(4) ≈ 0.143; ρ(5) = 0.125.
    # All ρ(k) for k≥3 are < ε=0.4. All decreasing. ε < 0.5.
    assert is_quasi_smooth(x, epsilon=0.4)


def test_quasi_smooth_volatile_fails() -> None:
    x = np.array([1.0, 5.0, 0.1, 8.0, 0.5])
    assert not is_quasi_smooth(x, epsilon=0.4)


def test_quasi_smooth_epsilon_too_large_returns_false() -> None:
    """Liu Condition 3: ε must be strictly < 0.5."""
    x = np.array([1.0, 1.05, 1.10, 1.16])
    assert not is_quasi_smooth(x, epsilon=0.5)
    assert not is_quasi_smooth(x, epsilon=0.6)
