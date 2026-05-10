"""Tests for similitude + closeness relational degrees (Liu §5.6)."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.relational.closeness import closeness_relational_degree
from gs_simulation.relational.similitude import similitude_relational_degree


# ---------------------------------------------------------------------------
# Similitude
# ---------------------------------------------------------------------------


def test_similitude_identity_yields_unity() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert similitude_relational_degree(x, x) == 1.0


def test_similitude_translation_invariant() -> None:
    """ε̃ depends only on the gap |s_i − s_j| of zero-shifted sequences."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(7)
    y = rng.standard_normal(7)
    ε_orig = similitude_relational_degree(x, y)
    ε_shifted = similitude_relational_degree(x + 99.0, y - 13.0)
    np.testing.assert_allclose(ε_shifted, ε_orig, atol=1e-12)


def test_similitude_in_unit_interval() -> None:
    rng = np.random.default_rng(42)
    for _ in range(15):
        x = rng.standard_normal(6)
        y = rng.standard_normal(6)
        ε = similitude_relational_degree(x, y)
        assert 0.0 < ε <= 1.0


def test_similitude_parallel_curves_yield_unity() -> None:
    """Parallel zero-starting curves (s_i == s_j) → ε̃ = 1."""
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = np.array([10.0, 11.0, 13.0, 17.0])  # zero-shifted == x zero-shifted
    np.testing.assert_allclose(
        similitude_relational_degree(x, y), 1.0, atol=1e-12,
    )


def test_similitude_inc_vs_dec_low() -> None:
    inc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dec = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ε = similitude_relational_degree(inc, dec)
    assert ε < 0.1   # |s_i − s_j| = 16 → ε = 1/17 ≈ 0.059


# ---------------------------------------------------------------------------
# Closeness
# ---------------------------------------------------------------------------


def test_closeness_identity_yields_unity() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert closeness_relational_degree(x, x) == 1.0


def test_closeness_NOT_translation_invariant() -> None:
    """ρ̃ uses raw integrals; shifting changes |S_i − S_j|."""
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = np.array([1.0, 2.0, 4.0, 8.0])
    ρ_orig = closeness_relational_degree(x, y)
    ρ_shifted = closeness_relational_degree(x, y + 100.0)
    assert ρ_shifted != ρ_orig
    assert ρ_shifted < ρ_orig


def test_closeness_in_unit_interval() -> None:
    rng = np.random.default_rng(7)
    for _ in range(15):
        x = rng.standard_normal(6) + 5.0
        y = rng.standard_normal(6) + 5.0
        ρ = closeness_relational_degree(x, y)
        assert 0.0 < ρ <= 1.0


def test_closeness_far_apart_low() -> None:
    """Sequences with very different absolute levels have small ρ̃."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([100.0, 110.0, 120.0, 130.0])
    ρ = closeness_relational_degree(x, y)
    assert ρ < 0.01


# ---------------------------------------------------------------------------
# Edge cases (both)
# ---------------------------------------------------------------------------


def test_similitude_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        similitude_relational_degree(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_closeness_2d_input_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        closeness_relational_degree(
            np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0, 3.0, 4.0]),
        )
