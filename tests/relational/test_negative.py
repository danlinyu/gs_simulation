"""Tests for negative grey relational analysis (Liu §5.7)."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.relational.negative import negative_similitude_relational_degree


@pytest.mark.book_example
def test_negative_book_example_liu_2022() -> None:
    """Liu §5.7 Example 1 (Liu 2022): X₁=(1,2,3,3,5), X₂=(5,4,2,2,1) → φᴺ ≈ -16/17."""
    X1 = np.array([1.0, 2.0, 3.0, 3.0, 5.0])
    X2 = np.array([5.0, 4.0, 2.0, 2.0, 1.0])
    phi = negative_similitude_relational_degree(X1, X2)
    np.testing.assert_allclose(phi, -16 / 17, atol=1e-12)


def test_negative_identity_yields_zero() -> None:
    """Identical sequences have no inverse coupling; φᴺ = 0 (Axiom 5.7.1)."""
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert negative_similitude_relational_degree(x, x) == 0.0


def test_negative_in_correct_range() -> None:
    """Liu Axiom 5.7.1: −1 < φᴺ ≤ 0."""
    rng = np.random.default_rng(0)
    for _ in range(15):
        x = rng.standard_normal(7)
        y = rng.standard_normal(7)
        phi = negative_similitude_relational_degree(x, y)
        assert -1.0 < phi <= 0.0


def test_negative_stronger_inverse_more_negative() -> None:
    """Larger |s_i − s_j| → larger |φᴺ| (Axiom 5.7.2 reversibility)."""
    inc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mild_dec = np.array([5.0, 4.5, 4.0, 3.5, 3.0])
    strong_dec = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    phi_mild = negative_similitude_relational_degree(inc, mild_dec)
    phi_strong = negative_similitude_relational_degree(inc, strong_dec)
    assert phi_strong < phi_mild  # stronger inverse → more negative


def test_negative_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        negative_similitude_relational_degree(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]),
        )
