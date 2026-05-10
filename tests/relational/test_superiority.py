"""Tests for superiority analysis (Liu §5.9)."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.relational.superiority import (
    SuperiorityResult,
    build_relational_matrix,
    favorable_indices,
    quasi_favorable_ranking,
    superiority_analysis,
)


def test_relational_matrix_shape_and_symmetry() -> None:
    """Γ is s × m; entries equal absolute_relational_degree(Y_i, X_j)."""
    Y = np.array([[1.0, 2.0, 4.0, 8.0], [2.0, 4.0, 6.0, 8.0]])
    X = np.array([[1.0, 2.0, 4.0, 8.0], [10.0, 9.0, 8.0, 7.0]])
    Gamma = build_relational_matrix(Y, X)
    assert Gamma.shape == (2, 2)
    # Y[0] == X[0] → identity → Γ[0, 0] = 1
    np.testing.assert_allclose(Gamma[0, 0], 1.0, atol=1e-12)


def test_favorable_indices_empty_when_no_dominator() -> None:
    """Generic random matrix usually has no dominator."""
    rng = np.random.default_rng(0)
    matrix = rng.uniform(0.5, 0.95, size=(4, 4))
    fav = favorable_indices(matrix, axis=0)
    # No row likely dominates all others entrywise.
    assert isinstance(fav, list)


def test_favorable_indices_finds_dominator() -> None:
    """Construct a dominator and verify it's identified."""
    matrix = np.array([
        [0.5, 0.6, 0.7],
        [0.6, 0.7, 0.8],   # dominates row 0 entrywise
        [0.4, 0.5, 0.6],
    ])
    fav = favorable_indices(matrix, axis=0)
    # Row 1 dominates row 0 and row 2 entrywise.
    assert 1 in fav


def test_quasi_favorable_ranking_descending_by_row_sum() -> None:
    matrix = np.array([
        [0.5, 0.5],   # sum 1.0
        [0.9, 0.8],   # sum 1.7  — biggest
        [0.6, 0.7],   # sum 1.3
    ])
    rank = quasi_favorable_ranking(matrix, axis=0)
    assert rank[0] == 1   # biggest sum first
    assert rank[-1] == 0  # smallest last


def test_quasi_favorable_ranking_axis_1_uses_column_sum() -> None:
    matrix = np.array([
        [0.5, 0.9, 0.7],
        [0.4, 0.8, 0.6],
    ])
    # Column sums: col 0 = 0.9, col 1 = 1.7, col 2 = 1.3
    rank = quasi_favorable_ranking(matrix, axis=1)
    assert rank[0] == 1
    assert rank[-1] == 0


def test_superiority_analysis_returns_full_result() -> None:
    Y = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.5, 2.5, 3.5, 4.5],
    ])
    X = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
    ])
    result = superiority_analysis(Y, X)
    assert isinstance(result, SuperiorityResult)
    assert result.Gamma.shape == (2, 2)
    assert isinstance(result.favorable_outputs, list)
    assert isinstance(result.quasi_preferred_outputs, list)


def test_relational_matrix_axes_mismatch_raises() -> None:
    Y = np.array([[1.0, 2.0, 3.0]])
    X = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="must equal"):
        build_relational_matrix(Y, X)


def test_relational_matrix_1d_input_raises() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        build_relational_matrix(np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0, 3.0]]))
