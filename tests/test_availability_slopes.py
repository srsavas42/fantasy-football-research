"""Shared versus per-position slope vectors in the availability layer.

The risky part is serving. A model fitted with one slope vector and served with
the per-position code path (or the reverse) would not raise -- the shapes are
compatible enough to broadcast into something plausible -- so the linear
predictor reads the posterior's own shape rather than a flag.
"""

from __future__ import annotations

import numpy as np
import pytest

from ffmodel.models.season_availability import _apply_slopes


def test_a_shared_vector_is_a_plain_matrix_product():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 3))
    beta = rng.normal(size=(3, 20))
    position_index = np.array([0, 1, 2, 3, 0, 1])

    assert np.allclose(_apply_slopes(beta, X, position_index), X @ beta)


def test_each_position_uses_its_own_vector():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3))
    beta = rng.normal(size=(4, 3, 20))
    position_index = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    out = _apply_slopes(beta, X, position_index)

    assert out.shape == (8, 20)
    for row, position in enumerate(position_index):
        assert np.allclose(out[row], X[row] @ beta[position])


def test_a_hierarchy_with_identical_positions_matches_the_shared_vector():
    """The hierarchy is a generalisation, not a different model.

    If the between-position scale shrinks to zero the per-position vectors
    coincide and the linear predictor must agree with the pooled one, which is
    what makes turning the flag on a safe thing to measure.
    """
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, 4))
    shared = rng.normal(size=(4, 12))
    stacked = np.repeat(shared[None, :, :], 4, axis=0)
    position_index = np.array([0, 1, 2, 3, 1])

    assert np.allclose(
        _apply_slopes(stacked, X, position_index), _apply_slopes(shared, X, position_index)
    )


def test_an_unexpected_shape_is_refused_rather_than_broadcast():
    X = np.ones((3, 2))
    with pytest.raises(ValueError, match="unexpected shape"):
        _apply_slopes(np.ones(2), X, np.zeros(3, dtype=int))


def test_a_position_absent_from_the_rows_is_simply_unused():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3, 2))
    beta = rng.normal(size=(4, 2, 6))
    position_index = np.array([1, 1, 1])

    out = _apply_slopes(beta, X, position_index)
    assert np.allclose(out, X @ beta[1])
