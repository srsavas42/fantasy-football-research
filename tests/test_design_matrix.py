"""Contract for the design-matrix helpers the season models share.

Six modules build their design through these, so a change here moves the
availability, opportunity, efficiency and allocation layers at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffmodel.models.design import (
    collinearity_projection,
    project,
    standardize,
    varying_features,
)


def test_a_constant_column_is_not_a_feature():
    rows = pd.DataFrame({"varies": [1.0, 2.0, 3.0], "flat": [7.0, 7.0, 7.0]})
    assert varying_features(rows, ("varies", "flat")) == ["varies"]


def test_a_column_the_frame_does_not_carry_is_skipped():
    rows = pd.DataFrame({"varies": [1.0, 2.0, 3.0]})
    assert varying_features(rows, ("varies", "absent")) == ["varies"]


def test_an_all_null_column_is_not_a_feature():
    rows = pd.DataFrame({"varies": [1.0, 2.0], "empty": [np.nan, np.nan]})
    assert varying_features(rows, ("varies", "empty")) == ["varies"]


def test_candidate_groups_concatenate_without_repeating_a_name():
    rows = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 5.0]})
    # A base set and a model's opt-in extras that overlap it must not produce
    # the same column twice.
    assert varying_features(rows, ("a", "b"), ("b", "a")) == ["a", "b"]


def test_fitting_records_the_constants_and_centres_the_design():
    rows = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    fill, mean, scale = {}, {}, {}
    matrix = standardize(rows, ["x"], fill, mean, scale, fit=True)
    assert fill == {"x": 2.0}
    assert mean == {"x": 2.0}
    assert scale["x"] == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert matrix.mean() == pytest.approx(0.0)


def test_a_holdout_row_is_scaled_by_the_training_constants():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    fill, mean, scale = {}, {}, {}
    standardize(train, ["x"], fill, mean, scale, fit=True)

    # Same value, a frame whose own mean is nowhere near the training mean.
    holdout = pd.DataFrame({"x": [3.0, 99.0, 100.0]})
    applied = standardize(holdout, ["x"], fill, mean, scale)
    expected = (3.0 - mean["x"]) / scale["x"]
    assert applied[0, 0] == pytest.approx(expected)


def test_a_missing_value_is_filled_from_the_training_median():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    fill, mean, scale = {}, {}, {}
    standardize(train, ["x"], fill, mean, scale, fit=True)

    holdout = pd.DataFrame({"x": [np.nan]})
    applied = standardize(holdout, ["x"], fill, mean, scale)
    assert applied[0, 0] == pytest.approx((fill["x"] - mean["x"]) / scale["x"])


def test_a_zero_spread_column_does_not_divide_by_zero():
    rows = pd.DataFrame({"flat": [4.0, 4.0]})
    fill, mean, scale = {}, {}, {}
    matrix = standardize(rows, ["flat"], fill, mean, scale, fit=True)
    assert scale["flat"] == 1.0
    assert np.isfinite(matrix).all()


def test_no_features_still_returns_a_design_with_the_row_count():
    rows = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    matrix = standardize(rows, [], {}, {}, {}, fit=True)
    assert matrix.shape == (3, 0)


def test_a_duplicated_column_loses_a_direction_in_the_projection():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(60, 1))
    matrix = np.column_stack([base, base, rng.normal(size=(60, 1))])
    projection = collinearity_projection(matrix)
    # Three columns, but only two independent directions.
    assert projection.shape == (3, 2)


def test_an_empty_design_projects_to_an_empty_basis():
    assert collinearity_projection(np.zeros((5, 0))).shape == (0, 0)


def test_an_unfitted_projection_passes_the_design_through():
    matrix = np.arange(6, dtype=float).reshape(3, 2)
    assert project(matrix, None) is matrix


def test_projecting_a_full_rank_design_preserves_its_row_geometry():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(40, 3))
    rotated = project(matrix, collinearity_projection(matrix))
    # The basis is a rotation, so distances between rows survive it.
    assert np.allclose(
        np.linalg.norm(matrix[0] - matrix[1]),
        np.linalg.norm(rotated[0] - rotated[1]),
    )
