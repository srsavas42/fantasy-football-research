"""The blend's contract: where its weight comes from and what it never does."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffmodel.models.market_blend import (
    MIN_RESIDUALS,
    MarketBlend,
    RankCurve,
    blend_samples,
    slope_weight,
)


def _board(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """A synthetic draft board where points really are a log curve in rank."""
    rng = np.random.default_rng(seed)
    rank = np.arange(1, n + 1, dtype=float)
    position = np.array(["QB", "RB", "WR", "TE"])[rng.integers(0, 4, n)]
    points = 300.0 - 40.0 * np.log(rank) + rng.normal(0, 15.0, n)
    rows = pd.DataFrame(
        {"adp_rank": rank, "adp_drafted": 1, "position": position}
    )
    return rows, np.maximum(points, 0.0)


def test_the_curve_recovers_a_log_relationship():
    rows, points = _board()
    curve = RankCurve().fit(rows, points)
    samples = curve.predict_samples(rows, draws=200, seed=1)

    assert samples.shape == (len(rows), 200)
    assert np.isfinite(samples).all()
    # Early picks outscore late ones, and the fit tracks the truth closely.
    mean = samples.mean(axis=1)
    assert mean[:20].mean() > mean[-20:].mean()
    assert np.corrcoef(mean, points)[0, 1] > 0.85


def test_the_curve_never_projects_negative_points():
    """A curve linear in log rank goes under zero deep on the board."""
    rows, points = _board(n=400)
    curve = RankCurve().fit(rows, points)
    deep = pd.DataFrame(
        {"adp_rank": [900.0, 1500.0], "adp_drafted": [1, 1], "position": ["WR", "RB"]}
    )
    samples = curve.predict_samples(deep, draws=100, seed=2)
    assert (samples >= 0.0).all()


def test_undrafted_rows_get_no_curve():
    rows, points = _board()
    curve = RankCurve().fit(rows, points)
    test = pd.DataFrame(
        {"adp_rank": [10.0, np.nan], "adp_drafted": [1, 0], "position": ["WR", "WR"]}
    )
    samples = curve.predict_samples(test, draws=50, seed=3)
    assert np.isfinite(samples[0]).all()
    assert np.isnan(samples[1]).all()


def test_the_curve_refuses_to_fit_on_too_little():
    rows, points = _board(n=MIN_RESIDUALS - 1)
    with pytest.raises(ValueError, match="drafted rows"):
        RankCurve().fit(rows, points)


def test_the_curve_is_fitted_on_drafted_rows_only():
    """Undrafted rows would drag the curve toward a fringe population."""
    rows, points = _board()
    padded = pd.concat(
        [
            rows,
            pd.DataFrame(
                {
                    "adp_rank": [np.nan] * 100,
                    "adp_drafted": [0] * 100,
                    "position": ["WR"] * 100,
                }
            ),
        ],
        ignore_index=True,
    )
    padded_points = np.concatenate([points, np.zeros(100)])

    drafted_only = RankCurve().fit(rows, points)
    with_fringe = RankCurve().fit(padded, padded_points)
    for name in ("QB", "RB", "WR", "TE"):
        assert np.allclose(
            drafted_only.coefficients[name], with_fringe.coefficients[name]
        )


def test_the_slope_weight_is_one_when_the_model_is_right():
    rng = np.random.default_rng(4)
    curve = rng.normal(100, 30, 500)
    model = rng.normal(100, 30, 500)
    # Observed *is* the model, so all of its disagreement with the curve is real.
    assert slope_weight(model, model, curve) == pytest.approx(1.0, abs=1e-6)


def test_the_slope_weight_is_zero_when_the_model_adds_nothing():
    rng = np.random.default_rng(5)
    curve = rng.normal(100, 30, 500)
    model = curve + rng.normal(0, 30, 500)
    # Observed is the curve, so the model's disagreement is pure noise. A
    # negative slope would mean betting against the model; it is clipped.
    assert slope_weight(curve, model, curve) == pytest.approx(0.0, abs=1e-6)


def test_the_slope_weight_stays_inside_the_unit_interval():
    rng = np.random.default_rng(6)
    curve = rng.normal(100, 30, 400)
    model = curve + rng.normal(0, 5, 400)
    observed = curve + 4.0 * (model - curve)
    assert slope_weight(observed, model, curve) == 1.0


def test_the_mixture_keeps_the_spread_that_averaging_destroys():
    """This is the whole reason the shipped combination is a mixture.

    Both give the same mean, so MAE cannot tell them apart. Averaging paired
    draws produces a distribution narrower than either input, which cost 11
    points of 80% coverage on the holdouts.
    """
    rng = np.random.default_rng(7)
    model = rng.normal(100, 40, (300, 500))
    curve = rng.normal(120, 40, (300, 500))
    weight = 0.3

    mixed = blend_samples(model, curve, weight, seed=8)
    averaged = weight * model + (1.0 - weight) * curve

    assert mixed.mean() == pytest.approx(averaged.mean(), rel=0.02)
    assert mixed.std(axis=1).mean() > averaged.std(axis=1).mean()


def test_rows_the_board_cannot_rank_keep_the_model_untouched():
    model = np.full((3, 10), 7.0)
    curve = np.full((3, 10), 99.0)
    curve[1] = np.nan
    blended = blend_samples(model, curve, 0.5, seed=9)

    assert np.isfinite(blended).all()
    assert (blended[1] == 7.0).all()


def test_a_weight_outside_the_unit_interval_is_refused():
    model = np.ones((2, 4))
    curve = np.ones((2, 4))
    for weight in (-0.1, 1.1):
        with pytest.raises(ValueError, match="weight"):
            blend_samples(model, curve, weight)


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="shape"):
        blend_samples(np.ones((2, 4)), np.ones((3, 4)), 0.5)


def test_the_blend_round_trips_through_the_dataclass():
    rows, points = _board()
    blend = MarketBlend.fit(rows, points, weight=0.3)
    model = np.full((len(rows), 64), 50.0)
    blended = blend.predict_samples(rows, model, seed=10)

    assert blended.shape == model.shape
    assert np.isfinite(blended).all()
    # Some draws came from each component.
    assert (blended == 50.0).any()
    assert (blended != 50.0).any()
