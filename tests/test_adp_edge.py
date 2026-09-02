"""The probability model has to degrade safely and must not fake discrimination.

Three ways it could mislead. It could return a confident-looking number from a
fit that never converged or never had the rows to attempt, which is worse than
returning the base rate. It could hand back a probability for a player whose gap
could not be computed, where a zero gap silently becomes "even money" rather than
"no opinion". And a constant predictor could score AUC 1.0 if ties are ranked
naively, which would make a model that says nothing look perfect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.models.adp_edge import (
    MIN_ROWS,
    AdpEdgeModel,
    auc,
    brier_score,
    calibration_table,
)


def _separable(n=800, seed=0):
    rng = np.random.default_rng(seed)
    gap = rng.normal(0, 30, n)
    beat = (0.4 * gap + rng.normal(0, 40, n) > 0).astype(float)
    return gap, beat


def test_it_learns_a_positive_slope_when_the_gap_is_informative():
    gap, beat = _separable()
    model = AdpEdgeModel().fit(gap, beat)
    assert model.fitted
    assert model.slope > 0


def test_too_few_rows_returns_the_base_rate_rather_than_a_curve():
    gap, beat = _separable(n=MIN_ROWS - 1)
    model = AdpEdgeModel().fit(gap, beat)
    assert not model.fitted
    assert np.allclose(model.predict(gap), beat.mean())


def test_a_single_outcome_class_does_not_produce_a_fit():
    gap = np.linspace(-50, 50, 400)
    model = AdpEdgeModel().fit(gap, np.ones(400))
    assert not model.fitted


def test_a_missing_gap_is_no_opinion_not_even_money():
    gap, beat = _separable()
    model = AdpEdgeModel().fit(gap, beat)
    got = model.predict(np.array([np.nan]))
    assert got[0] == model.base_rate


def test_the_gap_is_scaled_so_early_picks_do_not_dominate():
    """Doubling every gap must not change the fitted probabilities."""
    gap, beat = _separable()
    one = AdpEdgeModel().fit(gap, beat).predict(gap)
    two = AdpEdgeModel().fit(gap * 2.0, beat).predict(gap * 2.0)
    assert np.allclose(one, two, atol=1e-6)


def test_a_constant_prediction_scores_a_coin_flip_not_a_perfect_auc():
    """Naive tie handling would rank a useless model at 1.0."""
    beat = np.array([1.0, 0.0] * 50)
    assert abs(auc(np.full(100, 0.5), beat) - 0.5) < 1e-9


def test_auc_recognises_a_correct_ordering():
    beat = np.array([0.0] * 50 + [1.0] * 50)
    assert auc(np.arange(100, dtype=float), beat) > 0.99


def test_brier_beats_the_base_rate_when_the_signal_is_real():
    gap, beat = _separable()
    model = AdpEdgeModel().fit(gap, beat)
    assert brier_score(model.predict(gap), beat) < brier_score(
        np.full(len(beat), beat.mean()), beat
    )


def test_calibration_table_reports_buckets_that_sum_to_the_population():
    gap, beat = _separable()
    model = AdpEdgeModel().fit(gap, beat)
    table = calibration_table(model.predict(gap), beat, bins=5)
    assert not table.empty
    assert table.n.sum() == len(beat)
