"""The post-hoc width fix: what it does to a set of draws, and what it cannot do."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_hierarchical_ros import coverage, solve_factor, widen  # noqa: E402


def _draws(rng, rows=400, draws=600, scale=10.0):
    return np.maximum(rng.normal(80.0, scale, size=(rows, draws)), 0.0)


def test_a_factor_of_one_changes_nothing():
    got = _draws(np.random.default_rng(0))
    assert np.allclose(widen(got, 1.0), got)


def test_widening_holds_the_centre_and_spreads_the_rest():
    """About the median, because a season total is right-skewed and a scale
    about the mean would move the centre while it widened."""
    got = _draws(np.random.default_rng(1))
    wide = widen(got, 2.0)
    assert np.allclose(np.median(wide, axis=1), np.median(got, axis=1), atol=1e-9)
    assert (wide.std(axis=1) > got.std(axis=1)).all()


def test_a_season_total_is_never_negative():
    """Coverage bought with impossible outcomes is not coverage."""
    rng = np.random.default_rng(2)
    got = np.maximum(rng.normal(5.0, 4.0, size=(200, 300)), 0.0)
    assert (widen(got, 3.0) >= 0.0).all()


def test_the_solver_finds_a_width_that_covers_nominally():
    rng = np.random.default_rng(3)
    # The model's point estimate, the outcome that actually lands around it, and
    # draws that understate how far apart those two are -- which is the case the
    # fix is built for. Centring the draws on the truth instead would cover it
    # perfectly and test nothing.
    centre = rng.normal(80.0, 20.0, size=2000)
    truth = centre + rng.normal(0.0, 20.0, size=2000)
    samples = centre[:, None] + rng.normal(0.0, 8.0, size=(2000, 600))
    assert coverage(samples, truth) < 0.55
    factor = solve_factor(samples, truth)
    assert factor > 1.0
    assert coverage(widen(samples, factor), truth) == pytest.approx(0.80, abs=0.04)


def test_one_width_cannot_fix_a_wrong_shape():
    """The finding the document turns on.

    Draws whose tails are too thin relative to the truth -- Gaussian against a
    heavy-tailed outcome -- need a larger factor at 95% than at 50%. A single
    scalar tuned at one level therefore misses at the other, which is why the
    calibrated arm lands on 0.80 and still under-covers at 0.95.
    """
    rng = np.random.default_rng(4)
    truth = rng.standard_t(df=3, size=4000) * 20.0 + 80.0
    samples = 80.0 + rng.normal(0.0, 20.0, size=(4000, 600))

    def needed(level):
        grid = np.round(np.arange(1.0, 6.01, 0.05), 2)
        return min(grid, key=lambda f: abs(coverage(widen(samples, f), truth, level) - level))

    assert needed(0.95) > needed(0.50)
