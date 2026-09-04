"""Rate-mode fitting in draft_calibration, and the curve it did not replace.

A 2026-09 refit moved ROOKIE_CLAIM_CURVES onto a per-snap rate and was reverted
after the gate rejected it; docs/target-competition-2026-09.md has the numbers.
Rate mode stays, because the measurement is worth being able to repeat, so
these pin what it does and what the shipped curve still is.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffmodel.features import draft_calibration as calib
from ffmodel.features.draft import ROOKIE_CLAIM_CURVES, _claim


def test_the_shipped_curve_is_steep_in_draft_slot():
    """The reverted refit flattened this to about 1.7x round-1 : undrafted.

    It has to stay steep. The prior is not only pricing per-snap usage, where
    flat is right; it is also pricing whether draft capital converts into a
    role at all, which projected exposure does not fully carry. Flattening put
    undrafted rookies on 55% of all cold prior mass and cost every fold.
    """
    ratio = _claim(16, "WR", "target") / _claim(None, "WR", "target")
    assert ratio > 25, ratio


def test_structural_zeros_stay_zero():
    """Streams earlier holdout results zeroed, at any pick and undrafted alike."""
    for position, stream in (("WR", "carry"), ("QB", "target"),
                             ("TE", "carry"), ("RB", "pass")):
        assert ROOKIE_CLAIM_CURVES[(position, stream)][0] == 0.0, (position, stream)
        assert _claim(50, position, stream) == 0.0
        assert _claim(None, position, stream) == 0.0


def _rookies() -> pd.DataFrame:
    # Two picks with identical per-snap rate but very different snap counts:
    # a share fit sees them as different, a rate fit as the same.
    return pd.DataFrame({
        "overall_pick": [5.0, 150.0, np.nan],
        "experience": [0, 0, 0],
        "position": ["WR"] * 3,
        "offense_snaps": [800.0, 400.0, 300.0],
        "targets": [80.0, 40.0, 30.0],
        "target_share": [0.24, 0.12, 0.09],
        "rush_att": [0.0, 0.0, 0.0],
        "pass_att": [0.0, 0.0, 0.0],
        "carry_share": [0.0, 0.0, 0.0],
        "pass_attempt_share": [0.0, 0.0, 0.0],
    })


def test_rate_mode_admits_the_undrafted_tail():
    """61% of cold-start rows are undrafted. The share fit excluded them from
    the fit entirely and then served them by extrapolating the exponential past
    the end of its own data."""
    drafted_only = calib.rookie_seasons(_rookies())
    with_undrafted = calib.rookie_seasons(_rookies(), include_undrafted=True)
    assert len(drafted_only) == 2
    assert len(with_undrafted) == 3
    assert (with_undrafted["overall_pick"] == calib.UNDRAFTED_PICK).sum() == 1


def test_rate_mode_reads_a_rate_not_a_share():
    """The three rookies share a per-snap rate of 0.10 and differ in share, so
    a rate fit should come back flat where a share fit slopes."""
    rate = calib.fit_rookie_priors(
        _rookies(), positions=("WR",), min_rows=1, target="rate"
    )[("WR", "target")]
    share = calib.fit_rookie_priors(
        _rookies(), positions=("WR",), min_rows=1, target="share"
    )[("WR", "target")]
    rate_ratio = np.exp(-(5 - 1) / rate[1]) / np.exp(-(150 - 1) / rate[1])
    share_ratio = np.exp(-(5 - 1) / share[1]) / np.exp(-(150 - 1) / share[1])
    assert rate_ratio < share_ratio, (rate_ratio, share_ratio)
    assert rate_ratio < 1.6, rate_ratio


def test_rate_mode_drops_rookies_with_too_few_snaps():
    """A rate over a handful of snaps is noise, and those rows are exactly the
    ones the snap model is already projecting near zero exposure for."""
    frame = _rookies()
    frame.loc[1, "offense_snaps"] = 5.0
    frame.loc[1, "targets"] = 5.0  # a 1.0 rate that would dominate any fit
    fitted = calib.fit_rookie_priors(
        frame, positions=("WR",), min_rows=1, target="rate"
    )[("WR", "target")]
    assert fitted[0] < 0.5, fitted
