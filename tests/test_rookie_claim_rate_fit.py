"""The rookie claim curve, refitted on a per-snap rate.

The defect these guard against is a units mismatch rather than a bad number:
draft_calibration fitted the curve against a volume *share*, while _role_prior
consumes it as a per-snap *rate* and the softmax then multiplies by exposure.
See scripts/diagnose_cold_start_prior.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffmodel.features import draft_calibration as calib
from ffmodel.features.draft import (
    LEGACY_SHARE_FIT_CURVES,
    ROOKIE_CLAIM_CURVES,
    _claim,
)


def test_draft_capital_is_no_longer_applied_twice():
    """Exposure already carries draft capital at 7.14x round-1 to undrafted.

    The share-fit curve applied another 29.96x on top, so the softmax -- which
    adds log(role_prior) to log(exposure) -- was pricing roughly 214x where the
    observed per-snap rate varies by 1.79x. The refit lands near that.
    """
    ratio = _claim(16, "WR", "target") / _claim(None, "WR", "target")
    assert 1.3 < ratio < 2.5, ratio

    legacy_base, legacy_scale = LEGACY_SHARE_FIT_CURVES[("WR", "target")]
    legacy = (legacy_base * np.exp(-(16 - 1) / legacy_scale)) / (
        legacy_base * np.exp(-(220 - 1) / legacy_scale)
    )
    assert legacy > 25, legacy


def test_structural_zeros_survive_the_refit():
    """A refit aimed at one defect is not a reason to reopen settled ones.

    The rate fit returns small non-zero claims for these (WR carry 0.0097,
    TE carry 0.0010, QB target 0.0004); they stay zero because earlier holdout
    results put them there.
    """
    for position, stream in (("WR", "carry"), ("QB", "target"),
                             ("TE", "carry"), ("RB", "pass")):
        assert ROOKIE_CLAIM_CURVES[(position, stream)][0] == 0.0, (position, stream)
        # And the zero survives evaluation, at any pick and undrafted alike.
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
    """61% of cold-start rows are undrafted. Under the share fit they were
    excluded from the fit entirely and then served by extrapolating the
    exponential past the end of its own data."""
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
