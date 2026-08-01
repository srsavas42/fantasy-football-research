"""Rookie draft-capital prior: curve fitting and the shipped policy.

These curves are consumed as a prior wherever a lagged role is missing, so
magnitude is used directly rather than only the ordering by draft capital.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.features import draft
from ffmodel.features.draft_calibration import (
    HAND_SET_CLAIM_CURVES,
    claim_from_curve,
    fit_claim_curve,
    fit_rookie_priors,
    rookie_seasons,
)


def test_fit_recovers_known_parameters():
    picks = np.arange(1, 250, dtype=float)
    shares = 0.4 * np.exp(-(picks - 1) / 90.0)

    base, scale = fit_claim_curve(picks, shares)

    assert base == pytest.approx(0.4, abs=0.01)
    # The scale grid steps in twos, so exact recovery is not expected.
    assert scale == pytest.approx(90.0, abs=2.0)


def test_rookie_seasons_are_the_rows_carrying_draft_capital():
    rows = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "position": ["WR", "WR", "WR"],
            "overall_pick": [4.0, np.nan, 200.0],
            "target_share": [0.2, 0.3, 0.01],
        }
    )

    assert len(rookie_seasons(rows)) == 2


def test_streams_without_enough_observations_claim_nothing():
    rows = pd.DataFrame(
        {
            "season": [2024, 2024],
            "position": ["WR", "WR"],
            "overall_pick": [4.0, 40.0],
            "target_share": [0.2, 0.1],
        }
    )

    fitted = fit_rookie_priors(rows, min_rows=20)

    # Two rows cannot support a curve; a noisy fit would be worse than none.
    assert all(base == 0.0 for base, _ in fitted.values())


def test_a_missing_pick_is_treated_as_undrafted():
    curve = (0.5, 100.0)

    assert claim_from_curve(None, curve) < claim_from_curve(200, curve)
    assert claim_from_curve(np.nan, curve) == claim_from_curve(220, curve)


def test_shipped_curves_decay_with_draft_capital():
    for (position, stream), curve in draft.ROOKIE_CLAIM_CURVES.items():
        base, _ = curve
        if base <= 0:
            continue
        early, late = claim_from_curve(1, curve), claim_from_curve(200, curve)
        assert early > late > 0, f"{position}/{stream} must decay with pick"


def test_a_rookie_backs_rushing_claim_exceeds_its_receiving_claim():
    target, carry = draft.expected_rookie_claim(1, "RB")

    assert carry > target > 0


def test_retained_streams_still_match_the_hand_set_curve():
    # Streams the calibration did not promote must be byte-for-byte unchanged,
    # so a future refit cannot quietly move something that was never validated.
    for key in (("WR", "target"), ("QB", "pass"), ("WR", "carry"), ("TE", "carry")):
        assert draft.ROOKIE_CLAIM_CURVES[key] == HAND_SET_CLAIM_CURVES[key]


def test_rookie_passer_claim_is_quarterback_only():
    assert draft.expected_rookie_pass_claim(1, "WR") == 0.0
    assert draft.expected_rookie_pass_claim(1, "QB") > 0.0
