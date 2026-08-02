"""The ridge's log-share response must not be built on an arbitrary floor.

``RidgeRosterBaseline`` regresses ``log(share) - log(role)``. A raw roster share
is exactly zero for a large minority of rows — 35% of target rows on the
nflverse frame — so the response has to be defined for zeros somehow. Flooring
them at ``1e-5`` puts that whole population on one constant far outside the
range of the real data, and the regression then spends its fit reaching toward
it. Laplace smoothing within the roster sets the value of a zero from the
roster's size instead, which is what ``_estimate_role_innovation`` already does.

This matters beyond the baseline: ``add_walk_forward_volume_features`` uses this
same estimator to build the ``oof_*`` columns the promoted efficiency models
read as covariates.
"""

import numpy as np
import pandas as pd

from ffmodel.evaluation.season_average import _availability_adjusted_share


def _roster(counts: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023] * len(counts),
            "team": ["A"] * len(counts),
            "position": (["WR"] * len(counts)),
            "targets": counts,
            "observed_availability": [1.0] * len(counts),
        }
    )


def test_the_unsmoothed_share_is_a_simplex_with_exact_zeros():
    rows = _roster([100.0, 60.0, 0.0, 0.0])

    share = _availability_adjusted_share(rows, "target")

    assert share.sum() == 1.0
    assert (share[2:] == 0.0).all()


def test_smoothing_removes_the_zeros_without_needing_a_floor():
    rows = _roster([100.0, 60.0, 0.0, 0.0])

    smoothed = _availability_adjusted_share(rows, "target", smoothed=True)

    assert (smoothed > 0).all()
    assert np.isfinite(np.log(smoothed)).all()
    # Still essentially a share: it sums to one up to the smoothing mass.
    assert smoothed.sum() == 1.0


def test_a_zero_is_valued_by_roster_size_not_by_a_constant():
    # The whole point: two rosters of different size give a zero-volume player
    # different responses, because the evidence against him differs.
    small = _availability_adjusted_share(_roster([100.0, 0.0]), "target", smoothed=True)
    large = _availability_adjusted_share(
        _roster([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "target", smoothed=True
    )

    assert small[1] != large[1]
    # A hard floor would have made both exactly 1e-5.
    assert small[1] > 1e-5 and large[1] > 1e-5


def test_smoothing_compresses_the_response_spread():
    # The floored response is bimodal: a spike of floored rows far below a
    # cluster of real ones. That spread is what the ridge was fitting.
    rows = _roster([100.0, 60.0, 20.0] + [0.0] * 7)
    role = np.full(len(rows), 0.1)

    floored = np.log(
        np.clip(_availability_adjusted_share(rows, "target"), 1e-5, None)
    ) - np.log(role)
    smoothed = np.log(
        _availability_adjusted_share(rows, "target", smoothed=True)
    ) - np.log(role)

    assert smoothed.std() < floored.std() / 2
