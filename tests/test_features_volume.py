"""Usage shares and efficiency ratios: conservation and divide-by-zero safety."""

import numpy as np
import pandas as pd

from ffmodel.data import legacy
from ffmodel.features.efficiency_hist import add_efficiency
from ffmodel.features.volume import team_game_totals, usage_shares


def test_target_share_sums_to_one_per_team_week():
    pw = legacy.load_weekly([2020])
    sh = usage_shares(pw)
    # Every team-week with any targets must have shares summing to 1.
    totals = sh.groupby(["season", "week", "team"])["target_share"].sum()
    played = team_game_totals(pw).set_index(["season", "week", "team"])["team_targets"]
    active = totals[played.reindex(totals.index) > 0]
    assert np.allclose(active.to_numpy(), 1.0, atol=1e-9)


def test_carry_share_sums_to_one_per_team_week():
    pw = legacy.load_weekly([2019])
    sh = usage_shares(pw)
    totals = sh.groupby(["season", "week", "team"])["carry_share"].sum()
    rush = team_game_totals(pw).set_index(["season", "week", "team"])["team_rush_att"]
    active = totals[rush.reindex(totals.index) > 0]
    assert np.allclose(active.to_numpy(), 1.0, atol=1e-9)


def test_shares_have_no_inf_or_nan():
    sh = usage_shares(legacy.load_weekly([2020]))
    for col in ("target_share", "carry_share", "opportunity_share", "wopr"):
        assert np.isfinite(sh[col].to_numpy()).all()


def test_efficiency_zero_denominator_is_nan_not_inf():
    # A player with no targets/carries must not produce inf efficiency, and the
    # NaN ratio must not poison the trailing EWMA of players who do have data.
    df = pd.DataFrame(
        {
            "player_id": [None] * 3,
            "player_name": ["Z"] * 3,
            "position": ["RB"] * 3,
            "team": ["A"] * 3,
            "season": [2020] * 3,
            "week": [1, 2, 3],
            "rush_att": [0.0, 10.0, 10.0],
            "rush_yds": [0.0, 50.0, 40.0],
            "rush_td": [0.0, 1.0, 0.0],
            "targets": [0.0, 0.0, 0.0],
            "receptions": [0.0, 0.0, 0.0],
            "rec_yds": [0.0, 0.0, 0.0],
            "rec_td": [0.0, 0.0, 0.0],
        }
    )
    out = add_efficiency(df, span=3)
    assert pd.isna(out["ypc"].iloc[0])  # 0 carries -> undefined
    assert np.isfinite(out["ypc"].dropna().to_numpy()).all()
    # Week 3 trailing ypc should reflect week 2 (5.0), unaffected by week-1 NaN.
    assert np.isclose(out["ewma_ypc"].iloc[2], 5.0)
