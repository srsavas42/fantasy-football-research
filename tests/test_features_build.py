"""Roles and the end-to-end build_features contract."""

import numpy as np
import pandas as pd
import pytest

from ffmodel.features import FEATURE_COLUMNS, build_features
from ffmodel.features.roles import add_roles


def _frame(rows):
    cols = ["player_name", "position", "team", "season", "week",
            "ewma_opportunity_share", "opportunity_share"]
    return pd.DataFrame(rows, columns=cols)


def test_role_rank_orders_by_trailing_signal():
    # Three WRs on one team-week with decreasing trailing usage.
    df = _frame([
        ["A", "WR", "T", 2020, 5, 0.30, 0.3],
        ["B", "WR", "T", 2020, 5, 0.20, 0.2],
        ["C", "WR", "T", 2020, 5, 0.10, 0.1],
    ])
    df["player_id"] = None
    out = add_roles(df).sort_values("player_name")
    ranks = dict(zip(out["player_name"], out["role_rank"]))
    assert ranks["A"] == 1 and ranks["B"] == 2 and ranks["C"] == 3


def test_cold_start_uses_prior_season_when_no_history():
    # Player D has a prior-season (2019) opportunity share but a NaN trailing
    # signal in week 1 of 2020; role should still rank via the prior season.
    df = _frame([
        ["D", "WR", "T", 2019, 1, np.nan, 0.40],
        ["E", "WR", "T", 2019, 1, np.nan, 0.10],
        ["D", "WR", "T", 2020, 1, np.nan, np.nan],
        ["E", "WR", "T", 2020, 1, np.nan, np.nan],
    ])
    df["player_id"] = None
    out = add_roles(df)
    wk1_2020 = out[(out["season"] == 2020)].set_index("player_name")
    assert wk1_2020.loc["D", "role_rank"] == 1  # higher prior-season usage
    assert wk1_2020.loc["D", "role_tier"] is not pd.NA


def test_non_skill_positions_have_no_tier():
    df = _frame([["Q", "QB", "T", 2020, 5, 0.9, 0.9]])
    df["player_id"] = None
    out = add_roles(df)
    assert pd.isna(out["role_tier"].iloc[0])


@pytest.mark.parametrize("seasons", [[2020], [2019, 2020]])
def test_build_features_contract(seasons):
    df = build_features(seasons, source="legacy")
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"missing {col}"
    # No feature column should be entirely empty.
    for col in FEATURE_COLUMNS:
        assert not df[col].isna().all(), f"all-NaN: {col}"
    assert set(df["season"].unique()) == set(seasons)
    assert (df["is_active"].isin([0, 1])).all()


def test_build_features_team_season_key():
    df = build_features([2020], source="legacy")
    sample = df.iloc[0]
    assert sample["team_season"] == f"{sample['team']}_{sample['season']}"
