"""Cross-season feature correctness: vacated share, returning-only, no leakage."""

import numpy as np
import pandas as pd
import pytest

from ffmodel.features import crossseason as cs


def _usage(rows):
    df = pd.DataFrame(
        rows, columns=["player_name", "position", "season", "team", "target_share", "carry_share"]
    )
    df["key"] = cs.player_key(df)
    return df


def test_vacated_equals_departed_share_sum():
    # Team T in 2019: A/B/C. In 2020 only A returns -> B+C vacate their shares.
    u = _usage([
        ["A", "WR", 2019, "T", 0.40, 0.0],
        ["B", "WR", 2019, "T", 0.30, 0.0],
        ["C", "RB", 2019, "T", 0.10, 0.50],
        ["A", "WR", 2020, "T", 0.45, 0.0],
    ])
    vac = cs.vacated_opportunity(u, 2019)
    row = vac[vac["team"] == "T"].iloc[0]
    assert np.isclose(row["vacated_target_share"], 0.30 + 0.10)  # B + C targets
    assert np.isclose(row["vacated_carry_share"], 0.50)          # C carries
    assert row["next_season"] == 2020


def test_returning_player_does_not_vacate():
    u = _usage([
        ["A", "WR", 2019, "T", 0.5, 0.0],
        ["A", "WR", 2020, "T", 0.5, 0.0],
    ])
    vac = cs.vacated_opportunity(u, 2019)
    assert vac.empty or np.isclose(vac.iloc[0]["vacated_target_share"], 0.0)


def test_transitions_are_returning_only():
    t = cs.build_transitions([2018, 2019, 2020], source="legacy")
    usage = cs.season_usage([2018, 2019, 2020], source="legacy")
    for _, r in t.sample(min(40, len(t)), random_state=0).iterrows():
        y, yp1 = map(int, r["transition"].split("->"))
        in_y = ((usage["key"] == r["key"]) & (usage["season"] == y)).any()
        in_yp1 = ((usage["key"] == r["key"]) & (usage["season"] == yp1)).any()
        assert in_y and in_yp1


def test_predictors_from_year_y_labels_from_yp1():
    # A transition's prior share must equal the player's season-Y usage, and its
    # label must equal season-(Y+1) usage: predictors never read the future.
    usage = cs.season_usage([2018, 2019], source="legacy")
    t = cs.build_transitions([2018, 2019], source="legacy")
    r = t.iloc[0]
    y, yp1 = map(int, r["transition"].split("->"))
    uy = usage[(usage["key"] == r["key"]) & (usage["season"] == y)].iloc[0]
    uyp1 = usage[(usage["key"] == r["key"]) & (usage["season"] == yp1)].iloc[0]
    assert np.isclose(r["target_share"], uy["target_share"])
    assert np.isclose(r["next_target_share"], uyp1["target_share"])


def test_shares_within_unit_interval():
    t = cs.build_transitions([2018, 2019, 2020], source="legacy")
    for col in ("target_share", "carry_share", "next_target_share", "next_carry_share"):
        assert (t[col] >= -1e-9).all() and (t[col] <= 1 + 1e-9).all()


def test_a_player_who_stayed_but_never_played_is_not_counted_as_vacated():
    """Usage is built from stat rows, so "hurt all year" looks like "left".

    That overcount is not neutral: it inflates vacated opportunity with exactly
    the injury events the model exists to predict. Roster membership separates
    the two.
    """
    usage = pd.DataFrame(
        [
            {"season": 2019, "team": "KC", "key": "star|WR",
             "target_share": 0.30, "carry_share": 0.0},
            {"season": 2019, "team": "KC", "key": "gone|WR",
             "target_share": 0.20, "carry_share": 0.0},
            # 2020: only one of them records a stat line.
            {"season": 2020, "team": "KC", "key": "other|WR",
             "target_share": 0.50, "carry_share": 0.0},
        ]
    )

    # Without roster information both look departed: 0.30 + 0.20.
    blind = cs.vacated_opportunity(usage, 2019)
    assert blind.loc[0, "vacated_target_share"] == pytest.approx(0.50)

    # The star was on the 2020 roster and simply did not play.
    roster = pd.DataFrame([{"season": 2020, "team": "KC", "key": "star|WR"}])
    informed = cs.vacated_opportunity(usage, 2019, roster)
    assert informed.loc[0, "vacated_target_share"] == pytest.approx(0.20)


def test_an_empty_current_season_returns_empty_rather_than_raising():
    # The row-wise membership test this replaced raised KeyError here, because
    # DataFrame.apply over zero rows returns a frame, not a boolean Series.
    usage = pd.DataFrame(
        [{"season": 2020, "team": "KC", "key": "a|WR",
          "target_share": 0.2, "carry_share": 0.0}]
    )

    assert cs.vacated_opportunity(usage, 2019).empty
