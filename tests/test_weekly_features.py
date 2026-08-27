"""A weekly feature must not know its own week.

On a seventeen-game series an expanding mean that forgets to shift includes the
week being predicted, which is a large share of the average. The resulting model
validates beautifully and cannot be used, and nothing about the metrics says so.

The decisive test is not that a number looks right, it is that changing an
outcome cannot change any feature at or before the week it happened in --
and *does* change one after, because a feature layer that ignores history
entirely would pass the first half on its own.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly.features import (
    FEATURE_COLUMNS,
    add_features,
    relevant_population,
)


def _panel(weeks: int = 8, players: int = 3) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for player in range(players):
        for week in range(1, weeks + 1):
            played = int((week + player) % 4 != 0)
            rows.append(
                {
                    "player_key": f"P{player}",
                    "player_id": f"P{player}",
                    "season": 2020,
                    "week": week,
                    "team": f"T{player}",
                    "opponent": f"O{week}",
                    "position": ["QB", "RB", "WR"][player],
                    "played": played,
                    "points": float(rng.integers(0, 25)) if played else 0.0,
                    "targets": float(rng.integers(0, 10)) if played else 0.0,
                    "rush_att": float(rng.integers(0, 15)) if played else 0.0,
                    "pass_att": 0.0,
                    "receptions": 0.0,
                    "team_targets": 30.0,
                    "team_rush_att": 25.0,
                    "team_plays": 60.0,
                    "team_points": 40.0,
                    "team_pass_att": 35.0,
                }
            )
    return pd.DataFrame(rows)


def test_features_do_not_see_their_own_week() -> None:
    panel = _panel()
    base = add_features(panel)

    # Move one outcome by a large amount, in a week every player has history for.
    target_week = 5
    perturbed = panel.copy()
    hit = (perturbed["player_key"] == "P0") & (perturbed["week"] == target_week)
    assert hit.sum() == 1
    perturbed.loc[hit, "points"] = perturbed.loc[hit, "points"] + 1000.0
    perturbed.loc[hit, "targets"] = perturbed.loc[hit, "targets"] + 50.0
    after = add_features(perturbed)

    key = ["player_key", "week"]
    base = base.sort_values(key).reset_index(drop=True)
    after = after.sort_values(key).reset_index(drop=True)
    columns = [c for c in FEATURE_COLUMNS if c in base.columns]

    at_or_before = base["week"] <= target_week
    pd.testing.assert_frame_equal(
        base.loc[at_or_before, columns],
        after.loc[at_or_before, columns],
        check_exact=False,
        rtol=1e-12,
    )

    # And the change must actually reach the future, or the lag above is
    # passing because the features are constant rather than because they lag.
    later = (base["player_key"] == "P0") & (base["week"] > target_week)
    differences = [
        column
        for column in columns
        if not np.allclose(
            base.loc[later, column].fillna(-999.0),
            after.loc[later, column].fillna(-999.0),
        )
    ]
    assert differences, "perturbing an outcome changed no later feature"


def test_row_order_does_not_change_a_feature() -> None:
    """History must follow the row, not the position it happens to sit at.

    A grouped expanding statistic comes back ordered by group rather than by
    frame. Realigning it positionally works while the input is already sorted by
    that group and misaligns silently when it is not -- every player would get
    somebody else's history, with no error and entirely plausible numbers.
    """
    panel = _panel()
    key = ["player_key", "week"]
    ordered = add_features(panel).sort_values(key).reset_index(drop=True)
    shuffled = (
        add_features(panel.sample(frac=1.0, random_state=3))
        .sort_values(key)
        .reset_index(drop=True)
    )
    columns = [c for c in FEATURE_COLUMNS if c in ordered.columns]
    pd.testing.assert_frame_equal(
        ordered[columns], shuffled[columns], check_exact=False, rtol=1e-12
    )


def test_first_week_of_a_career_has_no_history() -> None:
    frame = add_features(_panel())
    first = frame.sort_values(["player_key", "season", "week"]).groupby("player_key").head(1)
    assert (first["prior_weeks"] == 0).all()
    assert (first["prior_games"] == 0).all()
    assert first["prior_points_mean"].isna().all()
    assert first["prior_points_recent"].isna().all()


def test_prior_games_counts_only_earlier_weeks() -> None:
    frame = add_features(_panel()).sort_values(["player_key", "week"])
    for _, block in frame.groupby("player_key"):
        played = block["played"].to_numpy()
        expected = np.concatenate([[0.0], np.cumsum(played)[:-1]])
        np.testing.assert_allclose(block["prior_games"].to_numpy(), expected)


def test_conditional_average_ignores_weeks_he_sat_out() -> None:
    """``prior_points_given_played`` must exclude the zeros, not average them in."""
    frame = add_features(_panel()).sort_values(["player_key", "week"])
    for _, block in frame.groupby("player_key"):
        points = block["points"].to_numpy(float)
        played = block["played"].to_numpy(bool)
        got = block["prior_points_given_played"].to_numpy(float)
        for i in range(len(block)):
            earlier = points[:i][played[:i]]
            if len(earlier) == 0:
                assert np.isnan(got[i])
            else:
                assert got[i] == pytest.approx(earlier.mean())


def test_weeks_since_played_tracks_the_gap() -> None:
    panel = _panel(weeks=6, players=1)
    panel["played"] = [1, 0, 0, 1, 1, 0]
    panel["points"] = [10.0, 0.0, 0.0, 12.0, 8.0, 0.0]
    frame = add_features(panel).sort_values("week")
    # Before his first appearance the quantity is undefined, not zero.
    assert np.isnan(frame["weeks_since_played"].iloc[0])
    np.testing.assert_allclose(
        frame["weeks_since_played"].to_numpy()[1:], [1.0, 2.0, 3.0, 1.0, 1.0]
    )


def test_relevant_population_reads_only_lagged_columns() -> None:
    """The population filter must not be a function of the outcome."""
    panel = _panel()
    base = add_features(panel)
    perturbed = panel.copy()
    perturbed["points"] = perturbed["points"] * 5.0 + 3.0
    after = add_features(perturbed)

    key = ["player_key", "week"]
    left = relevant_population(base.sort_values(key).reset_index(drop=True))
    right = relevant_population(after.sort_values(key).reset_index(drop=True))
    # Only rows whose *history* changed may move; week 1 never can.
    first = base.sort_values(key).reset_index(drop=True)["week"] == 1
    assert (left[first] == right[first]).all()
