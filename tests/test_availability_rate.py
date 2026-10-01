"""The denominator for a per-active-game rate: the target, the bounds, the fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly.availability_rate import (
    RATE_FEATURES,
    TARGET,
    PlayedRate,
    add_played_rate_target,
    calibration,
)


def _frame(played: list[int]) -> pd.DataFrame:
    n = len(played)
    return pd.DataFrame({
        "player_key": ["a"] * n,
        "season": [2024] * n,
        "week": list(range(1, n + 1)),
        "played": played,
        "games_remaining": list(range(n, 0, -1)),
        **{f: np.linspace(0.5, 0.9, n) for f in RATE_FEATURES},
    })


def test_the_target_counts_forward_from_each_week():
    """Week w's share spans w to the end, matching the rest-of-season window."""
    out = add_played_rate_target(_frame([1, 0, 1, 1]))
    # From week 1: 3 of 4 played. From week 2: 2 of 3. From week 3: 2 of 2.
    assert out[TARGET].tolist() == pytest.approx([0.75, 2 / 3, 1.0, 1.0])


def test_a_player_who_never_plays_scores_zero_not_missing():
    out = add_played_rate_target(_frame([0, 0, 0]))
    assert out[TARGET].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_the_prediction_is_a_bounded_share():
    """A rate outside [0, 1] is not a share, and a zero denominator is an
    infinite per-game projection for exactly the players least worth trusting."""
    train = add_played_rate_target(_frame([1, 0, 1, 1, 0, 1, 1, 1]))
    model = PlayedRate().fit(train)
    wild = train.copy()
    for column in RATE_FEATURES:
        wild[column] = [-50.0, 50.0] * (len(wild) // 2)
    got = model.predict(wild)
    assert (got >= 0.05).all() and (got <= 1.0).all()


def test_the_fit_beats_the_feature_it_replaces():
    """`prior_play_rate` is the obvious denominator and the reason this exists.

    It is a backward-looking average read as a forward-looking probability, and
    it is compressed at both ends. If the fit cannot beat it there is no case
    for the extra machinery.
    """
    from ffmodel.weekly.restofseason import add_rest_of_season_target

    cache = Path(".cache/weekly_features_2016_2025.pkl")
    if not cache.exists():
        pytest.skip("needs the built feature cache")
    # `games_remaining` is the offset the share divides by, and it is attached by
    # the rest-of-season target rather than by the feature layer.
    frame = add_played_rate_target(add_rest_of_season_target(pd.read_pickle(cache)))
    train = frame[frame["season"] < 2024]
    test = frame[frame["season"] == 2024].dropna(subset=[TARGET])
    model = PlayedRate().fit(train)
    fitted = np.abs(model.predict(test) - test[TARGET]).mean()
    raw = np.abs(
        pd.to_numeric(test["prior_play_rate"], errors="coerce").fillna(0.7) - test[TARGET]
    ).mean()
    assert fitted < raw


def test_calibration_reports_a_gap_per_bucket():
    got = calibration(np.linspace(0, 1, 500), np.linspace(0, 1, 500), bins=4)
    assert len(got) == 4
    assert np.allclose(got["gap"], 0.0, atol=1e-9)


def test_the_rate_target_divides_by_games_played_not_games_scheduled():
    """Points per game *played*. A player who sits half the remainder has the
    same rate as one who played all of it, which is the point of the column."""
    from ffmodel.weekly.availability_rate import RATE_TARGET, add_points_per_active_game_target

    frame = _frame([1, 0, 1, 1])
    frame["ros_points"] = [30.0, 20.0, 20.0, 10.0]  # remaining totals from each week
    out = add_points_per_active_game_target(frame)
    # From week 1: 30 points over the 3 games he played. From week 3: 20 over 2.
    assert out[RATE_TARGET].iloc[0] == pytest.approx(10.0)
    assert out[RATE_TARGET].iloc[2] == pytest.approx(10.0)


def test_a_player_who_never_plays_again_has_no_rate_not_a_zero():
    """A zero would teach the fit that a hurt player is a bad player."""
    from ffmodel.weekly.availability_rate import RATE_TARGET, add_points_per_active_game_target

    frame = _frame([1, 0, 0, 0])
    frame["ros_points"] = [5.0, 0.0, 0.0, 0.0]
    out = add_points_per_active_game_target(frame)
    assert out[RATE_TARGET].iloc[1:].isna().all()
