"""Usage by game state: the state flags, the lag, and that nothing about the projected week leaks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import state_usage as su


def _plays(**cols):
    n = len(next(iter(cols.values())))
    base = {
        "wp": [0.5] * n, "score_differential": [0.0] * n, "qtr": [2] * n,
        "half_seconds_remaining": [900.0] * n,
    }
    base.update(cols)
    return pd.DataFrame(base)


def test_a_close_game_with_time_left_is_neutral():
    f = su._flag_states(_plays(wp=[0.5, 0.25, 0.75]))
    assert f["neutral"].tolist() == [True, True, True]


def test_a_lopsided_game_is_not_neutral_and_the_edges_are_inclusive():
    f = su._flag_states(_plays(wp=[0.19, 0.20, 0.80, 0.81]))
    assert f["neutral"].tolist() == [False, True, True, False]


def test_the_last_two_minutes_of_a_half_are_not_neutral():
    f = su._flag_states(_plays(half_seconds_remaining=[130.0, 120.0, 30.0]))
    assert f["neutral"].tolist() == [True, False, False]


def test_overtime_is_not_neutral():
    assert su._flag_states(_plays(qtr=[4, 5]))["neutral"].tolist() == [True, False]


def test_garbage_time_is_a_decided_game_either_way():
    f = su._flag_states(_plays(wp=[0.04, 0.05, 0.50, 0.95, 0.96]))
    assert f["clean"].tolist() == [False, True, True, True, False]


def test_trailing_and_leading_start_at_nine_points():
    f = su._flag_states(_plays(score_differential=[-8, -9, 0, 9, 8]))
    assert f["trailing"].tolist() == [False, True, False, False, False]
    assert f["leading"].tolist() == [False, False, False, True, False]


def test_a_share_needs_a_denominator():
    out = su._share(pd.Series([2.0, 3.0]), pd.Series([10.0, 0.0]))
    assert out.iloc[0] == pytest.approx(0.2)
    assert np.isnan(out.iloc[1])


# --------------------------------------------------------------- the attached features

def _tables(week2_targets=3):
    """Two weeks for one receiver on a club that threw 20 neutral passes a week."""
    players = pd.DataFrame(
        {
            "season": [2025, 2025], "week": [1, 2], "team": ["AAA", "AAA"],
            "player_key": ["p", "p"],
            "st_tgt": [8.0, float(week2_targets)], "st_tgt_neutral": [4.0, float(week2_targets)],
            "st_tgt_clean": [8.0, float(week2_targets)], "st_tgt_trail": [4.0, 0.0],
            "st_rush": [0.0, 0.0], "st_rush_neutral": [0.0, 0.0],
            "st_rush_clean": [0.0, 0.0], "st_rush_lead": [0.0, 0.0],
        }
    )
    teams = pd.DataFrame(
        {
            "season": [2025] * 3, "week": [1, 2, 3], "team": ["AAA"] * 3,
            "st_team_pass": [40.0, 30.0, 35.0], "st_team_pass_neutral": [20.0, 20.0, 18.0],
            "st_team_pass_clean": [32.0, 30.0, 35.0],
            "st_team_rush": [20.0, 22.0, 25.0], "st_team_rush_neutral": [10.0, 12.0, 10.0],
            "st_team_rush_clean": [18.0, 20.0, 25.0], "st_team_garbage_frac": [0.2, 0.0, 0.1],
        }
    )
    return players, teams


def _panel(weeks=3, played=(1, 1, 0)):
    """Receiver ``p`` and a teammate ``q`` who plays every week but is never targeted."""
    mine = list(played[:weeks]) + [1] * max(0, weeks - len(played))
    return pd.DataFrame(
        {
            "player_key": ["p"] * weeks + ["q"] * weeks,
            "season": 2025,
            "week": list(range(1, weeks + 1)) * 2,
            "team": "AAA",
            "played": mine + [1] * weeks,
        }
    )


@pytest.fixture
def patched(monkeypatch):
    def install(week2_targets=3):
        players, teams = _tables(week2_targets)
        monkeypatch.setattr(su, "load_state_usage", lambda seasons: (players, teams))
    return install


def test_the_week_2_feature_is_week_1s_share_and_week_1s_is_empty(patched):
    patched()
    out = su.attach_state_usage(_panel())
    p = out[out["player_key"] == "p"].sort_values("week")
    assert np.isnan(p["prior_neutral_target_share_recent"].iloc[0])
    assert p["prior_neutral_target_share_recent"].iloc[1] == pytest.approx(4 / 20)
    assert p["prior_trailing_target_frac_recent"].iloc[1] == pytest.approx(4 / 8)


def test_a_games_own_numbers_never_reach_its_own_row(patched):
    patched(week2_targets=3)
    a = su.attach_state_usage(_panel())
    patched(week2_targets=19)
    b = su.attach_state_usage(_panel())
    row = lambda d, w: d[(d["player_key"] == "p") & (d["week"] == w)].iloc[0]
    for column in su.STATE_FEATURES:
        x, y = row(a, 2)[column], row(b, 2)[column]
        assert (np.isnan(x) and np.isnan(y)) or x == pytest.approx(y), column
    assert row(a, 3)["prior_neutral_target_share_recent"] != row(b, 3)["prior_neutral_target_share_recent"]


def test_a_player_who_played_without_a_target_has_a_zero_share_not_a_missing_one(patched):
    patched()
    q = su.attach_state_usage(_panel())
    q = q[q["player_key"] == "q"].sort_values("week")
    assert q["prior_neutral_target_share_recent"].iloc[1] == pytest.approx(0.0)


def test_a_week_with_no_play_by_play_inherits_the_clubs_latest_rate(patched):
    patched()
    out = su.attach_state_usage(_panel(weeks=4, played=(1, 1, 1, 0)))
    p = out[out["player_key"] == "p"].sort_values("week")
    rates = p["team_neutral_pass_rate_recent"].to_numpy()
    assert not np.isnan(rates[3])          # week 4 has no play-by-play of its own and still has a value


def test_row_order_and_index_are_those_of_the_panel(patched):
    patched()
    panel = _panel().sample(frac=1.0, random_state=1)
    out = su.attach_state_usage(panel)
    assert out.index.tolist() == panel.index.tolist()
    assert (out["player_key"].to_numpy() == panel["player_key"].to_numpy()).all()


def test_no_play_by_play_gives_empty_features_and_does_not_fail(monkeypatch):
    empty_p = pd.DataFrame(columns=["season", "week", "team", "player_key", *su.PLAYER_COUNTS])
    empty_t = pd.DataFrame(columns=["season", "week", "team", *su.TEAM_COUNTS, "st_team_garbage_frac"])
    monkeypatch.setattr(su, "load_state_usage", lambda seasons: (empty_p, empty_t))
    out = su.attach_state_usage(_panel())
    assert out[list(su.STATE_FEATURES)].isna().all().all()


def test_the_hurdle_reads_the_features_only_when_asked():
    from ffmodel.weekly.nextweek import STATE_FEATURES, Hurdle

    assert set(STATE_FEATURES) == set(su.STATE_FEATURES)
    assert set(STATE_FEATURES) <= set(Hurdle(use_state=True).magnitude_features)
    assert not set(STATE_FEATURES) & set(Hurdle(use_state=False).magnitude_features)
