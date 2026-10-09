"""Games cut short: the label, the lag, and that nothing about the projected game leaks in."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import ros_reconciled as R
from ffmodel.weekly.nextweek import Hurdle
from ffmodel.weekly.partial_game import (
    MIN_BASELINE,
    THRESHOLD,
    add_partial_game,
)


def _player(snaps, *, played=None, key="p1", season=2025, position="WR"):
    n = len(snaps)
    played = [1 if s is not None else 0 for s in snaps] if played is None else played
    return pd.DataFrame(
        {
            "player_key": key,
            "season": season,
            "week": range(1, n + 1),
            "position": position,
            "played": played,
            "snap_share": [np.nan if s is None else s for s in snaps],
        }
    )


def test_a_game_far_below_his_recent_snaps_is_flagged():
    out = add_partial_game(_player([0.80, 0.75, 0.85, 0.30]))
    assert out["left_early"].tolist() == [0, 0, 0, 1]


def test_an_ordinary_dip_is_not():
    # 0.55 of a 0.80 baseline is a quiet week, not an exit: above THRESHOLD.
    assert 0.55 / 0.80 > THRESHOLD
    assert add_partial_game(_player([0.80, 0.80, 0.80, 0.55]))["left_early"].iloc[-1] == 0


def test_a_part_time_player_has_no_role_to_cut_short():
    assert 0.40 < MIN_BASELINE
    out = add_partial_game(_player([0.40, 0.40, 0.40, 0.10]))
    assert out["left_early"].sum() == 0


def test_the_first_games_have_no_baseline_and_are_never_flagged():
    out = add_partial_game(_player([0.10, 0.80, 0.80]))
    assert out["left_early"].tolist() == [0, 0, 0]


def test_a_missed_game_is_not_a_game_cut_short():
    out = add_partial_game(_player([0.80, 0.80, 0.80, None]))
    assert out["left_early"].iloc[-1] == 0


def test_the_flag_reaches_the_next_row_and_not_its_own():
    out = add_partial_game(_player([0.80, 0.80, 0.80, 0.20, 0.70]))
    assert out["partial_prev"].tolist() == [0, 0, 0, 0, 1]
    assert out["partial_recent"].tolist()[-1] == 1


def test_repeat_exits_are_counted():
    out = add_partial_game(_player([0.80, 0.80, 0.80, 0.20, 0.80, 0.20, 0.70]))
    assert out["partial_recent"].iloc[-1] == 2


def test_a_later_game_never_changes_an_earlier_row():
    base = _player([0.80, 0.80, 0.80, 0.20, 0.70])
    cut = add_partial_game(base.iloc[:4])
    full = add_partial_game(base)
    cols = ["left_early", "partial_prev", "partial_recent"]
    assert cut[cols].to_numpy().tolist() == full[cols].iloc[:4].to_numpy().tolist()


def test_the_projected_row_reads_the_game_before_it():
    # A live row has no stat line and no snaps; it must see last game's exit.
    hist = _player([0.80, 0.80, 0.80, 0.20])
    live = pd.DataFrame(
        {"player_key": ["p1"], "season": [2025], "week": [5], "position": ["WR"], "played": [0], "snap_share": [np.nan]}
    )
    out = add_partial_game(pd.concat([hist, live], ignore_index=True))
    assert out["partial_prev"].iloc[-1] == 1
    assert out["left_early"].iloc[-1] == 0


def test_players_and_seasons_do_not_bleed_into_each_other():
    a = _player([0.80, 0.80, 0.80, 0.20], key="a")
    b = _player([0.80, 0.80, 0.80, 0.80], key="b")
    c = _player([0.80, 0.80, 0.80, 0.80], key="a", season=2026)
    out = add_partial_game(pd.concat([a, b, c], ignore_index=True))
    assert out.loc[out["player_key"] == "b", "partial_prev"].sum() == 0
    assert out.loc[out["season"] == 2026, "partial_prev"].sum() == 0


def test_the_original_index_and_order_survive():
    frame = pd.concat([_player([0.8, 0.8, 0.8, 0.2], key="z"), _player([0.8] * 4, key="a")], ignore_index=True)
    frame.index = frame.index * 10
    out = add_partial_game(frame)
    assert out.index.tolist() == frame.index.tolist()
    assert out.loc[30, "left_early"] == 1


def test_a_frame_without_snaps_gets_zeros_rather_than_failing():
    frame = _player([0.8, 0.8]).drop(columns="snap_share")
    out = add_partial_game(frame)
    assert out[["left_early", "partial_prev", "partial_recent"]].to_numpy().sum() == 0


# ------------------------------------------------------------- the model plumbing

def test_absence_state_carries_the_flag():
    frame = _player([0.8, 0.8, 0.8, 0.2, 0.7])
    frame["status"] = "ACT"
    frame["weeks_since_played"] = 1
    out = R.add_absence_state(frame)
    assert out["partial_prev"].iloc[-1] == 1


def test_a_cut_short_player_gets_a_group_of_his_own(monkeypatch):
    monkeypatch.setattr(R, "PARTIAL_GROUP", True)
    frame = _player([0.8, 0.8, 0.8, 0.2, 0.7])
    frame["status"] = "ACT"
    frame["weeks_since_played"] = 1
    groups = R._absence_groups(R.add_absence_state(frame))
    assert groups[-1] == 6
    assert set(groups[:-1]) == {0}


def test_the_group_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(R, "PARTIAL_GROUP", False)
    frame = _player([0.8, 0.8, 0.8, 0.2, 0.7])
    frame["status"] = "ACT"
    frame["weeks_since_played"] = 1
    assert 6 not in set(R._absence_groups(R.add_absence_state(frame)))


def test_the_weekly_hurdle_reads_the_flag_for_availability_only():
    on, off = Hurdle(use_partial=True), Hurdle(use_partial=False)
    assert "partial_prev" in on.availability_features
    assert "partial_prev" not in on.magnitude_features
    assert "partial_prev" not in off.availability_features


def test_the_games_model_uses_the_flag():
    frame = _player([0.8, 0.8, 0.8, 0.2, 0.7])
    frame["status"] = "ACT"
    frame["weeks_since_played"] = 1
    frame = R.add_absence_state(frame)
    frame["missed_last"] = 0.0
    design, _ = R._games_design(frame)
    assert {"partial_prev", "partial_recent"} <= set(design.columns)


# ------------------------------------------------------ weeks 1-2: the depth-chart baseline

def _early_frame(*, last_snap=0.80, first_snap=0.40, history=0.85, rookie=False, depth=1.0):
    """A 2024 WR1 league to learn from, and one 2025 player whose opener we judge."""
    hist = []
    for i in range(40):
        hist.append(
            pd.DataFrame(
                {
                    "player_key": f"h{i}", "season": 2024, "week": range(1, 9), "position": "WR",
                    "played": 1, "snap_share": history, "depth_rank": 1.0,
                }
            )
        )
    snaps = [first_snap, 0.80, 0.80]
    mine = pd.DataFrame(
        {"player_key": "x", "season": 2025, "week": [1, 2, 3], "position": "WR",
         "played": 1, "snap_share": snaps, "depth_rank": depth}
    )
    parts = hist + [mine]
    if not rookie:
        prior = pd.DataFrame(
            {"player_key": "x", "season": 2024, "week": range(1, 9), "position": "WR",
             "played": 1, "snap_share": last_snap, "depth_rank": 1.0}
        )
        parts.append(prior)
    return pd.concat(parts, ignore_index=True)


def _row(frame, week):
    return frame[(frame["player_key"] == "x") & (frame["season"] == 2025) & (frame["week"] == week)].iloc[0]


def test_a_wr1_on_half_his_usual_snaps_in_week_one_is_flagged():
    out = add_partial_game(_early_frame(first_snap=0.40))
    assert _row(out, 1)["left_early_depth"] == 1
    assert _row(out, 1)["left_early"] == 1


def test_it_reaches_week_two_through_its_own_column_and_not_the_in_season_one():
    out = add_partial_game(_early_frame(first_snap=0.40))
    assert _row(out, 2)["partial_prev_early"] == 1
    assert _row(out, 2)["partial_prev"] == 0
    assert _row(out, 2)["partial_recent"] == 0     # counts the in-season label only


def test_a_normal_week_one_is_not_flagged():
    out = add_partial_game(_early_frame(first_snap=0.78))
    assert _row(out, 1)["left_early_depth"] == 0


def test_the_in_season_baseline_takes_over_once_it_exists():
    # By week 3 he has two games of his own; the depth rule is no longer consulted.
    out = add_partial_game(_early_frame(first_snap=0.40))
    assert _row(out, 3)["left_early_depth"] == 0


def test_a_rookie_is_judged_on_the_depth_chart_alone_and_more_strictly():
    # No last season: one source, so the threshold is the stricter 0.5 x 0.85 = 0.425.
    assert add_partial_game(_early_frame(first_snap=0.40, rookie=True)).pipe(_row, 1)["left_early_depth"] == 1
    assert add_partial_game(_early_frame(first_snap=0.50, rookie=True)).pipe(_row, 1)["left_early_depth"] == 0


def test_a_player_who_is_a_part_timer_by_both_measures_is_not_flagged():
    out = add_partial_game(_early_frame(first_snap=0.10, last_snap=0.30, history=0.35))
    assert _row(out, 1)["left_early_depth"] == 0


def test_earlier_seasons_set_the_expectation_and_the_same_season_does_not():
    base = _early_frame(first_snap=0.40)
    later = base.copy()
    # Rewriting everyone's 2025 snaps (the season being judged) must change nothing...
    later.loc[later["season"] == 2025, "snap_share"] = later.loc[later["season"] == 2025, "snap_share"].where(
        later["player_key"] == "x", 0.99
    )
    a, b = add_partial_game(base), add_partial_game(later)
    assert _row(a, 1)["left_early_depth"] == _row(b, 1)["left_early_depth"]
    # ... while rewriting 2024, the season it learns from, does.
    lower = base.copy()
    lower.loc[(lower["season"] == 2024) & (lower["player_key"] != "x"), "snap_share"] = 0.40
    lower.loc[(lower["season"] == 2024) & (lower["player_key"] == "x"), "snap_share"] = 0.40
    assert _row(add_partial_game(lower), 1)["left_early_depth"] == 0


def test_the_first_season_has_nothing_to_learn_from():
    out = add_partial_game(_early_frame(first_snap=0.40)[lambda d: d["season"] == 2025])
    assert out["left_early_depth"].sum() == 0


def test_a_frame_with_no_depth_chart_still_works(monkeypatch):
    frame = _early_frame(first_snap=0.40).drop(columns="depth_rank")
    out = add_partial_game(frame)
    # last season's own average is still a baseline of one source
    assert _row(out, 1)["left_early_depth"] in (0, 1)


def test_the_early_rule_can_be_switched_off(monkeypatch):
    from ffmodel.weekly import partial_game

    monkeypatch.setattr(partial_game, "EARLY_BASELINE", False)
    out = add_partial_game(_early_frame(first_snap=0.40))
    assert out["left_early_depth"].sum() == 0


def test_the_hurdle_gets_the_early_flag_and_the_games_model_does_not():
    assert "partial_prev_early" in Hurdle(use_partial=True).availability_features
    assert "partial_prev_early" not in R.PARTIAL_FEATURES
