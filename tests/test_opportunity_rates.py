"""Per-opportunity rates: the ratio, the lag, and that nothing about the projected week leaks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import opportunity_rates as op


def _panel(targets, snaps, *, pass_snaps=None, team_dropbacks=None, played=None):
    n = len(targets)
    played = [1] * n if played is None else played
    df = pd.DataFrame(
        {
            "player_key": "p", "season": 2025, "week": range(1, n + 1), "team": "AAA",
            "position": "WR", "played": played, "targets": targets, "rush_att": 0.0,
            "offense_snaps": snaps, "rec_yds": [10.0 * t for t in targets],
            "points": [float(t) for t in targets],
        }
    )
    return df


@pytest.fixture
def no_participation(monkeypatch):
    empty_p = pd.DataFrame(columns=["season", "week", "team", "player_key", *op.PLAYER_COUNTS])
    empty_t = pd.DataFrame(columns=["season", "week", "team", *op.TEAM_COUNTS])
    monkeypatch.setattr(op, "load_opportunity", lambda seasons: (empty_p, empty_t))


def _with_participation(monkeypatch, pass_snaps, dropbacks):
    n = len(pass_snaps)
    players = pd.DataFrame(
        {"season": 2025, "week": range(1, n + 1), "team": "AAA", "player_key": "p",
         "op_pass_snaps": pass_snaps, "op_run_snaps": [20.0] * n}
    )
    teams = pd.DataFrame(
        {"season": 2025, "week": range(1, n + 1), "team": "AAA",
         "op_team_dropbacks": dropbacks, "op_team_runs": [25.0] * n}
    )
    monkeypatch.setattr(op, "load_opportunity", lambda seasons: (players, teams))


def test_the_first_game_has_no_history_and_the_second_reads_the_first(no_participation):
    out = op.attach_opportunity_rates(_panel([6.0, 8.0, 5.0], [50.0, 60.0, 55.0]))
    assert np.isnan(out["prior_targets_per_snap_recent"].iloc[0])
    assert out["prior_targets_per_snap_recent"].iloc[1] == pytest.approx(6 / 50)


def test_the_rate_is_weighted_by_snaps_not_an_average_of_weekly_rates(no_participation):
    # Week 1: 1 target on 2 snaps (0.5). Week 2: 10 on 100 (0.1). With a long memory the
    # pooled rate is near 11/102, not the mean of 0.5 and 0.1.
    out = op.attach_opportunity_rates(_panel([1.0, 10.0, 0.0], [2.0, 100.0, 50.0]))
    pooled = out["prior_targets_per_snap_level"].iloc[2]
    assert pooled < 0.2
    assert pooled == pytest.approx(11 / 102, abs=0.03)


def test_a_game_of_zero_snaps_leaves_the_rate_alone(no_participation):
    base = op.attach_opportunity_rates(_panel([6.0, 6.0], [50.0, 50.0]))
    gap = op.attach_opportunity_rates(_panel([6.0, 0.0, 6.0], [50.0, 0.0, 50.0], played=[1, 1, 1]))
    assert gap["prior_targets_per_snap_recent"].iloc[2] == pytest.approx(
        base["prior_targets_per_snap_recent"].iloc[1]
    )


def test_a_game_the_player_missed_leaves_the_rate_alone(no_participation):
    out = op.attach_opportunity_rates(_panel([6.0, 0.0, 6.0], [50.0, 0.0, 50.0], played=[1, 0, 1]))
    assert out["prior_targets_per_snap_recent"].iloc[2] == pytest.approx(6 / 50)


def test_a_games_own_numbers_never_reach_its_own_row(no_participation):
    a = op.attach_opportunity_rates(_panel([6.0, 6.0, 6.0], [50.0, 50.0, 50.0]))
    b = op.attach_opportunity_rates(_panel([6.0, 6.0, 40.0], [50.0, 50.0, 60.0]))
    for column in op.RATE_FEATURES:
        x, y = a[column].iloc[2], b[column].iloc[2]
        assert (np.isnan(x) and np.isnan(y)) or x == pytest.approx(y), column


def test_the_rates_are_bounded(no_participation):
    # More targets than snaps is a gap in the counts, and it must not produce a huge rate.
    out = op.attach_opportunity_rates(_panel([30.0, 5.0], [3.0, 50.0]))
    assert out["prior_targets_per_snap_recent"].max() <= 1.0
    assert out["prior_points_per_snap_recent"].max() <= 3.0


def test_pass_snap_share_and_targets_per_pass_snap_come_from_participation(monkeypatch):
    _with_participation(monkeypatch, pass_snaps=[30.0, 36.0], dropbacks=[40.0, 40.0])
    out = op.attach_opportunity_rates(_panel([6.0, 8.0], [50.0, 60.0]))
    assert out["prior_pass_snap_share_recent"].iloc[1] == pytest.approx(30 / 40)
    assert out["prior_targets_per_pass_snap_recent"].iloc[1] == pytest.approx(6 / 30)


def test_without_participation_only_the_simple_group_exists(no_participation):
    out = op.attach_opportunity_rates(_panel([6.0, 8.0, 5.0], [50.0, 60.0, 55.0]))
    assert out["prior_targets_per_snap_recent"].notna().sum() == 2
    assert out["prior_pass_snap_share_recent"].isna().all()
    assert out["prior_targets_per_pass_snap_recent"].isna().all()


def test_row_order_and_index_are_those_of_the_panel(no_participation):
    panel = _panel([6.0, 8.0, 5.0, 7.0], [50.0, 60.0, 55.0, 58.0]).sample(frac=1.0, random_state=3)
    out = op.attach_opportunity_rates(panel)
    assert out.index.tolist() == panel.index.tolist()
    assert (out["week"].to_numpy() == panel["week"].to_numpy()).all()


def test_the_groups_partition_the_features():
    groups = [op.SIMPLE_FEATURES, op.PARTICIPATION_FEATURES, op.EARN_FEATURES, op.YIELD_FEATURES]
    flat = [c for g in groups for c in g]
    assert len(flat) == len(set(flat)) == len(op.RATE_FEATURES)
    assert set(flat) == set(op.RATE_FEATURES)


def test_the_hurdle_reads_the_features_only_when_asked():
    from ffmodel.weekly.nextweek import OPPORTUNITY_FEATURES, Hurdle

    assert set(OPPORTUNITY_FEATURES) == set(op.RATE_FEATURES)
    assert set(OPPORTUNITY_FEATURES) <= set(Hurdle(use_rates=True).magnitude_features)
    assert not set(OPPORTUNITY_FEATURES) & set(Hurdle(use_rates=False).magnitude_features)
