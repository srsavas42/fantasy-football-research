"""Tracking summaries, and the two ways they are absent.

Next Gen Stats publishes only players clearing a volume threshold, so roughly
half the panel has no row -- and unlike a counting stat, none of these columns
has a meaningful zero. Zero separation is a receiver with a defender in his
jersey, not a receiver nobody measured. Every test here is about keeping those
two apart.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import charting as charting_module
from ffmodel.weekly.charting import (
    CHARTING_FAMILIES,
    TRACKED_COLUMNS,
    attach_charting,
    load_charting,
)
from ffmodel.weekly.features import add_features


def _feed(family: str) -> pd.DataFrame:
    columns = CHARTING_FAMILIES[family]
    return pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            # Week 0 is the feed's season-to-date summary, not a week.
            "week": [0, 1, 2],
            "player_gsis_id": ["star", "star", "star"],
            **{c: [99.0, 1.0, 2.0] for c in columns},
        }
    )


def _stub(monkeypatch, feeds: dict | None = None) -> None:
    feeds = feeds if feeds is not None else {f: _feed(f) for f in CHARTING_FAMILIES}

    def _load(seasons, stat_type="passing", **kwargs):
        return feeds.get(stat_type, pd.DataFrame())

    monkeypatch.setattr(charting_module.ingest, "load_nextgen_stats", _load)


def test_the_season_to_date_row_is_not_a_week(monkeypatch) -> None:
    _stub(monkeypatch)
    got = load_charting([2022])
    assert set(got["week"]) == {1, 2}
    assert 99.0 not in got["avg_separation"].to_numpy()


def test_a_tracked_flag_is_set_per_family(monkeypatch) -> None:
    _stub(monkeypatch, {"rushing": _feed("rushing")})
    got = load_charting([2022])
    assert (got["rushing_tracked"] == 1.0).all()
    assert "passing_tracked" not in got.columns


def test_an_untracked_week_stays_missing_and_is_flagged_zero(monkeypatch) -> None:
    """The fill belongs to the design, not to the join: a zero here would be a
    real and extreme value in every one of these units."""
    _stub(monkeypatch, {"receiving": _feed("receiving")})
    panel = pd.DataFrame(
        {
            "season": [2022, 2022],
            "week": [1, 1],
            "player_key": ["star", "nobody"],
            "position": ["WR", "WR"],
        }
    )
    got = attach_charting(panel).set_index("player_key")

    assert got.loc["star", "avg_separation"] == pytest.approx(1.0)
    assert np.isnan(got.loc["nobody", "avg_separation"])
    # The flag, unlike the metric, does have a meaningful zero.
    assert got.loc["star", "receiving_tracked"] == 1.0
    assert got.loc["nobody", "receiving_tracked"] == 0.0


def test_a_missing_feed_leaves_the_panel_untouched(monkeypatch) -> None:
    def _boom(seasons, stat_type="passing", **kwargs):
        raise RuntimeError("network")

    monkeypatch.setattr(charting_module.ingest, "load_nextgen_stats", _boom)
    assert load_charting([2022]).empty

    panel = pd.DataFrame(
        {"season": [2022], "week": [1], "player_key": ["star"], "position": ["WR"]}
    )
    got = attach_charting(panel)
    assert list(got.columns) == list(panel.columns)
    assert len(got) == 1


def test_the_families_merge_without_duplicating_rows(monkeypatch) -> None:
    _stub(monkeypatch)
    got = load_charting([2022])
    assert len(got) == 2
    assert set(TRACKED_COLUMNS) <= set(got.columns)


def _panel(weeks: int = 6) -> pd.DataFrame:
    rows = []
    for week in range(1, weeks + 1):
        rows.append(
            {
                "player_key": "star",
                "player_id": "star",
                "season": 2022,
                "week": week,
                "team": "AAA",
                "opponent": "BBB",
                "position": "WR",
                "played": 1,
                "points": 10.0,
                # Tracked only on odd weeks, which is the shape the real feed has.
                "avg_separation": float(week) if week % 2 else np.nan,
                "receiving_tracked": 1.0 if week % 2 else 0.0,
                "targets": 5.0,
                "rush_att": 0.0,
                "pass_att": 0.0,
                "receptions": 3.0,
                "rush_yds": 0.0,
                "rec_yds": 50.0,
                "rush_epa": 0.0,
                "rec_epa": 0.0,
                "team_targets": 30.0,
                "team_rush_att": 25.0,
                "team_plays": 60.0,
                "team_points": 20.0,
                "team_pass_att": 35.0,
            }
        )
    return pd.DataFrame(rows)


def test_an_untracked_week_does_not_pull_the_average_down() -> None:
    frame = add_features(_panel()).sort_values("week").set_index("week")

    # Weeks 1 and 3 are tracked at 1.0 and 3.0; week 2 is not tracked at all.
    # Week 3's history is week 1 alone, and week 4's is weeks 1 and 3 -- if the
    # untracked week counted as zero, week 4 would sit below 1.0.
    assert frame.loc[3, "prior_avg_separation_recent"] == pytest.approx(1.0)
    assert frame.loc[4, "prior_avg_separation_recent"] > 1.0
    assert np.isnan(frame.loc[1, "prior_avg_separation_recent"])


def test_tracking_history_does_not_see_its_own_week() -> None:
    panel = _panel()
    base = add_features(panel).sort_values("week").reset_index(drop=True)
    perturbed = panel.copy()
    perturbed.loc[perturbed["week"] == 3, "avg_separation"] += 1000.0
    after = add_features(perturbed).sort_values("week").reset_index(drop=True)

    columns = ["prior_avg_separation_recent"]
    upto = base["week"] <= 3
    pd.testing.assert_frame_equal(base.loc[upto, columns], after.loc[upto, columns])
    assert not np.allclose(
        base.loc[base["week"] == 4, columns].to_numpy(),
        after.loc[after["week"] == 4, columns].to_numpy(),
    )


def test_a_panel_without_tracking_still_builds() -> None:
    panel = _panel().drop(columns=["avg_separation", "receiving_tracked"])
    frame = add_features(panel)
    assert len(frame) == len(panel)
    assert "prior_avg_separation_recent" not in frame.columns
