"""Play-calling tendency, and the plays that must not count as play-calling.

The whole value of pass rate over expected is that it divides out the game
state. A kneel-down at the end of a blowout is not a run-first identity and a
spike is not a pass-first one; the expected-pass model declines to price both,
and counting them would put the score back into the column built to remove it.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import tendency as tendency_module
from ffmodel.weekly.features import add_features
from ffmodel.weekly.tendency import attach_tendency, load_team_tendency


def _plays() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022] * 6,
            "week": [1] * 6,
            "posteam": ["AAA", "AAA", "AAA", "BBB", "BBB", None],
            "xpass": [0.5, 0.7, np.nan, 0.4, 0.6, 0.5],
            "pass_oe": [10.0, 20.0, -99.0, -5.0, 5.0, 50.0],
        }
    )


def _stub(monkeypatch, plays: pd.DataFrame) -> None:
    monkeypatch.setattr(tendency_module.ingest, "load_pbp", lambda s: plays)


def test_unpriced_plays_do_not_become_tendency(monkeypatch) -> None:
    _stub(monkeypatch, _plays())
    got = load_team_tendency([2022]).set_index("team")

    # AAA's third play has no expected pass rate, so its -99 must not land in
    # the mean; the two priced plays average to 15.
    assert got.loc["AAA", "proe"] == pytest.approx(15.0)
    assert got.loc["BBB", "proe"] == pytest.approx(0.0)


def test_a_play_with_no_offence_belongs_to_no_team(monkeypatch) -> None:
    _stub(monkeypatch, _plays())
    got = load_team_tendency([2022])
    assert set(got["team"]) == {"AAA", "BBB"}
    assert len(got) == 2


def test_the_situations_are_kept_apart_from_the_coach(monkeypatch) -> None:
    """xpass is what the down-and-distance asked for; proe is the deviation."""
    _stub(monkeypatch, _plays())
    got = load_team_tendency([2022]).set_index("team")
    assert got.loc["AAA", "xpass"] == pytest.approx(0.6)
    assert got.loc["BBB", "xpass"] == pytest.approx(0.5)


def test_a_missing_feed_leaves_the_columns_empty_not_neutral(monkeypatch) -> None:
    """Unknown tendency is not average tendency, and a zero here would say it is."""

    def _boom(seasons):
        raise RuntimeError("network")

    monkeypatch.setattr(tendency_module.ingest, "load_pbp", _boom)
    assert load_team_tendency([2022]).empty

    panel = pd.DataFrame(
        {"season": [2022], "week": [1], "team": ["AAA"], "player_key": ["p"]}
    )
    got = attach_tendency(panel)
    assert len(got) == 1
    assert got["proe"].isna().all()
    assert got["xpass"].isna().all()


def test_every_player_on_a_roster_shares_the_team_week(monkeypatch) -> None:
    _stub(monkeypatch, _plays())
    panel = pd.DataFrame(
        {
            "season": [2022] * 3,
            "week": [1] * 3,
            "team": ["AAA", "AAA", "BBB"],
            "player_key": ["a1", "a2", "b1"],
        }
    )
    got = attach_tendency(panel)
    assert len(got) == 3
    assert got.loc[got["team"] == "AAA", "proe"].nunique() == 1
    assert got.loc[got["team"] == "AAA", "proe"].iloc[0] == pytest.approx(15.0)


def _panel(weeks: int = 6) -> pd.DataFrame:
    """Two teams playing each other every week, with tendency attached."""
    rows = []
    for week in range(1, weeks + 1):
        for team, opponent, proe in (("AAA", "BBB", 5.0 * week), ("BBB", "AAA", -2.0 * week)):
            rows.append(
                {
                    "player_key": f"P{team}",
                    "player_id": f"P{team}",
                    "season": 2022,
                    "week": week,
                    "team": team,
                    "opponent": opponent,
                    "position": "WR",
                    "played": 1,
                    "points": 10.0,
                    "proe": proe,
                    "xpass": 0.6,
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


def test_tendency_history_is_lagged() -> None:
    frame = add_features(_panel())
    aaa = frame[frame["team"] == "AAA"].sort_values("week").set_index("week")

    assert np.isnan(aaa.loc[1, "team_proe_recent"])
    # A one-game team decay is not assumed here; whatever the decay, week 2 can
    # only see week 1, so it must equal it exactly.
    assert aaa.loc[2, "team_proe_recent"] == pytest.approx(5.0)


def test_the_defence_reads_what_it_faced_not_what_it_called() -> None:
    frame = add_features(_panel())
    aaa = frame[frame["team"] == "AAA"].sort_values("week").set_index("week")

    # AAA plays BBB every week, so what AAA's opponent-defence has faced is
    # AAA's own tendency -- and what AAA faces from BBB's defence is BBB's.
    assert aaa.loc[2, "def_proe_faced_recent"] == pytest.approx(5.0)
    assert aaa.loc[2, "team_proe_recent"] != aaa.loc[2, "def_xpass_faced_recent"]


def test_tendency_does_not_see_its_own_week() -> None:
    panel = _panel()
    base = add_features(panel).sort_values(["team", "week"]).reset_index(drop=True)
    perturbed = panel.copy()
    perturbed.loc[perturbed["week"] == 4, "proe"] += 1000.0
    after = add_features(perturbed).sort_values(["team", "week"]).reset_index(drop=True)

    columns = ["team_proe_recent", "def_proe_faced_recent"]
    upto = base["week"] <= 4
    pd.testing.assert_frame_equal(base.loc[upto, columns], after.loc[upto, columns])
    assert not np.allclose(
        base.loc[base["week"] == 5, columns].to_numpy(),
        after.loc[after["week"] == 5, columns].to_numpy(),
    )


def test_a_panel_without_tendency_still_builds() -> None:
    """The columns are optional: the feature layer must not require the feed."""
    panel = _panel().drop(columns=["proe", "xpass"])
    frame = add_features(panel)
    assert len(frame) == len(panel)
    assert "team_proe_recent" not in frame.columns
