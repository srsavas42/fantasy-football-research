"""A specialist feature must not know its own week.

Same decisive test the skill panel's feature layer gets: changing an outcome
cannot move any feature at or before the week it happened in, and *must* move one
after -- because a feature layer that ignored history entirely would pass the
first half by doing nothing.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly.specialists import (
    DEFENSE_HISTORY_FEATURES,
    KICKER_HISTORY_FEATURES,
    add_defense_features,
    add_kicker_features,
)


def _kicker_panel(weeks: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for kicker in ("K1", "K2"):
        for week in range(1, weeks + 1):
            played = int(week % 5 != 0)
            rows.append(
                {
                    "player_key": kicker,
                    "player_id": kicker,
                    "player_name": kicker,
                    "season": 2022,
                    "week": week,
                    "team": f"T{kicker}",
                    "position": "K",
                    "played": played,
                    "points": float(rng.integers(0, 16)) if played else 0.0,
                    "fg_att": float(rng.integers(0, 5)),
                    "fg_made": float(rng.integers(0, 4)),
                    "fg_long": float(rng.integers(20, 56)),
                    "pat_att": float(rng.integers(0, 6)),
                    "pat_made": float(rng.integers(0, 6)),
                    "fg_made_40_49": float(rng.integers(0, 2)),
                    "fg_att_50_plus": float(rng.integers(0, 2)),
                }
            )
    return pd.DataFrame(rows)


def _defense_panel(weeks: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    clubs = ("D1", "D2")
    for club in clubs:
        for week in range(1, weeks + 1):
            rows.append(
                {
                    "player_key": club,
                    "player_id": club,
                    "player_name": club,
                    "season": 2022,
                    "week": week,
                    "team": club,
                    "opponent": clubs[1] if club == clubs[0] else clubs[0],
                    "position": "DST",
                    "played": 1,
                    "points": float(rng.integers(-4, 20)),
                    "def_sacks": float(rng.integers(0, 7)),
                    "def_interceptions": float(rng.integers(0, 3)),
                    "fumble_recovery_opp": float(rng.integers(0, 3)),
                    "def_tds": 0.0,
                    "points_allowed": float(rng.integers(3, 40)),
                    "yards_allowed": float(rng.integers(150, 500)),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("builder", "panel_factory", "columns"),
    [
        (add_kicker_features, _kicker_panel, KICKER_HISTORY_FEATURES),
        (add_defense_features, _defense_panel, DEFENSE_HISTORY_FEATURES),
    ],
    ids=["kicker", "defense"],
)
def test_changing_a_week_cannot_move_a_feature_at_or_before_it(
    builder, panel_factory, columns
):
    panel = panel_factory()
    changed = panel.copy()
    target = (changed["player_key"] == changed["player_key"].iloc[0]) & (
        changed["week"] == 5
    )
    changed.loc[target, "points"] = changed.loc[target, "points"] + 100.0

    before = builder(panel).sort_values(["player_key", "week"]).reset_index(drop=True)
    after = builder(changed).sort_values(["player_key", "week"]).reset_index(drop=True)

    present = [c for c in columns if c in before.columns]
    same_player = before["player_key"] == before["player_key"].iloc[0]

    # Nothing at or before week 5 may move.
    upto = same_player & (before["week"] <= 5)
    pd.testing.assert_frame_equal(
        before.loc[upto, present], after.loc[upto, present], check_exact=False
    )

    # And something after it must, or the features are not reading history.
    later = same_player & (before["week"] > 5)
    moved = (
        (before.loc[later, present] - after.loc[later, present]).abs().to_numpy()
    )
    assert np.nanmax(moved) > 1e-9


def test_the_relevant_population_columns_are_present_on_both_panels():
    """``walk_forward`` reads these, so a rename upstream must fail loudly here."""
    for builder, factory in (
        (add_kicker_features, _kicker_panel),
        (add_defense_features, _defense_panel),
    ):
        out = builder(factory())
        assert "prior_points_recent_given_played" in out.columns
        assert "prior_games" in out.columns


def test_a_kickers_given_played_level_ignores_the_weeks_he_did_not_kick():
    """Otherwise a run of inactive weeks reads as a run of bad ones."""
    panel = _kicker_panel()
    out = add_kicker_features(panel).sort_values(["player_key", "week"])
    block = out[out["player_key"] == "K1"]
    # Week 5 is an inactive week by construction; the level entering week 6 must
    # not have been dragged toward its zero.
    level = block.set_index("week")["prior_points_recent_given_played"]
    played_points = block[block["played"] == 1].set_index("week")["points"]
    assert level.loc[6] >= played_points.loc[:5].min()


def test_the_league_baseline_is_lagged_and_pooled():
    """It must not contain the week it describes, or it reads the answer."""
    from ffmodel.weekly.specialists import add_league_baseline

    panel = _kicker_panel()
    out = add_league_baseline(panel).sort_values(["season", "week"])
    # The very first week has no prior league history at all.
    assert pd.isna(out[out["week"] == 1]["league_points_recent"]).all()

    # Blowing up one week cannot move the baseline at or before it.
    changed = panel.copy()
    hit = changed["week"] == 4
    changed.loc[hit, "points"] = changed.loc[hit, "points"] + 500.0
    before = add_league_baseline(panel).sort_values(["season", "week"]).reset_index(drop=True)
    after = add_league_baseline(changed).sort_values(["season", "week"]).reset_index(drop=True)
    upto = before["week"] <= 4
    pd.testing.assert_series_equal(
        before.loc[upto, "league_points_recent"],
        after.loc[upto, "league_points_recent"],
    )
    # And must move it afterwards, or the column is not reading the league.
    later = before["week"] > 4
    assert (
        before.loc[later, "league_points_recent"]
        - after.loc[later, "league_points_recent"]
    ).abs().max() > 1e-9


def test_relocated_franchises_are_not_dropped_by_code_mismatch(monkeypatch):
    """nflverse labels relocations inconsistently across its own feeds.

    `team_stats` uses the modern code in every season (2016 Raiders are `LV`)
    while `schedules` uses the code in use at the time (`OAK`). An inner merge
    on the club code therefore drops the franchise from exactly the seasons
    before it moved -- silently, and only for a handful of teams, which is the
    hardest kind of gap to notice. This pins the translation that prevents it.
    """
    from ffmodel.weekly import specialists

    fake_teams = pd.DataFrame(
        {
            "team_abbr": ["OAK", "LV", "SF"],
            "team_id": ["2520", "2520", "4500"],
            "team_name": ["Oakland Raiders", "Las Vegas Raiders", "SF 49ers"],
        }
    )

    class _Frame:
        def to_pandas(self):
            return fake_teams

    monkeypatch.setitem(
        __import__("sys").modules, "nflreadpy", type("M", (), {"load_teams": staticmethod(lambda: _Frame())})
    )
    monkeypatch.setattr(
        specialists.ingest,
        "load_schedules",
        lambda seasons: pd.DataFrame(
            {
                "season": [2016, 2016],
                "week": [1, 1],
                "game_type": ["REG", "REG"],
                "home_team": ["OAK", "SF"],
                "away_team": ["SF", "OAK"],
            }
        ),
    )

    codes = specialists._canonical_team_codes([2016])
    # The modern code must translate to the one 2016 actually used...
    assert codes[(2016, "LV")] == "OAK"
    # ...and a code already correct for that season must survive unchanged.
    assert codes[(2016, "OAK")] == "OAK"
    assert codes[(2016, "SF")] == "SF"
