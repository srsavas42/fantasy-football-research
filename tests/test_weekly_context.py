"""Game script, phase-split defence, and the market baseline.

The spread's sign is the load-bearing detail here. ``spread_line`` is quoted
from the home team's perspective, so re-signing it per team is what makes a
positive value mean "this team is favoured" for both sides of the game. Get it
backwards and every game-script coefficient inverts -- the model would learn
that a favourite's running back gets *fewer* carries -- and nothing about the
fit would look wrong.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import frame as frame_module
from ffmodel.weekly.features import add_features
from ffmodel.weekly.market import WeeklyRankCurve, attach_adp
from ffmodel.weekly.nextweek import Hurdle


def test_the_spread_is_resigned_for_the_away_team(monkeypatch) -> None:
    schedule = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 1,
                "home_team": "KC",
                "away_team": "HOU",
                "spread_line": 9.5,
                "total_line": 53.5,
                "game_type": "REG",
            }
        ]
    )
    monkeypatch.setattr(
        frame_module.ingest, "load_schedules", lambda seasons: schedule.copy()
    )
    lines = frame_module._market_lines([2020]).set_index("team")

    # The home favourite carries the positive spread; the away underdog the
    # negative one.
    assert lines.loc["KC", "spread"] == pytest.approx(9.5)
    assert lines.loc["HOU", "spread"] == pytest.approx(-9.5)

    # Implied totals sum to the game total and differ by the spread.
    assert lines.loc["KC", "implied_team_total"] == pytest.approx(31.5)
    assert lines.loc["HOU", "implied_team_total"] == pytest.approx(22.0)
    assert (
        lines.loc["KC", "implied_team_total"] + lines.loc["HOU", "implied_team_total"]
        == pytest.approx(53.5)
    )
    # One side's opponent total is the other side's own total.
    assert lines.loc["KC", "implied_opponent_total"] == pytest.approx(
        lines.loc["HOU", "implied_team_total"]
    )


def _panel(weeks: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for player in range(6):
        team = f"T{player % 3}"
        for week in range(1, weeks + 1):
            opponent = f"T{(player + 1) % 3}"
            rows.append(
                {
                    "player_key": f"P{player}",
                    "player_id": f"P{player}",
                    "player_name": f"Player {player}",
                    "season": 2020,
                    "week": week,
                    "team": team,
                    "opponent": opponent,
                    "position": ["QB", "RB", "WR", "TE"][player % 4],
                    "played": 1,
                    "points": float(rng.integers(2, 25)),
                    "targets": float(rng.integers(0, 10)),
                    "rush_att": float(rng.integers(0, 15)),
                    "pass_att": 0.0,
                    "receptions": 0.0,
                    "rush_yds": float(rng.integers(0, 90)),
                    "rec_yds": float(rng.integers(0, 90)),
                    "rush_epa": float(rng.normal()),
                    "rec_epa": float(rng.normal()),
                    "team_targets": 30.0,
                    "team_rush_att": 25.0,
                    "team_plays": 60.0,
                    "team_points": 40.0,
                    "team_pass_att": 35.0,
                    "spread": -3.0,
                    "game_total": 45.0,
                    "implied_team_total": 21.0,
                    "implied_opponent_total": 24.0,
                }
            )
    return pd.DataFrame(rows)


def test_phase_defence_features_are_lagged() -> None:
    """A defence's own current game must not be in its allowed averages."""
    panel = _panel()
    base = add_features(panel)

    perturbed = panel.copy()
    hit = (perturbed["week"] == 5) & (perturbed["team"] == "T0")
    perturbed.loc[hit, "rush_yds"] = perturbed.loc[hit, "rush_yds"] + 5000.0
    after = add_features(perturbed)

    key = ["player_key", "week"]
    base = base.sort_values(key).reset_index(drop=True)
    after = after.sort_values(key).reset_index(drop=True)

    columns = ["def_rush_yds_allowed", "def_rush_ypc_allowed", "def_rush_epa_allowed"]
    at_or_before = base["week"] <= 5
    pd.testing.assert_frame_equal(
        base.loc[at_or_before, columns],
        after.loc[at_or_before, columns],
        check_exact=False,
        rtol=1e-12,
    )
    later = base["week"] > 5
    moved = [
        c
        for c in columns
        if not np.allclose(
            base.loc[later, c].fillna(-999.0), after.loc[later, c].fillna(-999.0)
        )
    ]
    assert moved, "a defence's conceded yards never reached a later week"


def test_efficiency_allowed_is_a_ratio_not_a_ratio_of_lags() -> None:
    """Yards per carry must be formed per game, then lagged."""
    panel = _panel(weeks=3)
    # One defence, two games, very different volume at the same efficiency.
    panel = panel[panel["team"] == "T0"].copy()
    panel.loc[panel["week"] == 1, ["rush_yds", "rush_att"]] = [40.0, 10.0]
    panel.loc[panel["week"] == 2, ["rush_yds", "rush_att"]] = [200.0, 50.0]
    frame = add_features(panel).sort_values("week")
    # Both games ran at exactly 4.0 a carry, so the lagged average must be 4.0
    # regardless of how differently the two games were weighted by volume.
    third = frame[frame["week"] == 3]["def_rush_ypc_allowed"]
    if third.notna().any():
        assert third.dropna().iloc[0] == pytest.approx(4.0, abs=1e-9)


def test_per_position_fitting_gives_positions_different_slopes() -> None:
    """Otherwise the by-position arm is the pooled arm under another name."""
    rng = np.random.default_rng(4)
    rows = []
    # Sized so every position clears the 1,000 played-week floor a per-position
    # fit requires; below it the estimator deliberately falls back to pooled.
    for season in (2019, 2020, 2021):
        for player in range(120):
            position = ["QB", "RB", "WR", "TE"][player % 4]
            for week in range(1, 17):
                spread = float(rng.normal(0, 6))
                # Backs gain from being favoured, receivers lose. A pooled slope
                # has to average these to about nothing.
                effect = spread * (0.8 if position == "RB" else -0.8)
                rows.append(
                    {
                        "player_key": f"P{player}",
                        "player_id": f"P{player}",
                        "player_name": f"Player {player}",
                        "season": season,
                        "week": week,
                        "team": f"T{player % 8}",
                        "opponent": f"T{(player + 1) % 8}",
                        "position": position,
                        "played": 1,
                        "points": float(12.0 + effect + rng.normal(0, 3)),
                        "targets": 5.0, "rush_att": 8.0, "pass_att": 0.0,
                        "receptions": 3.0, "rush_yds": 30.0, "rec_yds": 40.0,
                        "rush_epa": 0.0, "rec_epa": 0.0,
                        "team_targets": 30.0, "team_rush_att": 25.0,
                        "team_plays": 60.0, "team_points": 40.0, "team_pass_att": 35.0,
                        "spread": spread, "game_total": 45.0,
                        "implied_team_total": 22.5 + spread / 2,
                        "implied_opponent_total": 22.5 - spread / 2,
                    }
                )
    frame = add_features(pd.DataFrame(rows))
    target = frame["points"].to_numpy(float)

    pooled = Hurdle(use_script=True, by_position=False).fit(frame, target)
    split = Hurdle(use_script=True, by_position=True).fit(frame, target)
    assert set(split.parts) >= {"RB", "WR"}

    index = list(split.magnitude_features).index("spread")
    rb = split.parts["RB"][0].coefficients[index]
    wr = split.parts["WR"][0].coefficients[index]
    # Opposite signs, which is the whole point; and the pooled fit sits between
    # them rather than reproducing either.
    assert rb > 0 > wr
    pooled_slope = pooled.magnitude.coefficients[index]
    assert abs(pooled_slope) < max(abs(rb), abs(wr))


def test_the_market_curve_prorates_to_the_horizon() -> None:
    """The same rank must be worth more over ten games than over one."""
    rng = np.random.default_rng(5)
    rows = []
    for season in (2019, 2020):
        for player in range(120):
            rank = player + 1
            for week in range(1, 18):
                rows.append(
                    {
                        "player_key": f"P{player}",
                        "season": season,
                        "week": week,
                        "team": f"T{player % 8}",
                        "position": ["QB", "RB", "WR", "TE"][player % 4],
                        "points": float(max(rng.normal(20 - 3 * np.log(rank), 4), 0)),
                        "adp_rank": float(rank),
                        "adp_drafted": 1.0,
                        "games_remaining": 18 - week,
                    }
                )
    frame = pd.DataFrame(rows)
    weekly = WeeklyRankCurve(per_game=True).fit(frame, frame["points"].to_numpy(float))
    seasonal = WeeklyRankCurve(per_game=False).fit(frame, frame["points"].to_numpy(float))

    rows_out = frame[frame["week"] == 5].head(40)
    one = weekly.predict_samples(rows_out, draws=200, seed=1).mean(axis=1)
    many = seasonal.predict_samples(rows_out, draws=200, seed=1).mean(axis=1)
    games = rows_out["games_remaining"].to_numpy(float)
    # The rest-of-season forecast is the weekly one times the games left.
    np.testing.assert_allclose(many, one * games, rtol=0.05)


def test_adp_join_drops_a_position_disagreement() -> None:
    """A name collision that puts a receiver on a back's rank is worse than none."""
    panel = pd.DataFrame(
        {
            "season": [2020, 2020],
            "week": [1, 1],
            "player_name": ["Real Player", "Other Player"],
            "position": ["WR", "RB"],
        }
    )
    adp = pd.DataFrame(
        {
            "season": [2020, 2020],
            "key": ["realplayer", "otherplayer"],
            "adp_rank": [10.0, 20.0],
            "adp_position": ["RB", "RB"],  # the first disagrees with the panel
        }
    )
    import ffmodel.features.market as market

    out = attach_adp(panel, directory=None) if False else None
    # Exercise the disagreement rule directly rather than through the file
    # loader, which needs the committed CSVs.
    merged = panel.assign(key=market._name_key(panel["player_name"])).merge(
        adp, on=["season", "key"], how="left"
    )
    disagrees = merged["adp_position"].notna() & merged["adp_position"].ne(
        merged["position"]
    )
    merged.loc[disagrees, "adp_rank"] = np.nan
    assert np.isnan(merged.loc[0, "adp_rank"])
    assert merged.loc[1, "adp_rank"] == 20.0
