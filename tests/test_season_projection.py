"""Forward projection onto a season that has not been played.

The season-average pipeline is otherwise backtest-shaped: every season it scores
already has play-by-play. These cover the projection path, whose defining risk is
that an absent outcome is silently presented to the model as a realized zero.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.features.season_average import (
    PRESEASON_FEATURES,
    PROJECTION_BLANK_LABELS,
    build_projection_data,
    preseason_roster_snapshot,
    team_transition_rows,
)


def _team_volume(seasons, teams=("BAL", "CIN")) -> pd.DataFrame:
    rate_columns = {
        "opportunity_plays_per_game": 60.0,
        "plays_per_game": 63.0,
        "pass_rate": 0.58,
        "sack_rate": 0.06,
        "target_rate": 0.61,
        "pass_attempts_per_game": 34.0,
        "sacks_per_game": 2.2,
        "dropbacks_per_game": 36.0,
        "rush_attempts_per_game": 26.0,
        "targets_per_game": 34.0,
    }
    rows = []
    for season in seasons:
        for team in teams:
            rows.append({"season": season, "team": team, "games": 17, **rate_columns})
    return pd.DataFrame(rows)


def test_team_transition_rows_adds_projection_season_from_prior_rates():
    volume = _team_volume([2023, 2024])

    rows = team_transition_rows(volume, projection_seasons=[2025])

    projected = rows[rows["season"].eq(2025)]
    assert len(projected) == 2, "each team needs a projection-season row"
    # Prior-season rates carry forward...
    assert projected["prior_pass_rate"].eq(0.58).all()
    # ...while nothing realized is invented for a season that has not happened.
    assert projected["pass_rate"].isna().all()
    assert projected["games"].isna().all()


def test_team_transition_rows_unchanged_without_projection_seasons():
    volume = _team_volume([2023, 2024])

    assert team_transition_rows(volume).equals(
        team_transition_rows(volume, projection_seasons=[])
    )


def _roster_snapshot(season: int) -> pd.DataFrame:
    rosters = pd.DataFrame(
        {
            "season": season,
            "team": ["BAL", "BAL", "BAL", "CIN", "CIN", "CIN"],
            "position": ["QB", "RB", "WR", "QB", "WR", "TE"],
            "week": 1,
            "player_name": [
                "Lamar Jackson",
                "Rookie Back",
                "Marquise Brown",
                "Joe Burrow",
                "Ja'Marr Chase",
                "Rookie End",
            ],
            "player_id": ["qb-bal", "rb-new", "wr-bal", "qb-cin", "wr-cin", "te-new"],
            "status": "ACT",
            "years_exp": [4, 0, 2, 2, 0, 0],
        }
    )
    return preseason_roster_snapshot(rosters, None, cutoff_week=1)


def test_projection_rows_carry_no_realized_outcomes():
    snapshot = _roster_snapshot(2021)

    data = build_projection_data(
        2021,
        roster_snapshot=snapshot,
        history_seasons=[2018, 2019, 2020],
        source="legacy",
    )
    rows = data.player_rows

    assert sorted(rows["season"].unique()) == [2021]
    assert rows["is_projection"].eq(1).all()
    # The label merge zero-fills absent rows; on an unplayed season that would
    # read as a realized zero rather than an absent measurement.
    for column in PROJECTION_BLANK_LABELS:
        if column in rows:
            assert rows[column].isna().all(), f"{column} must stay missing"
    # Predictors still have to be there for the frame to be usable.
    assert set(PRESEASON_FEATURES) <= set(rows.columns)


def test_projection_exposure_is_the_scheduled_slate():
    data = build_projection_data(
        2021,
        roster_snapshot=_roster_snapshot(2021),
        history_seasons=[2018, 2019, 2020],
        source="legacy",
    )

    games = data.player_rows["games"].dropna().unique()
    assert len(games) == 1
    # Read as an integer exposure at predict time, so it must be finite and whole.
    assert np.isfinite(games[0]) and float(games[0]).is_integer()


def test_projection_team_rows_include_the_projected_season():
    data = build_projection_data(
        2021,
        roster_snapshot=_roster_snapshot(2021),
        history_seasons=[2018, 2019, 2020],
        source="legacy",
    )

    assert 2021 in set(data.team_rows["season"])


def test_projection_without_roster_snapshot_is_refused():
    # No published week-1 roster exists before the season starts, so the caller
    # has to supply one rather than get an empty or inferred frame.
    with pytest.raises(ValueError, match="roster_snapshot"):
        build_projection_data(2021, roster_snapshot=pd.DataFrame({"season": [2020]}))


def test_projection_season_must_follow_the_history():
    with pytest.raises(ValueError, match="history must run up to"):
        build_projection_data(
            2021,
            roster_snapshot=_roster_snapshot(2021),
            history_seasons=[2017, 2018],
            source="legacy",
        )


def test_projection_season_cannot_also_be_history():
    with pytest.raises(ValueError, match="must not appear"):
        build_projection_data(
            2021,
            roster_snapshot=_roster_snapshot(2021),
            history_seasons=[2019, 2020, 2021],
            source="legacy",
        )
