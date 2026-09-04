"""The weather encoding, where 'missing' and 'controlled' must not be confused.

A dome has no temperature reading because the building supplies one, and an
outdoor game with no reading has one nobody wrote down. Filling both the same way
is the mistake this module exists to avoid: it would teach the fit that 29% of
the league plays in whatever the median outdoor Sunday happened to be.
"""

import pandas as pd
import pytest

from ffmodel.weekly.weather import (
    FREEZING,
    INDOOR_TEMP,
    INDOOR_WIND,
    WEATHER_COLUMNS,
    WIND_HIGH,
    attach_weather,
    load_game_conditions,
)


@pytest.fixture
def schedule(monkeypatch):
    """A four-game week covering every roof category."""
    frame = pd.DataFrame(
        {
            "game_id": ["1", "2", "3", "4"],
            "season": [2022] * 4,
            "week": [1] * 4,
            "game_type": ["REG"] * 4,
            "home_team": ["OUT", "DOM", "CLO", "GAP"],
            "away_team": ["out", "dom", "clo", "gap"],
            "roof": ["outdoors", "dome", "closed", "outdoors"],
            "temp": [28.0, None, None, None],
            "wind": [22.0, None, None, None],
        }
    )
    from ffmodel.weekly import weather

    monkeypatch.setattr(weather.ingest, "load_schedules", lambda seasons: frame)
    return frame


def test_both_clubs_in_a_game_face_the_same_conditions(schedule):
    conditions = load_game_conditions([2022]).set_index("team")
    assert conditions.loc["OUT", "wx_temp"] == conditions.loc["out", "wx_temp"] == 28.0
    assert len(conditions) == 8


def test_an_indoor_game_is_controlled_not_missing(schedule):
    """A dome carries the building's climate and is never flagged missing."""
    conditions = load_game_conditions([2022]).set_index("team")
    for team in ("DOM", "CLO"):
        assert conditions.loc[team, "roof_indoor"] == 1.0
        assert conditions.loc[team, "wx_temp"] == INDOOR_TEMP
        assert conditions.loc[team, "wx_wind"] == INDOOR_WIND
        assert conditions.loc[team, "wx_missing"] == 0.0


def test_an_outdoor_game_with_no_reading_is_missing_not_mild(schedule):
    """The gap stays a gap, so the design fills it and flags that it did."""
    conditions = load_game_conditions([2022]).set_index("team")
    assert conditions.loc["GAP", "roof_indoor"] == 0.0
    assert pd.isna(conditions.loc["GAP", "wx_temp"])
    assert pd.isna(conditions.loc["GAP", "wx_wind"])
    assert conditions.loc["GAP", "wx_missing"] == 1.0


def test_a_threshold_on_an_unknown_reading_is_unknown(schedule):
    """Not false. An unrecorded game is not thereby a calm, mild one."""
    conditions = load_game_conditions([2022]).set_index("team")
    assert pd.isna(conditions.loc["GAP", "wx_wind_high"])
    assert pd.isna(conditions.loc["GAP", "wx_freezing"])


def test_the_thresholds_fire_on_the_conditions_they_name(schedule):
    conditions = load_game_conditions([2022]).set_index("team")
    assert conditions.loc["OUT", "wx_wind"] >= WIND_HIGH
    assert conditions.loc["OUT", "wx_wind_high"] == 1.0
    assert conditions.loc["OUT", "wx_temp"] <= FREEZING
    assert conditions.loc["OUT", "wx_freezing"] == 1.0
    # Indoors clears both by construction.
    assert conditions.loc["DOM", "wx_wind_high"] == 0.0
    assert conditions.loc["DOM", "wx_freezing"] == 0.0


def test_a_retractable_roof_left_open_is_an_outdoor_game(monkeypatch):
    frame = pd.DataFrame(
        {
            "season": [2022],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["OPN"],
            "away_team": ["opn"],
            "roof": ["open"],
            "temp": [55.0],
            "wind": [4.0],
        }
    )
    from ffmodel.weekly import weather

    monkeypatch.setattr(weather.ingest, "load_schedules", lambda seasons: frame)
    conditions = load_game_conditions([2022]).set_index("team")
    assert conditions.loc["OPN", "roof_indoor"] == 0.0
    assert conditions.loc["OPN", "wx_temp"] == 55.0


def test_attach_leaves_every_column_present_when_the_feed_is_unavailable(monkeypatch):
    """A model asking for these columns must not crash offline."""
    from ffmodel.weekly import weather

    def boom(seasons):
        raise RuntimeError("no network")

    monkeypatch.setattr(weather.ingest, "load_schedules", boom)
    panel = pd.DataFrame({"season": [2022], "week": [1], "team": ["OUT"]})
    out = attach_weather(panel)
    for column in WEATHER_COLUMNS:
        assert column in out.columns
        assert out[column].isna().all()
