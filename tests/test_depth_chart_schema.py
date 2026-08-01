"""Depth-chart schema conformance across the 2025 nflverse feed change.

Through 2024 the feed published weekly rows keyed by ``season``/``week``. From
2025 it publishes timestamped snapshots keyed by ``dt``, with renamed team, name,
position, and depth columns. Both shapes have to reach feature code as one
schema, and a snapshot has to land on the week it actually informs — a wrong
week silently changes which chart a point-in-time cutoff selects.
"""

import pandas as pd
import pytest

from ffmodel.data import ingest


def _schedule(season: int = 2026) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": season,
            "game_type": "REG",
            "week": [1, 2, 3],
            "gameday": ["2026-09-09", "2026-09-16", "2026-09-23"],
        }
    )


def _snapshot_frame(stamps) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dt": stamps,
            "team": "BAL",
            "player_name": "Lamar Jackson",
            "gsis_id": "qb-bal",
            "pos_abb": "QB",
            "pos_grp": "3WR 1TE",
            "pos_rank": 1,
            "pos_slot": 9,
        }
    )


@pytest.fixture
def offline_schedule(monkeypatch):
    monkeypatch.setattr(
        ingest, "load_schedules", lambda *a, **k: _schedule(), raising=True
    )


def test_snapshot_feed_is_mapped_onto_the_historical_schema(offline_schedule):
    frame = _snapshot_frame(["2026-08-01T10:00:00Z"])

    out = ingest._conform_depth_charts(frame, 2026)

    assert out.loc[0, "club_code"] == "BAL"
    assert out.loc[0, "full_name"] == "Lamar Jackson"
    assert out.loc[0, "position"] == "QB"
    assert out.loc[0, "depth_team"] == 1
    assert out.loc[0, "formation"] == "Offense"
    assert out.loc[0, "season"] == 2026


def test_preseason_snapshot_lands_on_week_one(offline_schedule):
    # Week 1 kicks off 2026-09-09, so an August chart describes that game.
    frame = _snapshot_frame(["2026-08-01T10:00:00Z", "2026-09-08T23:00:00Z"])

    out = ingest._conform_depth_charts(frame, 2026)

    assert out["week"].tolist() == [1.0, 1.0]
    assert out["game_type"].tolist() == ["REG", "REG"]


def test_in_season_snapshot_lands_on_the_next_week(offline_schedule):
    # Taken after week 1 kicked off, so it describes week 2, not week 1.
    frame = _snapshot_frame(["2026-09-10T12:00:00Z"])

    out = ingest._conform_depth_charts(frame, 2026)

    assert out.loc[0, "week"] == 2.0


def test_snapshots_after_the_final_kickoff_are_flagged_postseason(offline_schedule):
    frame = _snapshot_frame(["2026-12-01T12:00:00Z"])

    out = ingest._conform_depth_charts(frame, 2026)

    # Clipped onto the last known week, but excluded from regular-season views.
    assert out.loc[0, "game_type"] == "POST"


def test_position_groups_map_to_historical_formations(offline_schedule):
    frame = _snapshot_frame(["2026-08-01T10:00:00Z"] * 4)
    frame["pos_grp"] = ["3WR 1TE", "Base 3-4 D", "Base 4-3 D", "Special Teams"]

    out = ingest._conform_depth_charts(frame, 2026)

    assert out["formation"].tolist() == [
        "Offense",
        "Defense",
        "Defense",
        "Special Teams",
    ]


def test_missing_schedule_leaves_the_week_unknown(monkeypatch):
    def _unavailable(*args, **kwargs):
        raise ingest.DataUnavailableError("schedules offline")

    monkeypatch.setattr(ingest, "load_schedules", _unavailable, raising=True)

    out = ingest._conform_depth_charts(_snapshot_frame(["2026-08-01T10:00:00Z"]), 2026)

    # Guessing a week would silently move a point-in-time cutoff.
    assert out["week"].isna().all()


def test_historical_frames_pass_through_untouched():
    legacy = pd.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "club_code": ["BAL"],
            "full_name": ["Lamar Jackson"],
            "position": ["QB"],
            "depth_team": ["1"],
            "formation": ["Offense"],
            "game_type": ["REG"],
            "gsis_id": ["qb-bal"],
        }
    )

    assert ingest._conform_depth_charts(legacy, 2024).equals(legacy)


def test_conformed_snapshot_feeds_the_preseason_depth_snapshot(offline_schedule):
    from ffmodel.features.season_average import _preseason_depth_snapshot

    frame = _snapshot_frame(["2026-08-01T10:00:00Z"])
    conformed = ingest._conform_depth_charts(frame, 2026)

    depth = _preseason_depth_snapshot(conformed, cutoff_week=1)

    assert len(depth) == 1
    assert depth.loc[0, "depth_rank"] == 1
