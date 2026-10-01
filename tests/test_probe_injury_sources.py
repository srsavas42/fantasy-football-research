"""The injury-source probe: field accounting, keyword scans and the snapshot it writes.

The payloads below are synthetic and only approximate what ESPN and Sleeper serve; they
test the probe's own logic, not the providers' schemas. The live run is what reports the
real fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import probe_injury_sources as probe  # noqa: E402

ESPN = {
    "injuries": [
        {
            "displayName": "Team A",
            "injuries": [
                {
                    "status": "Out",
                    "shortComment": "Smith (hamstring) torn; out 4-6 weeks.",
                    "details": {"type": "Hamstring", "detail": "Tear", "returnDate": "2026-11-01"},
                },
                {"status": "Questionable", "shortComment": "Jones is day-to-day with soreness.", "details": {"type": "Ankle"}},
            ],
        },
        {"displayName": "Team B", "injuries": []},
    ]
}

SLEEPER = {
    "1": {"position": "RB", "fantasy_positions": ["RB"], "injury_status": "Out", "injury_body_part": "Knee",
          "injury_notes": "Torn ACL, season-ending surgery."},
    "2": {"position": "WR", "fantasy_positions": ["WR"], "injury_status": None},
    "3": {"position": "K", "fantasy_positions": ["K"], "injury_status": "Out"},
    "4": "not a record",
}


def test_flatten_drops_list_indexes_and_keeps_leaves():
    leaves = list(probe.flatten({"a": [{"b": 1}, {"b": 2}], "c": {"d": "x"}}))
    assert leaves == [("a[].b", 1), ("a[].b", 2), ("c.d", "x")]


def test_espn_records_carry_the_team_and_skip_empty_teams():
    records = probe.espn_records(ESPN)
    assert len(records) == 2
    assert {r["team"] for r in records} == {"Team A"}


def test_field_table_counts_filled_fields():
    table = {row["path"]: row for row in probe.field_table(probe.espn_records(ESPN))}
    assert table["details.type"]["filled"] == 2
    assert table["details.returnDate"]["filled"] == 1
    assert table["details.returnDate"]["share"] == pytest.approx(0.5)


def test_sleeper_keeps_only_injured_fantasy_positions():
    kept = probe.sleeper_injury_records(SLEEPER)
    assert [r["player_id"] for r in kept] == ["1"]


def test_keyword_scan_finds_nature_and_timeline_words():
    nature = probe.keyword_hits(["Torn ACL", "left ankle sprain", "sore hamstring", "healthy"], probe.NATURE_WORDS)
    assert nature["tear"] == 1 and nature["sprain"] == 1 and nature["soreness"] == 1
    timeline = probe.keyword_hits(["out 4-6 weeks", "week-to-week", "day-to-day"], probe.TIMELINE_WORDS)
    assert timeline["weeks"] == 1 and timeline["week-to-week"] == 1 and timeline["day-to-day"] == 1


def test_main_writes_a_dated_snapshot_and_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(probe, "get_json", lambda url, **_: ESPN if "espn" in url else SLEEPER)
    assert probe.main(["--out", str(tmp_path)]) == 0
    folder = next(tmp_path.iterdir())
    assert {p.name for p in folder.iterdir()} == {"espn_injuries.json", "sleeper_injuries.json", "summary.json"}
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["espn"]["records"] == 2 and summary["sleeper"]["records"] == 1
    assert summary["sleeper"]["nature"]["tear"] == 1
    assert "details.returnDate" in capsys.readouterr().out


def test_a_blocked_host_is_reported_not_raised(tmp_path, monkeypatch):
    def blocked(url, **_):
        raise probe.RemoteDataError("403")

    monkeypatch.setattr(probe, "get_json", blocked)
    assert probe.main(["--out", str(tmp_path)]) == 1
    summary = json.loads(next(tmp_path.iterdir()).joinpath("summary.json").read_text())
    assert "error" in summary["espn"] and "error" in summary["sleeper"]
