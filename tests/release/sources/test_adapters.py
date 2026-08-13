from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from ffmodel.data import sleeper
from ffmodel.data.providers import nflverse
from ffmodel.release import ReleaseIdentity
from ffmodel.release.sources import ImmutableSourceCapture, SourceCaptureError, json_schema_fingerprint, nflverse_dataset_source, sleeper_players_schema_fingerprint, sleeper_players_source


def test_sleeper_capture_adapter_preserves_deterministic_raw_json(monkeypatch):
    monkeypatch.setattr(sleeper, "get_json", lambda _: {"z": {"position": "QB"}, "a": {"position": "RB"}})
    payload = b'{"a":{"position":"RB"},"z":{"position":"QB"}}'
    source = sleeper_players_source(expected_schema_fingerprint=sleeper_players_schema_fingerprint(payload))
    assert source.source_id == "sleeper.players"
    assert source.loader() == b'{"a":{"position":"RB"},"z":{"position":"QB"}}'


def test_sleeper_schema_allows_routine_turnover_but_rejects_actual_schema_drift(tmp_path):
    original = b'{"p1":{"position":"QB","team":"A"},"p2":{"position":"RB","team":"B"}}'
    routine_turnover = b'{"p2":{"position":"WR","team":"C"},"p3":{"position":"TE","team":"D"}}'
    drifted = b'{"p3":{"role":"TE","team":"D"}}'
    expected = sleeper_players_schema_fingerprint(original)
    assert sleeper_players_schema_fingerprint(routine_turnover) == expected
    assert sleeper_players_schema_fingerprint(drifted) != expected
    capture = ImmutableSourceCapture(tmp_path / "Documents" / "releases", ReleaseIdentity(2026, "attempt-01"))
    source = sleeper_players_source(expected_schema_fingerprint=expected, loader=lambda: routine_turnover)
    now = lambda: datetime(2026, 8, 30, tzinfo=timezone.utc)
    capture.capture([source], code_identity="revision", runtime_identity={"python": "3.12"}, configuration={}, now=now)
    drifted_source = sleeper_players_source(expected_schema_fingerprint=expected, loader=lambda: drifted)
    second = ImmutableSourceCapture(tmp_path / "Documents" / "other", ReleaseIdentity(2026, "attempt-01"))
    with pytest.raises(SourceCaptureError, match="configured expected schema"):
        second.capture([drifted_source], code_identity="revision", runtime_identity={"python": "3.12"}, configuration={}, now=now)


def test_nflverse_capture_adapter_replays_a_self_describing_json_table(monkeypatch):
    monkeypatch.setattr(nflverse, "load", lambda *args, **kwargs: pd.DataFrame({"player_id": ["00-1"], "season": [2025]}))
    expected = json_schema_fingerprint(nflverse.capture_dataset_payload("players"))
    source = nflverse_dataset_source("players", expected_schema_fingerprint=expected)
    payload = source.loader()
    parsed = json.loads(payload)
    assert source.source_id.startswith("nflverse.players.")
    assert parsed["schema"]["fields"][0]["name"] == "player_id"


def test_nflverse_source_identity_binds_full_seasons_and_params():
    expected = "a" * 64
    one = nflverse_dataset_source("player_stats", [2024], summary_level="week", expected_schema_fingerprint=expected, loader=lambda: b"{}")
    two = nflverse_dataset_source("player_stats", [2025], summary_level="week", expected_schema_fingerprint=expected, loader=lambda: b"{}")
    three = nflverse_dataset_source("player_stats", [2024], summary_level="reg", expected_schema_fingerprint=expected, loader=lambda: b"{}")
    assert len({one.source_id, two.source_id, three.source_id}) == 3
    assert len({one.source_uri, two.source_uri, three.source_uri}) == 3
