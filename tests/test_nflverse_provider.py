"""nflreadpy adapter tests do not require the real dependency or network."""

import sys

import pandas as pd
import pytest

from ffmodel.data.providers import nflverse


class _PolarsLike:
    def __init__(self, frame):
        self.frame = frame

    def to_pandas(self):
        return self.frame.copy()


class _FakeNflreadpy:
    @staticmethod
    def load_player_stats(seasons=None, summary_level="week"):
        return _PolarsLike(
            pd.DataFrame({"season": seasons, "summary_level": [summary_level] * len(seasons)})
        )

    @staticmethod
    def load_players():
        return _PolarsLike(pd.DataFrame({"gsis_id": ["00-1"]}))


def test_dispatches_seasons_and_parameters(monkeypatch):
    monkeypatch.setitem(sys.modules, "nflreadpy", _FakeNflreadpy)
    out = nflverse.load("player_stats", [2023, 2024], summary_level="reg")
    assert out["season"].tolist() == [2023, 2024]
    assert set(out["summary_level"]) == {"reg"}


def test_static_dataset_rejects_seasons(monkeypatch):
    monkeypatch.setitem(sys.modules, "nflreadpy", _FakeNflreadpy)
    with pytest.raises(ValueError, match="does not accept seasons"):
        nflverse.load("players", [2024])


def test_static_dataset(monkeypatch):
    monkeypatch.setitem(sys.modules, "nflreadpy", _FakeNflreadpy)
    assert nflverse.load("players")["gsis_id"].tolist() == ["00-1"]
