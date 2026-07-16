"""Parquet cache: fetch once, then serve from disk."""

import pandas as pd

from ffmodel.data import cache


def test_get_or_fetch_roundtrip(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    first = cache.get_or_fetch("demo", fetch, season=2024, cache_dir=tmp_path)
    second = cache.get_or_fetch("demo", fetch, season=2024, cache_dir=tmp_path)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)
    assert (tmp_path / "demo_2024.parquet").exists()


def test_refresh_forces_fetch(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return pd.DataFrame({"a": [len(calls)]})

    cache.get_or_fetch("demo", fetch, cache_dir=tmp_path)
    out = cache.get_or_fetch("demo", fetch, refresh=True, cache_dir=tmp_path)
    assert len(calls) == 2
    assert out["a"].iloc[0] == 2
