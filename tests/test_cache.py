"""Parquet cache: fetch once, then serve from disk."""

import json

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


def test_structured_cache_has_manifest_and_parameter_identity(tmp_path):
    first = cache.get_or_fetch(
        "nextgen_stats",
        lambda: pd.DataFrame({"player": ["A"], "value": [1.0]}),
        season=2024,
        cache_dir=tmp_path,
        provider="nflverse",
        params={"stat_type": "receiving"},
        source_url="https://example.test/data",
        license_name="test-license",
    )
    path = cache.cache_path(
        "nextgen_stats",
        2024,
        tmp_path,
        provider="nflverse",
        params={"stat_type": "receiving"},
    )
    assert len(first) == 1
    assert path.exists()
    manifest = json.loads(cache.manifest_path(path).read_text(encoding="utf-8"))
    assert manifest["rows"] == 1
    assert manifest["params"] == {"stat_type": "receiving"}
    assert manifest["sha256"]


def test_cache_path_changes_for_params_and_snapshot(tmp_path):
    receiving = cache.cache_path(
        "ngs", 2024, tmp_path, provider="nflverse", params={"type": "receiving"}
    )
    rushing = cache.cache_path(
        "ngs", 2024, tmp_path, provider="nflverse", params={"type": "rushing"}
    )
    later = cache.cache_path(
        "ngs",
        2024,
        tmp_path,
        provider="nflverse",
        params={"type": "receiving"},
        as_of="2024-09-02T12:00:00Z",
    )
    assert receiving != rushing
    assert receiving != later
