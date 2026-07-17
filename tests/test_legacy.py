"""Legacy CSVs load into the canonical schema across eras."""

import pandas as pd

from ffmodel.data import legacy
from ffmodel.data.schema import PLAYER_WEEK_COLUMNS


def test_weekly_schema_and_coverage():
    df = legacy.load_weekly([1999, 2021])
    assert list(df.columns) == PLAYER_WEEK_COLUMNS
    assert set(df["season"].unique()) == {1999, 2021}
    # 2021 moved to 18 regular-season weeks
    assert df[df["season"] == 2021]["week"].max() == 18
    assert df[df["season"] == 1999]["week"].max() == 17
    assert (df["source"] == "legacy").all()
    assert (df["pass_sacks_available"] == 0).all()
    assert df["player_name"].notna().all()


def test_weekly_missing_season_is_empty():
    df = legacy.load_weekly([1990])
    assert df.empty
    assert list(df.columns) == PLAYER_WEEK_COLUMNS


def test_yearly_old_and_new_headers():
    df = legacy.load_yearly([1970, 2021])
    assert list(df.columns) == PLAYER_WEEK_COLUMNS
    assert df["week"].isna().all()
    y1970 = df[df["season"] == 1970]
    assert len(y1970) > 0
    assert (y1970["pass_yds"] >= 0).all() or y1970["pass_yds"].notna().all()


def test_snapcounts():
    df = legacy.load_snapcounts([2013, 2020])
    assert {"player_name", "position", "team", "snap_pct", "season"} <= set(df.columns)
    assert set(df["season"].unique()) == {2013, 2020}


def test_stat_columns_numeric():
    df = legacy.load_weekly([2019])
    assert pd.api.types.is_float_dtype(df["rec_yds"])
    assert pd.api.types.is_float_dtype(df["pass_yds"])
