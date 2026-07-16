"""Trailing features must never see the present or future — the core guarantee."""

import numpy as np
import pandas as pd

from ffmodel.features.trailing import add_trailing, player_key


def _series(vals, weeks=None, season=2020, name="X", pos="WR"):
    weeks = weeks or list(range(1, len(vals) + 1))
    return pd.DataFrame(
        {
            "player_id": [None] * len(vals),
            "player_name": [name] * len(vals),
            "position": [pos] * len(vals),
            "team": ["A"] * len(vals),
            "season": [season] * len(vals),
            "week": weeks,
            "val": list(map(float, vals)),
        }
    )


def test_no_leakage_spike_excluded():
    df = _series([10, 20, 30, 40, 999])
    out = add_trailing(df, ["val"], span=3)
    # Week 5's trailing value must be built from weeks 1-4 only.
    expected = pd.Series([10.0, 20.0, 30.0, 40.0]).ewm(span=3).mean().iloc[-1]
    assert np.isclose(out["ewma_val"].iloc[4], expected)
    # And must not equal a version that includes the week-5 spike.
    leaked = pd.Series([10.0, 20.0, 30.0, 40.0, 999.0]).ewm(span=3).mean().iloc[-1]
    assert not np.isclose(out["ewma_val"].iloc[4], leaked)


def test_first_week_has_no_history():
    out = add_trailing(_series([5, 6, 7]), ["val"], span=3)
    assert pd.isna(out["ewma_val"].iloc[0])
    assert out["ewma_val"].iloc[1:].notna().all()


def test_players_do_not_bleed_together():
    a = _series([100, 100, 100], name="A")
    b = _series([1, 1, 1], name="B")
    out = add_trailing(pd.concat([a, b], ignore_index=True), ["val"], span=3)
    a_out = out[out["player_name"] == "A"]["ewma_val"].iloc[-1]
    b_out = out[out["player_name"] == "B"]["ewma_val"].iloc[-1]
    assert np.isclose(a_out, 100.0)
    assert np.isclose(b_out, 1.0)


def test_original_row_order_preserved():
    df = _series([1, 2, 3, 4], weeks=[4, 2, 1, 3])  # deliberately unsorted
    out = add_trailing(df, ["val"], span=3)
    assert list(out["week"]) == [4, 2, 1, 3]
    # Chronologically week 1 is first -> its trailing value is NaN even though
    # it sits in row position 2.
    assert pd.isna(out.loc[out["week"] == 1, "ewma_val"].iloc[0])


def test_player_key_falls_back_without_id():
    df = _series([1, 2], name="Y", pos="RB")
    key = player_key(df)
    assert (key == "Y|RB").all()
