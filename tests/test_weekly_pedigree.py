"""Draft capital, and the encoding decision that carries it.

Undrafted has to be a *value* rather than a gap. Filling a missing pick with the
column median asserts an average draft slot for a player nobody drafted, which is
the opposite of what his absence says -- the same mistake the ADP encoding was
built to avoid, in a column where it is easier to make by accident because a
missing pick looks like missing data.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import pedigree as pedigree_module
from ffmodel.weekly.pedigree import (
    UNDRAFTED_OVERALL,
    UNDRAFTED_ROUND,
    add_pedigree_features,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_key": ["first", "seventh", "never"],
            "season": [2020, 2020, 2020],
            "week": [1, 1, 1],
            "position": ["RB", "WR", "TE"],
        }
    )


def _stub(monkeypatch, capital, experience) -> None:
    monkeypatch.setattr(pedigree_module, "load_draft_capital", lambda s: capital)
    monkeypatch.setattr(pedigree_module, "load_experience", lambda s: experience)


def test_undrafted_is_a_value_not_a_gap(monkeypatch) -> None:
    capital = pd.DataFrame(
        {
            "player_key": ["first", "seventh"],
            "draft_round": [1.0, 7.0],
            "draft_overall": [3.0, 240.0],
        }
    )
    experience = pd.DataFrame(
        {"season": [2020] * 3, "player_key": ["first", "seventh", "never"],
         "years_exp": [2.0, 5.0, 4.0]}
    )
    _stub(monkeypatch, capital, experience)
    got = add_pedigree_features(_panel()).set_index("player_key")

    assert got.loc["never", "undrafted"] == 1.0
    assert got.loc["first", "undrafted"] == 0.0
    # Placed past the last pick, not at the middle of the board.
    assert got.loc["never", "draft_overall"] == UNDRAFTED_OVERALL
    assert got.loc["never", "draft_round"] == UNDRAFTED_ROUND
    assert got.loc["never", "draft_overall"] > got.loc["seventh", "draft_overall"]
    # No missing values reach the design at all.
    for column in ("draft_round", "draft_log_overall", "undrafted"):
        assert got[column].notna().all()


def test_the_log_pick_orders_the_board(monkeypatch) -> None:
    capital = pd.DataFrame(
        {
            "player_key": ["first", "seventh"],
            "draft_round": [1.0, 7.0],
            "draft_overall": [3.0, 240.0],
        }
    )
    _stub(monkeypatch, capital, pd.DataFrame(
        columns=["season", "player_key", "years_exp"]
    ))
    got = add_pedigree_features(_panel()).set_index("player_key")
    assert (
        got.loc["first", "draft_log_overall"]
        < got.loc["seventh", "draft_log_overall"]
        < got.loc["never", "draft_log_overall"]
    )
    assert got.loc["first", "draft_log_overall"] == pytest.approx(np.log(3.0))


def test_missing_feeds_leave_everyone_undrafted(monkeypatch) -> None:
    """No draft feed must mean no signal, not a crash and not a fabricated one."""
    _stub(
        monkeypatch,
        pd.DataFrame(columns=["player_key", "draft_round", "draft_overall"]),
        pd.DataFrame(columns=["season", "player_key", "years_exp"]),
    )
    got = add_pedigree_features(_panel())
    assert (got["undrafted"] == 1.0).all()
    assert got["years_exp"].isna().all()


def test_a_player_keeps_one_draft_slot(monkeypatch) -> None:
    """A duplicated feed row must not average two picks into a third."""
    raw = pd.DataFrame(
        {
            "season": [2018, 2018],
            "gsis_id": ["dupe", "dupe"],
            "round": [1.0, 4.0],
            "pick": [12.0, 118.0],
        }
    )
    monkeypatch.setattr(pedigree_module.ingest, "load_draft_picks", lambda s: raw)
    got = pedigree_module.load_draft_capital([2020])
    assert len(got) == 1
    assert got.iloc[0]["draft_overall"] == 12.0
