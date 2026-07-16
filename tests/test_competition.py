"""Share-competition features: draft capital, incoming veterans, net opportunity."""

import numpy as np
import pandas as pd

from ffmodel.features import crossseason as cs
from ffmodel.features import draft


# ---- draft capital -------------------------------------------------------
def test_draft_capital_parses_known_pick():
    d = draft.load_draft_capital([2019], source="legacy")
    jacobs = d[d["player_name"].str.contains("Josh Jacobs")]
    assert not jacobs.empty
    row = jacobs.iloc[0]
    assert row["team"] == "OAK" and row["round"] == 1 and row["overall_pick"] == 24


def test_expected_rookie_claim_decreases_with_pick():
    early_t, early_c = draft.expected_rookie_claim(1, "RB")
    late_t, late_c = draft.expected_rookie_claim(150, "RB")
    assert early_c > late_c > 0
    # A WR's claim is receiving, not rushing.
    wt, wc = draft.expected_rookie_claim(10, "WR")
    assert wt > 0 and wc == 0


# ---- incoming competition ------------------------------------------------
def _usage(rows):
    df = pd.DataFrame(
        rows, columns=["player_name", "position", "season", "team", "target_share", "carry_share"]
    )
    df["key"] = cs.player_key(df)
    return df


def test_incoming_veteran_counts_prior_share_on_new_team():
    # V plays for A in 2019 (0.2 target share), moves to B in 2020. R stays on B.
    u = _usage([
        ["V", "WR", 2019, "A", 0.20, 0.0],
        ["R", "WR", 2019, "B", 0.10, 0.0],
        ["V", "WR", 2020, "B", 0.15, 0.0],
        ["R", "WR", 2020, "B", 0.10, 0.0],
    ])
    comp = cs.incoming_competition(u, 2019).set_index("team")
    # B gains V's 0.20 of incoming target competition; R is not "incoming".
    assert np.isclose(comp.loc["B", "incoming_comp_target"], 0.20)


def test_returning_player_is_not_incoming_competition():
    u = _usage([
        ["R", "WR", 2019, "B", 0.10, 0.0],
        ["R", "WR", 2020, "B", 0.10, 0.0],
    ])
    comp = cs.incoming_competition(u, 2019)
    assert comp.empty or np.isclose(comp["incoming_comp_target"].sum(), 0.0)


def test_rookie_draft_adds_competition():
    u = _usage([
        ["R", "RB", 2019, "B", 0.10, 0.30],
        ["R", "RB", 2020, "B", 0.10, 0.30],
    ])
    dc = pd.DataFrame([
        {"player_name": "Rook", "position": "RB", "season": 2020, "team": "B",
         "round": 1, "overall_pick": 5},
    ])
    comp = cs.incoming_competition(u, 2019, draft_capital=dc).set_index("team")
    # A high pick RB contributes meaningful carry competition to B.
    assert comp.loc["B", "incoming_comp_carry"] > 0.1


# ---- wired into transitions ---------------------------------------------
def test_transitions_expose_net_opportunity():
    t = cs.build_transitions([2017, 2018, 2019], source="legacy")
    for col in ("incoming_comp_target", "incoming_comp_carry",
                "net_target_opportunity", "net_carry_opportunity"):
        assert col in t.columns
    # Net = vacated - incoming, by construction.
    assert np.allclose(
        t["net_target_opportunity"],
        t["vacated_target_share"] - t["incoming_comp_target"],
    )
    assert (t["incoming_comp_target"] >= -1e-9).all()
