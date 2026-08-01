"""Cross-provider player identity resolution.

The draft feed's own ``gsis_id`` is not dependable for a recent class — it
carries PFR-style values there, which match nothing in the roster or depth
feeds. Draft capital is the one signal a rookie has, so resolving these rows to
a real identifier is what lets a rookie join the rest of the pipeline at all.
"""

import numpy as np
import pandas as pd

from ffmodel.data.identity import (
    is_gsis_id,
    normalize_player_name,
    resolve_player_ids,
)
from ffmodel.features import season_average as sa


def _player_dim() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["00-0041438", "00-0041027", "00-0030000", "00-0030001"],
            "pfr_id": ["TAT143045", "LOV121782", "OLD000001", "OLD000002"],
            "player_name": [
                "Carnell Tate",
                "Jeremiyah Love",
                "Mike Williams",
                "Mike Williams",
            ],
        }
    )


def test_gsis_shape_check_rejects_provider_native_ids():
    values = pd.Series(["00-0041438", "TAT143045", None])

    assert is_gsis_id(values).tolist() == [True, False, False]


def test_name_normalisation_strips_case_punctuation_and_suffixes():
    names = pd.Series(["Omar Cooper Jr.", "Ja'Marr Chase", "A.J. Brown", "Odell-Beckham"])

    assert normalize_player_name(names).tolist() == [
        "omar cooper",
        "jamarr chase",
        "aj brown",
        "odell beckham",
    ]


def test_provider_id_resolves_exactly():
    frame = pd.DataFrame(
        {"pfr_player_id": ["TAT143045"], "player_name": ["Carnell Tate"]}
    )

    resolved = resolve_player_ids(frame, player_dim=_player_dim())

    assert resolved.tolist() == ["00-0041438"]


def test_unique_name_resolves_when_the_provider_id_is_unknown():
    frame = pd.DataFrame(
        {"pfr_player_id": ["NOT_IN_MAP"], "player_name": ["Jeremiyah Love"]}
    )

    resolved = resolve_player_ids(frame, player_dim=_player_dim())

    assert resolved.tolist() == ["00-0041027"]


def test_ambiguous_name_is_left_unresolved():
    # Two players share this name. Guessing would silently attach one player's
    # career history to the other, which is worse than no identifier at all.
    frame = pd.DataFrame(
        {"pfr_player_id": ["NOT_IN_MAP"], "player_name": ["Mike Williams"]}
    )

    resolved = resolve_player_ids(frame, player_dim=_player_dim())

    assert resolved.isna().all()


def test_espn_id_resolves_when_the_pfr_id_is_unknown():
    dim = _player_dim().assign(espn_id=["4837248", "4685522", "1", "2"])
    frame = pd.DataFrame(
        {
            "pfr_player_id": [None],
            "espn_id": ["4837248"],
            "player_name": ["Someone Renamed Upstream"],
        }
    )

    assert resolve_player_ids(frame, player_dim=dim).tolist() == ["00-0041438"]


def test_a_provider_id_pointing_at_another_placeholder_does_not_resolve():
    # The id map itself carries placeholder gsis values for the newest players.
    # Returning one would look canonical while matching nothing downstream.
    dim = pd.DataFrame(
        {
            "gsis_id": ["LAW090280"],
            "pfr_id": ["LawKe00"],
            "espn_id": ["4685441"],
            "player_name": ["Kendrick Law"],
        }
    )
    frame = pd.DataFrame(
        {
            "pfr_player_id": ["LawKe00"],
            "espn_id": ["4685441"],
            "player_name": ["Kendrick Law"],
        }
    )

    assert resolve_player_ids(frame, player_dim=dim).isna().all()


def _usage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_key": ["00-0041438", "name-only|WR"],
            "season": [2026, 2026],
            "player_name": ["Carnell Tate", "Name Only"],
            "position": ["WR", "WR"],
            "team": ["CHI", "CHI"],
        }
    )


def test_draft_capital_joins_on_identifier_then_falls_back_to_name(monkeypatch):
    draft = pd.DataFrame(
        {
            # Renamed upstream, so the name join alone would miss this player.
            "player_name": ["Carnell Tate II"],
            "position": ["WR"],
            "season": [2026],
            "team": ["CHI"],
            "round": [1],
            "overall_pick": [4],
            "player_id": ["00-0041438"],
        }
    )
    named = pd.DataFrame(
        {
            "player_name": ["Name Only"],
            "position": ["WR"],
            "season": [2026],
            "team": ["CHI"],
            "round": [3],
            "overall_pick": [80],
            "player_id": [None],
        }
    )
    monkeypatch.setattr(
        sa, "load_draft_capital", lambda *a, **k: pd.concat([draft, named])
    )

    out = sa._merge_draft_capital(_usage(), [2026], "auto")

    # Matched by identifier despite the name differing...
    assert out.loc[0, "overall_pick"] == 4
    # ...and the nameless-identifier row still resolves the old way.
    assert out.loc[1, "overall_pick"] == 80


def test_draft_capital_merge_survives_an_unavailable_feed(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("draft feed offline")

    monkeypatch.setattr(sa, "load_draft_capital", _boom)

    out = sa._merge_draft_capital(_usage(), [2026], "auto")

    assert out["overall_pick"].isna().all()
    assert len(out) == 2
