"""The scheme carrier's carried backfield-usage tendency, as a role feature.

``scripts/screen_coaching_tree_transfer.py`` measures which parts of an
offense's *shape* an arriving play-caller brings with him. Of three candidates
only one survives a control for the team's own previous three seasons:
``rb_target_share``, the fraction of team targets going to running backs, at a
partial correlation of +0.204 (p=0.027, n=121). Target concentration and
run/pass balance are entirely absorbed by team persistence.

Two properties make it usable rather than another team effect wearing a hat:

* It is leakage-safe by construction. ``build_scheme_lineage`` restricts every
  prior stop to ``prior_season < season``, and coaching hires are known in the
  offseason, so the whole feature is settled before week 1.
* The effect *strengthens* when stops at the coach's current franchise are
  dropped (+0.163 to +0.204), which rules out the mechanical reading -- that a
  coach who stayed simply carries his own team's history under another name.

The emitted column is deliberately an interaction, not a level.
``carried_rb_target_share`` is constant within a team-season, and the target
softmax normalises within team-season, so a main effect cancels *exactly* --
the same reason ``prior_target_room_competition`` measured as nothing in
docs/target-competition-2026-09.md. Multiplying by the back indicator is what
moves running backs relative to receivers inside the room instead of moving the
room.

The value is centred on the league mean before interacting, so a back on a team
whose play-caller carries a league-average tendency contributes zero rather than
a constant offset that the position effect would have to absorb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.data.coaching import load_scheme_lineage

COACHING_SCHEME_FEATURES = ("prior_coach_rb_target_share_x_rb",)

# Roles at a prior stop that plausibly mean the coach shaped the distribution.
# A quality-control or position coach does not choose the offense; the
# quarterbacks coach is included because the screen's role sweep found it adds
# signal rather than noise (n rises 121 -> 145, partial +0.204 -> +0.231).
SCHEME_ROLE_PATTERNS = ("offensive coordinator", "head coach", "quarterbacks coach")
EXCLUDED_ROLE_PATTERNS = ("assistant", "quality control", "intern")

# Flat weighting across a coach's career. The screen swept half-lives from two
# years to flat and found the result nearly unchanged (+0.189 to +0.210), with
# flat marginally the strongest -- back usage reads as a stable career trait
# rather than recent form, so there is nothing for a decay to buy.
RECENCY_HALF_LIFE: float | None = None


def _team_rb_target_share(rows: pd.DataFrame) -> pd.DataFrame:
    """Observed running-back share of team targets, per team-season."""
    frame = rows[
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    ].copy()
    targets = pd.to_numeric(frame.get("targets"), errors="coerce").fillna(0.0)
    team_total = targets.groupby([frame["season"], frame["team"]]).transform("sum")
    back_targets = targets.where(frame["position"].eq("RB"), 0.0)
    grouped = pd.DataFrame(
        {
            "season": frame["season"],
            "team": frame["team"],
            "_rb": back_targets,
            "_all": targets,
        }
    ).groupby(["season", "team"], as_index=False).sum()
    grouped["rb_target_share"] = grouped["_rb"] / grouped["_all"].where(grouped["_all"] > 0)
    return grouped[["season", "team", "rb_target_share"]]


def add_coaching_scheme_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach the scheme carrier's carried backfield tendency, interacted.

    Rows whose team-season has no usable lineage -- an unresolved scheme coach,
    or one whose prior stops are all at this same franchise -- get 0.0, which
    for a centred interaction reads as "no information", not as a low value.
    """
    out = rows.copy()
    required = {"season", "team", "position"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"coaching scheme rows are missing columns: {sorted(missing)}")

    shapes = _team_rb_target_share(out)
    lineage = load_scheme_lineage()
    if lineage.empty:
        out[COACHING_SCHEME_FEATURES[0]] = 0.0
        return out

    role = lineage["prior_role"].astype(str).str.lower()
    lineage = lineage[
        role.str.contains("|".join(SCHEME_ROLE_PATTERNS), na=False)
        & ~role.str.contains("|".join(EXCLUDED_ROLE_PATTERNS), na=False)
    ]
    # Stops at the franchise he now works for are dropped: including them makes
    # the feature partly a restatement of the team's own history for a coach who
    # stayed, which is the confound the screen ruled out by removing them.
    lineage = lineage[lineage["prior_team_code"] != lineage["franchise_code"]]

    stops = lineage.merge(
        shapes.rename(columns={"season": "prior_season", "team": "prior_team_code"}),
        on=["prior_season", "prior_team_code"],
        how="inner",
    )
    stops = stops[pd.to_numeric(stops["rb_target_share"], errors="coerce").notna()]
    if stops.empty:
        out[COACHING_SCHEME_FEATURES[0]] = 0.0
        return out

    if RECENCY_HALF_LIFE is None:
        stops = stops.assign(_weight=1.0)
    else:
        stops = stops.assign(
            _weight=0.5 ** (stops["recency_years"].astype(float) / RECENCY_HALF_LIFE)
        )
    stops = stops.assign(_value=stops["rb_target_share"].astype(float) * stops["_weight"])
    carried = stops.groupby(["season", "franchise_code"], as_index=False).agg(
        _numerator=("_value", "sum"), _denominator=("_weight", "sum")
    )
    carried["carried_rb_target_share"] = carried["_numerator"] / carried[
        "_denominator"
    ].where(carried["_denominator"] > 0)
    carried = carried.rename(columns={"franchise_code": "team"})

    out = out.merge(
        carried[["season", "team", "carried_rb_target_share"]],
        on=["season", "team"],
        how="left",
    )
    values = pd.to_numeric(out["carried_rb_target_share"], errors="coerce")
    # Centre on the league mean of the carried values themselves, computed once
    # over every team-season that has one. Centring on the *observed* league
    # share instead would leak the response season's own outcome into the
    # feature's location.
    centre = float(values.dropna().mean()) if values.notna().any() else 0.0
    centred = (values - centre).fillna(0.0)
    is_back = out["position"].astype(str).str.upper().eq("RB").astype(float)
    out[COACHING_SCHEME_FEATURES[0]] = (centred * is_back).to_numpy(dtype=float)
    return out.drop(columns=["carried_rb_target_share"])
