"""The scheme carrier's carried backfield-usage tendency, as a role feature.

``scripts/screen_coaching_tree_transfer.py`` measures which parts of an
offense's *shape* an arriving play-caller brings with him. Of eight candidates
exactly one survives a control for the team's own previous three seasons:
``rb_target_share``, the fraction of team targets going to running backs.
Target concentration, run/pass balance, five depth-of-target shapes, and
tight-end and receiver share are all absorbed by team persistence or measure
as nothing.

Measured properly -- era-normalised, over stops back to 1999 -- that survivor
is **+0.211 at p=0.003 on n=197** (scripts/screen_coaching_deep_history.py).
The first version of this screen reported +0.204 at p=0.027 on n=121 because it
could only see stops from 2015 on; both the window and the normalisation were
corrections to how the quantity is measured, not new hypotheses.

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

The value is a within-season z-score before interacting, so a back on a team
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

# Flat weighting across a coach's career. Swept on the deep window, where the
# sweep has real range to resolve: +0.223 at a three-year half-life against
# +0.216 at ten. Back usage reads as a stable career trait rather than recent
# form. (An earlier sweep on 2015+ stops only found the same thing, but every
# stop there was within nine years, so it could not have refuted the claim.)
RECENCY_HALF_LIFE: float | None = None

# How far back to reach for the shapes of a coach's prior stops. The binding
# limit on this feature was never the coaching data: of 1,630 external
# play-calling stops behind 2016-2025 seasons, 1,087 are pre-2015 and had no
# shape to attach when shapes came from the modeling frames. nflverse player
# weeks run to 1999 and carry the three columns the shape needs, which is a far
# lighter requirement than the full feature frame.
DEEP_HISTORY_FIRST_SEASON = 1999


def _deep_team_rb_target_share(last_season: int) -> pd.DataFrame:
    """Running-back share of team targets per team-season, back to 1999.

    Era-normalised, and that is not optional. League-wide back usage drifts
    hard -- 0.230 in 1999 against 0.175 in 2024 -- so averaging a 2005 stop with
    a 2020 stop on the raw scale adds eras together rather than tendencies. On
    raw shares the transfer measures +0.249 over a 2015+ window and *-0.021*
    over a 1999+ one; z-scored within season it is +0.267 and +0.211. The raw
    collapse was the drift, not the absence of an effect.
    """
    from ffmodel.data import load_player_weeks
    from ffmodel.features.season_average import normalize_model_positions

    weeks = normalize_model_positions(
        load_player_weeks(
            range(DEEP_HISTORY_FIRST_SEASON, int(last_season) + 1), source="nflverse"
        )
    )
    weeks = weeks[weeks["team"].notna() & weeks["season"].notna()].copy()
    weeks["targets"] = pd.to_numeric(weeks["targets"], errors="coerce").fillna(0.0)
    weeks["_rb"] = weeks["targets"].where(weeks["position"].eq("RB"), 0.0)
    grouped = weeks.groupby(["season", "team"], as_index=False).agg(
        _rb=("_rb", "sum"), _all=("targets", "sum")
    )
    grouped["season"] = grouped["season"].astype(int)
    share = grouped["_rb"] / grouped["_all"].where(grouped["_all"] > 0)
    by_season = share.groupby(grouped["season"])
    spread = by_season.transform("std")
    grouped["rb_target_share_z"] = (share - by_season.transform("mean")) / spread.where(
        spread > 0
    )
    return grouped[["season", "team", "rb_target_share_z"]]


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

    shapes = _deep_team_rb_target_share(int(pd.to_numeric(out["season"]).max()))
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
    stops = stops[pd.to_numeric(stops["rb_target_share_z"], errors="coerce").notna()]
    if stops.empty:
        out[COACHING_SCHEME_FEATURES[0]] = 0.0
        return out

    if RECENCY_HALF_LIFE is None:
        stops = stops.assign(_weight=1.0)
    else:
        stops = stops.assign(
            _weight=0.5 ** (stops["recency_years"].astype(float) / RECENCY_HALF_LIFE)
        )
    stops = stops.assign(_value=stops["rb_target_share_z"].astype(float) * stops["_weight"])
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
    # Already a z-score against each stop's own season, so it is centred on the
    # league by construction and needs no further recentring against the
    # response season -- which is what keeps the response season's own outcome
    # out of the feature's location.
    centred = values.fillna(0.0)
    is_back = out["position"].astype(str).str.upper().eq("RB").astype(float)
    out[COACHING_SCHEME_FEATURES[0]] = (centred * is_back).to_numpy(dtype=float)
    return out.drop(columns=["carried_rb_target_share"])
