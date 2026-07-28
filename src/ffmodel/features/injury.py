"""Availability and the active set.

The Dirichlet-Multinomial share model allocates opportunity only among players
who actually play a given week, so training needs the observed active set per
team-position-week. Availability is inferred from the stat lines themselves
(a player who recorded a snap/touch/target was active); when nflverse injury
reports are reachable they add designation detail, but they're never required.

Projection-time *forward* active sets (who's expected to play next week) are a
Phase 5/7 concern; this module only labels observed history.
"""

from __future__ import annotations

import pandas as pd


def add_availability(pw: pd.DataFrame, injuries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add `is_active` (played this week) and optional injury `report_status`."""
    out = pw.copy()
    activity = (
        out["pass_att"] + out["rush_att"] + out["targets"] + out["receptions"]
    )
    if "offense_snaps" in out.columns:
        activity = activity + pd.to_numeric(out["offense_snaps"], errors="coerce").fillna(0)
    out["is_active"] = (activity > 0).astype(int)

    if injuries is not None and not injuries.empty:
        inj = injuries.rename(
            columns={"gsis_id": "player_id", "report_status": "report_status"}
        )
        keys = [k for k in ("player_id", "season", "week") if k in inj.columns]
        if {"player_id", "season", "week"} <= set(inj.columns) and "player_id" in out.columns:
            out = out.merge(
                inj[keys + ["report_status"]].drop_duplicates(keys),
                on=keys,
                how="left",
            )
    if "report_status" not in out.columns:
        out["report_status"] = pd.NA
    return out


def active_set(team_week: pd.DataFrame) -> pd.DataFrame:
    """Active players in one team-position-week (the Dirichlet support)."""
    return team_week[team_week["is_active"] == 1]
