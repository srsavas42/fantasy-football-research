"""Snap-share integration and PFR-to-GSIS identity resolution."""

from __future__ import annotations

import pandas as pd

from ffmodel.features.trailing import add_trailing

SNAP_COLUMNS = [
    "player_id",
    "season",
    "week",
    "team",
    "position",
    "offense_snaps",
    "snap_share",
]


def _share(values: pd.Series) -> pd.Series:
    strings = values.astype("string")
    had_percent = strings.str.endswith("%", na=False)
    numeric = pd.to_numeric(strings.str.rstrip("%"), errors="coerce")
    numeric = numeric.where(~had_percent, numeric / 100.0)
    # nflverse normally supplies a 0..1 fraction, but tolerate providers that
    # use 0..100 percentage points.
    numeric = numeric.where(numeric <= 1.0, numeric / 100.0)
    return numeric.clip(lower=0.0, upper=1.0)


def canonicalize_snaps(snaps: pd.DataFrame, player_dim: pd.DataFrame) -> pd.DataFrame:
    """Map snap rows to GSIS IDs and normalize offensive snap percentage."""
    out = snaps.copy()
    if "player_id" not in out.columns or out["player_id"].isna().all():
        if "player_id" in out.columns:
            out = out.drop(columns="player_id")
        pfr_col = next(
            (c for c in ("pfr_player_id", "pfr_id") if c in out.columns), None
        )
        dim_pfr = next(
            (c for c in ("pfr_id", "pfr_player_id") if c in player_dim.columns), None
        )
        if pfr_col and dim_pfr and "player_id" in player_dim.columns:
            lookup = player_dim[[dim_pfr, "player_id"]].dropna().drop_duplicates(dim_pfr)
            out = out.merge(lookup, left_on=pfr_col, right_on=dim_pfr, how="left")
        else:
            out["player_id"] = pd.NA

    offense_col = next(
        (c for c in ("offense_snaps", "off_snaps") if c in out.columns), None
    )
    pct_col = next(
        (c for c in ("offense_pct", "offense_percentage", "snap_pct") if c in out.columns),
        None,
    )
    out["offense_snaps"] = (
        pd.to_numeric(out[offense_col], errors="coerce") if offense_col else 0.0
    )
    out["snap_share"] = _share(out[pct_col]) if pct_col else pd.NA
    if "team" not in out.columns:
        out["team"] = pd.NA
    if "position" not in out.columns:
        out["position"] = pd.NA
    for column in ("season", "week"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    keep = [column for column in SNAP_COLUMNS if column in out.columns]
    return out[keep].drop_duplicates(["player_id", "season", "week", "team"])


def merge_snap_usage(
    player_weeks: pd.DataFrame,
    snaps: pd.DataFrame,
    player_dim: pd.DataFrame,
    *,
    span: int = 5,
) -> pd.DataFrame:
    """Left-join observed snaps and add leak-free trailing snap share."""
    canonical = canonicalize_snaps(snaps, player_dim)
    keys = ["player_id", "season", "week", "team"]
    values = canonical[keys + ["offense_snaps", "snap_share"]]
    out = player_weeks.merge(values, on=keys, how="left")
    return add_trailing(out, ["snap_share"], span=span)


SNAP_EXPOSURE_FEATURES = ("snap_games", "snap_availability")


def add_snap_exposure(rows: pd.DataFrame, snaps: pd.DataFrame | None = None):
    """Attach weeks-with-an-offensive-snap to existing season-average rows.

    The exposure the pipeline projects is ``games``, roster-active weeks. For a
    drafted player that is within about a game of what he actually played; for
    an undrafted one it measures employment instead of participation, and the
    gap is not small -- over 2022-2025 an undrafted quarterback is on a roster
    11.31 weeks and takes an offensive snap in 4.36. One regression fitted to a
    label meaning two different things for two halves of its population cannot
    separate the halves, which is what the availability diagnosis found.

    Added to existing frames rather than rebuilt into new ones, for the reason
    ``scripts/augment_cache_features.py`` gives: a fresh pull differs from the
    old one in far more than the new column, and an arm scored on a rebuild
    against a baseline scored on the old cache is not measuring the feature.

    Joined on player, season *and* team. A season-average row covers one
    player-team stint, so counting a traded player's snaps at both teams
    against one stint's ``team_games`` would credit him with exposure the row
    does not cover.
    """
    import numpy as np

    from ffmodel.data import ingest

    out = rows.copy()
    seasons = sorted(
        {int(s) for s in pd.to_numeric(out["season"], errors="coerce").dropna()}
    )
    if snaps is None:
        snaps = ingest.load_snap_counts(seasons)
    if "game_type" in snaps:
        snaps = snaps[snaps["game_type"].astype(str).eq("REG")]

    bridge = ingest.load_ids()[["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    played = snaps.merge(
        bridge, left_on="pfr_player_id", right_on="pfr_id", how="inner"
    )
    played["_snaps"] = pd.to_numeric(played["offense_snaps"], errors="coerce").fillna(0)
    played = played[played["_snaps"].gt(0)]
    counted = (
        played.drop_duplicates(["gsis_id", "season", "week", "team"])
        .groupby(["gsis_id", "season", "team"], as_index=False)
        .size()
        .rename(columns={"gsis_id": "player_id", "size": "snap_games"})
    )
    counted["season"] = counted["season"].astype(int)

    before = len(out)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype(int)
    out = out.merge(counted, on=["player_id", "season", "team"], how="left")
    if len(out) != before:
        raise AssertionError(
            f"the snap-exposure join changed the row count {before} -> {len(out)}"
        )

    # Missing means no offensive snap, not unknown: a player absent from a
    # covered team-season took none. Those rows average 1.70 roster games and
    # 0.01 games with a stat line across 2015-2025, and 3.7% were drafted.
    covered = (
        pd.to_numeric(out.get("snap_counts_observed"), errors="coerce").fillna(0).gt(0)
    )
    out["snap_games"] = out["snap_games"].where(
        out["snap_games"].notna() | ~covered
    ).fillna(0.0)
    slate = pd.to_numeric(out["team_games"], errors="coerce")
    # A stint cannot contain more games than its team played.
    out["snap_games"] = np.minimum(out["snap_games"], slate)
    out["snap_availability"] = out["snap_games"] / slate.replace(0, np.nan)
    return out
