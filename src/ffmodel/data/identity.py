"""Canonical NFL player dimension and cross-provider identifier joins."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ffmodel.data import ingest


def canonicalize_player_dim(
    players: pd.DataFrame, fantasy_ids: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Use GSIS as ``player_id`` while retaining every provider identifier."""
    out = players.copy()
    if "gsis_id" not in out.columns and "player_id" in out.columns:
        out["gsis_id"] = out["player_id"]
    out["player_id"] = out.get("gsis_id", pd.Series(pd.NA, index=out.index))
    if "player_name" not in out.columns:
        for candidate in ("display_name", "full_name", "football_name"):
            if candidate in out.columns:
                out["player_name"] = out[candidate]
                break

    if fantasy_ids is not None and not fantasy_ids.empty and "gsis_id" in fantasy_ids.columns:
        extra = fantasy_ids.drop_duplicates("gsis_id").copy()
        new_columns = [c for c in extra.columns if c == "gsis_id" or c not in out.columns]
        out = out.merge(extra[new_columns], on="gsis_id", how="left")
    return out


def load_player_dim(
    *, refresh: bool = False, cache_dir: Path | None = None
) -> pd.DataFrame:
    """Load nflverse player metadata and augment it with fantasy-platform IDs."""
    players = ingest.load_ids(refresh=refresh, cache_dir=cache_dir)
    try:
        fantasy_ids = ingest.load_ff_playerids(refresh=refresh, cache_dir=cache_dir)
    except ingest.DataUnavailableError:
        fantasy_ids = None
    return canonicalize_player_dim(players, fantasy_ids)
