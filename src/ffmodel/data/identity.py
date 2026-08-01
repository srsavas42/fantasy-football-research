"""Canonical NFL player dimension and cross-provider identifier joins."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ffmodel.data import ingest

# A real GSIS identifier looks like ``00-0041438``. Some upstream feeds populate
# a ``gsis_id`` column with a provider-native id instead — the draft-pick feed
# carries PFR-style values such as ``TAT143045`` for the newest class — so the
# shape is checked rather than trusted.
_GSIS_PREFIX = "00-"

_NAME_SUFFIXES = r"\b(jr|sr|ii|iii|iv|v)\b"


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


def is_gsis_id(values: pd.Series) -> pd.Series:
    """Whether each value is shaped like a real GSIS identifier."""
    return values.astype("string").str.startswith(_GSIS_PREFIX).fillna(False)


def normalize_player_name(names: pd.Series) -> pd.Series:
    """Casing-, punctuation- and suffix-free name for last-resort matching."""
    out = names.astype("string").str.lower()
    out = out.str.replace(r"[.'’]", "", regex=True)
    out = out.str.replace("-", " ", regex=False)
    out = out.str.replace(_NAME_SUFFIXES, "", regex=True)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def resolve_player_ids(
    frame: pd.DataFrame,
    *,
    player_dim: pd.DataFrame | None = None,
    pfr_column: str = "pfr_player_id",
    name_column: str = "player_name",
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> pd.Series:
    """Canonical GSIS ``player_id`` for rows carrying provider identifiers.

    Resolution is by provider id first, which is exact. Rows the id map cannot
    place that way fall back to a normalized name, but only when that name maps
    to exactly one player in the map — an ambiguous name is left unresolved
    rather than joined to the wrong career, which for a name shared across eras
    would silently attach one player's history to another.
    """
    if frame.empty:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    if player_dim is None:
        player_dim = load_player_dim(refresh=refresh, cache_dir=cache_dir)

    dim = player_dim[is_gsis_id(player_dim.get("gsis_id", pd.Series(dtype="string")))]
    resolved = pd.Series(pd.NA, index=frame.index, dtype="string")

    if pfr_column in frame.columns and "pfr_id" in dim.columns:
        bridge = (
            dim[["pfr_id", "gsis_id"]]
            .dropna(subset=["pfr_id"])
            .drop_duplicates("pfr_id")
            .set_index("pfr_id")["gsis_id"]
        )
        resolved = frame[pfr_column].astype("string").map(bridge).astype("string")

    if name_column in frame.columns and "player_name" in dim.columns:
        named = dim.assign(_key=normalize_player_name(dim["player_name"]))
        unique = named.groupby("_key")["gsis_id"].nunique()
        lookup = (
            named[named["_key"].isin(unique[unique.eq(1)].index)]
            .drop_duplicates("_key")
            .set_index("_key")["gsis_id"]
        )
        fallback = normalize_player_name(frame[name_column]).map(lookup).astype("string")
        resolved = resolved.where(resolved.notna(), fallback)

    return resolved


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
