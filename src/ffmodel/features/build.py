"""`build_features`: the single entry point Phase 3 imports.

Chains the feature layers into one enriched player-week frame:
load -> usage shares -> trailing usage -> trailing efficiency -> roles ->
availability -> game context. Snap/injury/schedule enrichments activate only
when nflverse is reachable; everything else works offline from the legacy CSVs.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ffmodel.data import ingest, load_player_weeks
from ffmodel.data.identity import load_player_dim
from ffmodel.features import context as context_mod
from ffmodel.features import injury as injury_mod
from ffmodel.features import roles as roles_mod
from ffmodel.features import snaps as snaps_mod
from ffmodel.features.efficiency_hist import EFFICIENCY_COLUMNS, add_efficiency
from ffmodel.features.trailing import add_trailing
from ffmodel.features.volume import (
    USAGE_COLUMNS,
    normalize_model_positions,
    usage_shares,
)

# Trailing usage covariates the share model consumes (leak-free EWMAs).
TRAILING_USAGE = [f"ewma_{c}" for c in ("target_share", "carry_share", "opportunity_share", "wopr")]

# Full set of engineered columns build_features guarantees on its output.
FEATURE_COLUMNS = [
    *USAGE_COLUMNS,
    *TRAILING_USAGE,
    *EFFICIENCY_COLUMNS,
    "role_signal",
    "role_rank",
    "role_tier",
    "is_active",
    "team_season",
]


def build_features(
    seasons: Iterable[int],
    source: str = "auto",
    span: int = 5,
    with_context: bool = True,
    cache_dir=None,
) -> pd.DataFrame:
    """Enriched player-week frame ready for the volume model."""
    seasons = list(seasons)
    pw = load_player_weeks(seasons, source=source, cache_dir=cache_dir)
    pw = normalize_model_positions(pw)

    pw = usage_shares(pw)
    pw = add_trailing(
        pw, ["target_share", "carry_share", "opportunity_share", "wopr"], span=span
    )
    pw = add_efficiency(pw, span=span)

    snap_signal = None
    injuries = None
    if source != "legacy":
        try:
            snap_rows = ingest.load_snap_counts(seasons, cache_dir=cache_dir)
            player_dim = load_player_dim(cache_dir=cache_dir)
            pw = snaps_mod.merge_snap_usage(pw, snap_rows, player_dim, span=span)
            snap_signal = "ewma_snap_share"
        except (ingest.DataUnavailableError, KeyError, ValueError):
            # Opportunity share remains the documented offline fallback.
            pass
        try:
            injuries = ingest.load_injuries(seasons, cache_dir=cache_dir)
        except (ingest.DataUnavailableError, ValueError):
            pass

    pw = roles_mod.add_roles(pw, snap_share_col=snap_signal)
    pw = injury_mod.add_availability(pw, injuries=injuries)

    if with_context:
        try:
            ctx = context_mod.game_context(seasons, cache_dir=cache_dir)
        except Exception:
            ctx = context_mod._neutral_frame(seasons)
        if not ctx.empty:
            pw = pw.merge(
                ctx.drop(columns=[c for c in ("team_season",) if c in ctx.columns]),
                on=["season", "week", "team"],
                how="left",
            )
    # team_season is always derivable, independent of schedule availability.
    pw["team_season"] = context_mod.team_season_key(pw)

    return pw
