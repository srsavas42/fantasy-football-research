"""`build_features`: the single entry point Phase 3 imports.

Chains the feature layers into one enriched player-week frame:
load -> usage shares -> trailing usage -> trailing efficiency -> roles ->
availability -> game context. Snap/injury/schedule enrichments activate only
when nflverse is reachable; everything else works offline from the legacy CSVs.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ffmodel.data import load_player_weeks
from ffmodel.features import context as context_mod
from ffmodel.features import injury as injury_mod
from ffmodel.features import roles as roles_mod
from ffmodel.features.efficiency_hist import EFFICIENCY_COLUMNS, add_efficiency
from ffmodel.features.trailing import add_trailing
from ffmodel.features.volume import USAGE_COLUMNS, usage_shares

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

    pw = usage_shares(pw)
    pw = add_trailing(
        pw, ["target_share", "carry_share", "opportunity_share", "wopr"], span=span
    )
    pw = add_efficiency(pw, span=span)
    pw = roles_mod.add_roles(pw)
    pw = injury_mod.add_availability(pw)

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
