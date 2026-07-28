"""Empirical role tiers.

Role is *earned usage*, not a listed depth chart. Within each
(season, week, team, position) group we rank players by their trailing usage
signal and bucket the rank into a tier (1 / 2 / 3 / 4+). The signal is trailing
snap share when snap data is present (nflverse path), otherwise trailing
opportunity share (always available offline).

Cold start: a player with no in-season history yet (week 1, just changed teams,
returning from injury) has a NaN trailing signal. We fall back to that player's
prior-season mean opportunity share so week-1 ranks are still meaningful, and
the empirical signal takes over automatically once games accumulate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.features.volume import MODEL_POSITIONS

ROLE_GROUP = ["season", "week", "team", "position"]
MAX_TIER = 4  # tiers 1..3 explicit; 4 means "4th or deeper".


def _prior_season_opportunity(pw_with_shares: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season) mean opportunity_share, mapped forward one season.

    Returns columns [player_name, position, season, prior_opp_share] where
    prior_opp_share is the player's mean from season-1 (NaN if none).
    """
    per = (
        pw_with_shares.groupby(["player_name", "position", "season"], dropna=False)[
            "opportunity_share"
        ]
        .mean()
        .reset_index(name="_season_opp")
    )
    per["season"] = per["season"] + 1  # attribute to the *following* season
    return per.rename(columns={"_season_opp": "prior_opp_share"})


def add_roles(pw: pd.DataFrame, snap_share_col: str | None = None) -> pd.DataFrame:
    """Add `role_signal`, `role_rank`, and `role_tier` columns.

    Expects `opportunity_share` (contemporaneous) and `ewma_opportunity_share`
    (trailing) already present. `snap_share_col`, if given and present, is the
    preferred trailing signal.
    """
    out = pw.copy()

    # Preferred trailing signal.
    if snap_share_col and snap_share_col in out.columns:
        signal = out[snap_share_col]
    else:
        signal = out.get("ewma_opportunity_share", pd.Series(np.nan, index=out.index))

    # Cold-start fallback: prior-season opportunity share.
    prior = _prior_season_opportunity(out)
    out = out.merge(prior, on=["player_name", "position", "season"], how="left")
    signal = signal.reindex(out.index)
    out["role_signal"] = signal.where(signal.notna(), out["prior_opp_share"])

    # Rank within team-position-week; highest signal = rank 1. Unknown signals
    # sort last (rank NaN -> placed after ranked players).
    out["role_rank"] = (
        out.groupby(ROLE_GROUP, dropna=False)["role_signal"]
        .rank(method="first", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    out["role_tier"] = out["role_rank"].clip(upper=MAX_TIER).astype("Int64")

    # Every modeled position (including QB) gets a meaningful role tier.
    non_model = ~out["position"].isin(MODEL_POSITIONS)
    out.loc[non_model, ["role_rank", "role_tier"]] = pd.NA

    return out.drop(columns=["prior_opp_share"])
