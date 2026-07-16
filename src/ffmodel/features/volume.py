"""Team volume aggregates and per-player usage shares.

All quantities here are *contemporaneous* — they describe what actually
happened in week W. Usage shares become the volume model's regression targets
/ multinomial exposure; their trailing EWMAs (built via `trailing.add_trailing`)
become the predictors. Nothing here reads external data: it's derived purely
from the canonical player-week stat frame, so it works fully offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Positions we model opportunity for. QB volume is pass attempts, handled by
# the team model, so it's excluded from the receiving/rushing share groups.
SKILL_POSITIONS = ("RB", "WR", "TE")

TEAM_TOTAL_COLUMNS = [
    "team_pass_att",
    "team_rush_att",
    "team_targets",
    "team_plays",
    "team_pass_rate",
]

USAGE_COLUMNS = [
    "target_share",
    "carry_share",
    "opportunity",
    "opportunity_share",
    "wopr",
]


def team_game_totals(pw: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team) offensive totals used as share denominators."""
    grp = pw.groupby(["season", "week", "team"], dropna=False)
    totals = grp.agg(
        team_pass_att=("pass_att", "sum"),
        team_rush_att=("rush_att", "sum"),
        team_targets=("targets", "sum"),
    ).reset_index()
    totals["team_plays"] = totals["team_pass_att"] + totals["team_rush_att"]
    totals["team_pass_rate"] = _safe_div(totals["team_pass_att"], totals["team_plays"])
    return totals


def usage_shares(pw: pd.DataFrame) -> pd.DataFrame:
    """Add contemporaneous usage-share columns to the player-week frame."""
    totals = team_game_totals(pw)
    out = pw.merge(totals, on=["season", "week", "team"], how="left")

    out["target_share"] = _safe_div(out["targets"], out["team_targets"])
    out["carry_share"] = _safe_div(out["rush_att"], out["team_rush_att"])
    out["opportunity"] = out["targets"] + out["rush_att"]
    out["opportunity_share"] = _safe_div(
        out["opportunity"], out["team_targets"] + out["team_rush_att"]
    )
    # WOPR (Weighted Opportunity Rating), Josh Hermsmeyer's standard weights.
    out["wopr"] = 1.5 * out["target_share"] + 0.7 * out["carry_share"]
    return out


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise num/den, returning 0 where den == 0 (no inf/NaN)."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = np.divide(
        num, den, out=np.zeros(len(num), dtype=float), where=(den.to_numpy() != 0)
    )
    return pd.Series(out, index=num.index)
