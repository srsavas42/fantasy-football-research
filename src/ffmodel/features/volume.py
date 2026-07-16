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
# the team model, so it is excluded from the receiving allocation but retained
# as a possible carry recipient.
SKILL_POSITIONS = ("RB", "WR", "TE")
TARGET_POSITIONS = SKILL_POSITIONS
CARRY_POSITIONS = ("QB", *SKILL_POSITIONS)

TEAM_TOTAL_COLUMNS = [
    "team_pass_att",
    "team_rush_att",
    "team_targets",
    "team_target_support",
    "team_plays",
    "team_pass_rate",
    "team_target_rate",
    "team_unallocated_targets",
    "team_opportunity_valid",
]

USAGE_COLUMNS = [
    "target_share",
    "carry_share",
    "opportunity",
    "opportunity_share",
    "wopr",
]


def opportunity_position(values: pd.Series) -> pd.Series:
    """Map provider-specific offensive labels onto volume-model positions.

    The legacy files use labels such as ``HB``, ``FB`` and ``WR/RS``.  Treating
    those as separate or non-skill positions discards legitimate targets and
    carries from the allocation support.  Unknown and defensive labels remain
    ``OTHER`` rather than being assigned an opportunity prior accidentally.
    """
    raw = values.astype("string").str.upper().str.strip()
    out = pd.Series("OTHER", index=values.index, dtype="string")
    out.loc[raw.str.contains("QB", na=False)] = "QB"
    out.loc[raw.str.contains("WR", na=False)] = "WR"
    out.loc[raw.str.contains("TE", na=False)] = "TE"
    out.loc[raw.str.contains(r"RB|HB|FB", regex=True, na=False)] = "RB"
    return out


def team_game_totals(pw: pd.DataFrame) -> pd.DataFrame:
    """Per-team totals plus explicit opportunity-accounting quality flags.

    ``team_targets`` remains the full recorded target total used by generic
    usage-share features. ``team_target_support`` is the RB/WR/TE total after
    provider-normalization, and is the response represented by the target
    allocator. Keeping both prevents rare unmodelled targets from breaking the
    basic invariant that player target shares sum to one.
    """
    work = pw.copy()
    work["_opportunity_position"] = opportunity_position(work["position"])
    work["_target_in_support"] = np.where(
        work["_opportunity_position"].isin(TARGET_POSITIONS),
        pd.to_numeric(work["targets"], errors="coerce").fillna(0.0),
        0.0,
    )
    grp = work.groupby(["season", "week", "team"], dropna=False)
    totals = grp.agg(
        team_pass_att=("pass_att", "sum"),
        team_rush_att=("rush_att", "sum"),
        team_target_support=("_target_in_support", "sum"),
        team_targets=("targets", "sum"),
    ).reset_index()
    totals["team_plays"] = totals["team_pass_att"] + totals["team_rush_att"]
    totals["team_pass_rate"] = _safe_div(totals["team_pass_att"], totals["team_plays"])
    totals["team_target_rate"] = _safe_div(
        totals["team_target_support"], totals["team_pass_att"]
    )
    totals["team_unallocated_targets"] = (
        totals["team_targets"] - totals["team_target_support"]
    )
    # A target is necessarily a pass attempt.  The legacy source has a small
    # number of violations caused by missing QB rows; flag rather than quietly
    # teaching the model impossible target rates.
    totals["team_opportunity_valid"] = (
        (totals["team_pass_att"] >= 0)
        & (totals["team_rush_att"] >= 0)
        & (totals["team_target_support"] >= 0)
        & (totals["team_target_support"] <= totals["team_pass_att"])
    )
    return totals


def opportunity_accounting_summary(pw: pd.DataFrame) -> dict[str, float | int]:
    """Return compact data-quality diagnostics for a player-week frame."""
    totals = team_game_totals(pw)
    target_gap = totals["team_pass_att"] - totals["team_target_support"]
    return {
        "team_weeks": int(len(totals)),
        "invalid_team_weeks": int((~totals["team_opportunity_valid"]).sum()),
        "invalid_team_week_rate": float((~totals["team_opportunity_valid"]).mean()),
        "mean_target_gap": float(target_gap.mean()),
        "mean_absolute_target_gap": float(target_gap.abs().mean()),
        "mean_unallocated_targets": float(totals["team_unallocated_targets"].mean()),
    }


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
