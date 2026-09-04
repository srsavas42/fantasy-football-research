"""Convert stat lines to fantasy points.

Skill-position weights live in config.ScoringRules and were verified to
reproduce the legacy CSVs' StandardFantasyPoints / HalfPPRFantasyPoints /
PPRFantasyPoints exactly. Kickers and team defenses have their own rules
(config.KickerRules, config.DefenseRules) because their scoring is tiered
rather than linear in a stat line.
"""

from __future__ import annotations

import pandas as pd

from ffmodel.config import (
    DEFENSE_FORMATS,
    KICKER_FORMATS,
    SCORING_FORMATS,
    DefenseRules,
    KickerRules,
    ScoringRules,
)


def fantasy_points(df: pd.DataFrame, rules: ScoringRules | str = "ppr") -> pd.Series:
    """Fantasy points for each row of a canonical-schema stat frame."""
    if isinstance(rules, str):
        rules = SCORING_FORMATS[rules]
    return (
        df["pass_yds"] * rules.pass_yd
        + df["pass_td"] * rules.pass_td
        + df["pass_int"] * rules.interception
        + df["rush_yds"] * rules.rush_yd
        + df["rush_td"] * rules.rush_td
        + df["rec_yds"] * rules.rec_yd
        + df["rec_td"] * rules.rec_td
        + df["receptions"] * rules.reception
        + df["fumbles_lost"] * rules.fumble_lost
    )


def kicker_points(df: pd.DataFrame, rules: KickerRules | str = "standard") -> pd.Series:
    """Fantasy points for each row of a kicker stat frame.

    Expects the distance-bucketed nflverse columns. The 0-19 and 20-29 buckets
    are folded into the 0-39 tier because no mainstream scoring separates them,
    and a missed extra point is scored apart from a missed field goal because
    most leagues price them differently even when both are negative.
    """
    if isinstance(rules, str):
        rules = KICKER_FORMATS[rules]

    def column(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)

    short = column("fg_made_0_19") + column("fg_made_20_29") + column("fg_made_30_39")
    medium = column("fg_made_40_49")
    long = column("fg_made_50_59") + column("fg_made_60_")
    # A blocked kick is a miss for scoring: the points did not go up.
    misses = column("fg_missed") + column("fg_blocked")
    pat_misses = column("pat_missed") + column("pat_blocked")

    return (
        short * rules.fg_0_39
        + medium * rules.fg_40_49
        + long * rules.fg_50_plus
        + misses * rules.fg_miss
        + column("pat_made") * rules.pat_made
        + pat_misses * rules.pat_miss
    )


def points_allowed_score(
    points_allowed, rules: DefenseRules | str = "standard"
) -> pd.Series:
    """The points-allowed step function, evaluated on a series of opponent scores."""
    if isinstance(rules, str):
        rules = DEFENSE_FORMATS[rules]
    allowed = pd.to_numeric(pd.Series(points_allowed), errors="coerce")
    out = pd.Series(rules.points_allowed_worst, index=allowed.index, dtype=float)
    # Walk the tiers from the top down so the tightest bound that a score
    # satisfies is the one that ends up applied.
    for bound, value in sorted(rules.points_allowed_tiers, reverse=True):
        out = out.mask(allowed <= bound, value)
    return out.mask(allowed.isna())


def defense_points(df: pd.DataFrame, rules: DefenseRules | str = "standard") -> pd.Series:
    """Fantasy points for each row of a team defense/special-teams frame.

    ``points_allowed`` is the opponent's final score and comes from the schedule
    rather than from the defence's own box score. Return and defensive
    touchdowns are summed together: leagues that award them separately still
    award them at the same rate, and nflverse reports them in two columns.
    """
    if isinstance(rules, str):
        rules = DEFENSE_FORMATS[rules]

    def column(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)

    touchdowns = column("def_tds") + column("special_teams_tds")
    blocks = column("def_punt_blocks") + column("def_pat_blocks") + column("def_fg_blocks")

    events = (
        column("def_sacks") * rules.sack
        + column("def_interceptions") * rules.interception
        + column("fumble_recovery_opp") * rules.fumble_recovery
        + touchdowns * rules.touchdown
        + column("def_safeties") * rules.safety
        + blocks * rules.block
    )
    return events + points_allowed_score(df.get("points_allowed"), rules).to_numpy()
