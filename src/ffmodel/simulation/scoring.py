"""Convert stat lines to fantasy points.

Weights live in config.ScoringRules and were verified to reproduce the legacy
CSVs' StandardFantasyPoints / HalfPPRFantasyPoints / PPRFantasyPoints exactly.
"""

from __future__ import annotations

import pandas as pd

from ffmodel.config import SCORING_FORMATS, ScoringRules


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
