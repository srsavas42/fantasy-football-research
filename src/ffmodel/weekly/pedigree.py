"""What a player was before he played: draft capital and career stage.

Every other input to this layer is a transform of something that happened on a
field, plus the draft board and two pre-game reports. Draft capital is neither.
It is a statement clubs made about a player before he took a snap, and it keeps
predicting opportunity long after — a first-round back gets carried through a bad
month in a way an undrafted one does not, and a box score records the consequence
rather than the cause.

This is also the cheapest test of a larger question: whether the season-average
pipeline's projections would be worth feeding in here. That pipeline's genuinely
distinctive inputs over this one are draft capital, combine athleticism, coaching
continuity and preseason win totals. Its *output* is expensive — the posteriors
are build artefacts and refitting three folds is an hour of sampling — and it is
a preseason quantity competing with ADP, which this layer already reads. Adding
the exogenous inputs directly answers most of the question for a fraction of the
cost: if the information behind the projection does not help, the projection
built from it is unlikely to.

Two fields, both static per player:

``draft_round`` / ``draft_overall``
    Where he was taken. Undrafted is a value rather than a gap — placed one past
    the last pick and flagged — for the same reason the ADP encoding does it:
    imputing a median draft slot to a player nobody drafted asserts the opposite
    of what his absence says.

``years_exp``
    Career stage, read from the weekly roster. A rookie and a ninth-year veteran
    with identical recent usage are not the same forecasting problem, and nothing
    else in the feature set separates them.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ffmodel.data import ingest

PEDIGREE_COLUMNS = (
    "draft_round",
    "draft_log_overall",
    "undrafted",
    "years_exp",
)

# One past the last pick of a seven-round draft. Undrafted players are placed
# here rather than left missing.
UNDRAFTED_OVERALL = 263.0
UNDRAFTED_ROUND = 8.0


def load_draft_capital(seasons: Iterable[int]) -> pd.DataFrame:
    """Round and overall pick per player, keyed like the panel.

    Draft classes are pulled for every season back to the earliest one a player
    in the window could plausibly have entered, because a 2016 panel row can
    belong to a player drafted in 2005.
    """
    seasons = sorted({int(s) for s in seasons})
    if not seasons:
        return pd.DataFrame(columns=["player_key", "draft_round", "draft_overall"])
    classes = range(min(seasons) - 22, max(seasons) + 1)
    try:
        picks = ingest.load_draft_picks(list(classes))
    except Exception:
        return pd.DataFrame(columns=["player_key", "draft_round", "draft_overall"])
    if picks.empty or "gsis_id" not in picks.columns:
        return pd.DataFrame(columns=["player_key", "draft_round", "draft_overall"])
    out = pd.DataFrame(
        {
            "player_key": picks["gsis_id"].astype(str),
            "draft_round": pd.to_numeric(picks.get("round"), errors="coerce"),
            "draft_overall": pd.to_numeric(picks.get("pick"), errors="coerce"),
        }
    ).dropna(subset=["player_key"])
    out = out[out["player_key"].ne("nan")]
    # A player appears in exactly one draft; keep the earliest if the feed
    # disagrees with itself rather than averaging two picks together.
    return out.sort_values("draft_overall").drop_duplicates("player_key", keep="first")


def load_experience(seasons: Iterable[int]) -> pd.DataFrame:
    """Years of NFL experience per player-season, from the weekly roster."""
    try:
        rosters = ingest.load_weekly_rosters(sorted({int(s) for s in seasons}))
    except Exception:
        return pd.DataFrame(columns=["season", "player_key", "years_exp"])
    if rosters.empty or "years_exp" not in rosters.columns:
        return pd.DataFrame(columns=["season", "player_key", "years_exp"])
    out = pd.DataFrame(
        {
            "season": pd.to_numeric(rosters["season"], errors="coerce"),
            "player_key": rosters["gsis_id"].astype(str),
            "years_exp": pd.to_numeric(rosters["years_exp"], errors="coerce"),
        }
    ).dropna(subset=["season", "player_key"])
    out["season"] = out["season"].astype(int)
    return out.groupby(["season", "player_key"], as_index=False)["years_exp"].max()


def add_pedigree_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach draft capital and career stage to a weekly panel."""
    seasons = sorted(panel["season"].unique().tolist())
    frame = panel.copy()

    capital = load_draft_capital(seasons)
    if capital.empty:
        frame["draft_round"] = UNDRAFTED_ROUND
        frame["draft_overall"] = UNDRAFTED_OVERALL
        frame["undrafted"] = 1.0
    else:
        frame = frame.merge(capital, on="player_key", how="left")
        frame["undrafted"] = frame["draft_overall"].isna().astype(float)
        frame["draft_round"] = frame["draft_round"].fillna(UNDRAFTED_ROUND)
        frame["draft_overall"] = frame["draft_overall"].fillna(UNDRAFTED_OVERALL)
    frame["draft_log_overall"] = np.log(
        frame["draft_overall"].clip(lower=1.0).to_numpy(float)
    )

    experience = load_experience(seasons)
    if experience.empty:
        frame["years_exp"] = np.nan
    else:
        frame = frame.merge(experience, on=["season", "player_key"], how="left")
    return frame
