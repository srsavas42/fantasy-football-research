"""Expected fantasy points: what the opportunities were worth, before the bounce.

nflverse `ff_opportunity` prices every play from its context — down, distance,
field position, air yards — and reports what a player's week *should* have been
worth alongside what it was. The difference is the luck term: two red-zone
targets that both fall incomplete and two that both score are the same
opportunity and eleven points apart.

This layer has a specific reason to want it that the season layer did not. The
decay on player history was selected at a **one-game half-life**, which means the
single most heavily weighted input to a projection is what happened last Sunday —
and last Sunday's points are mostly touchdown variance. Expected points are the
same signal with that variance removed, which is exactly the substitution a
one-game window makes valuable and a four-game window makes redundant.

**The season layer already tested this and it failed there**, which is why the
question is worth re-asking rather than assuming. On 2,047 consecutive
player-season pairs, prior actual points per game predicted next season better
than prior expected points at every position, and adding expected on top of
actual moved MAE by +0.12%. That is a statement about a *year-long* average,
where a season of touchdowns is already most of the way to its own expectation.
Over one week it is not.

Three columns are attached, and the third is the one carrying the hypothesis:

``points_exp``
    Expected PPR points for the week, from opportunity alone.

``points_luck``
    Actual minus expected. Positive is a week that scored above what the
    opportunities were worth.

Both are lagged and recency-weighted by the feature layer exactly like actual
points, so the model sees them on the same footing and can weigh them against
each other rather than being told which to trust.

**The hypothesis was wrong, and this module is kept for the record rather than
because it ships.** Expected points really are the better single column: they
correlate 0.365 with next week against 0.331 for actual points. Conditioned on
the usage features the model already reads, the advantage reverses -- usage
explains 80% of expected points against 46% of actual, and the residual
correlation falls to 0.071, below actual points' own 0.126. The feed prices
opportunities and this model already reads the opportunities. Run
`scripts/probe_over_expected.py` for the numbers; see
`docs/weekly-modeling-2026-08.md` for the ladder.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ffmodel.data import ingest

EXPECTED_COLUMNS = ("points_exp", "points_luck")

# The feed reports expected points on its own PPR-style basis. It is used as a
# predictor rather than a response, so it needs to be on a stable scale rather
# than exactly reconciled with this package's scoring rules.
SOURCE_COLUMN = "total_fantasy_points_exp"


def load_expected_points(seasons: Iterable[int]) -> pd.DataFrame:
    """Expected fantasy points per player-week, keyed like the panel."""
    seasons = sorted({int(s) for s in seasons})
    try:
        raw = ingest.load_ff_opportunity(seasons)
    except Exception:
        return pd.DataFrame(columns=["season", "week", "player_key", "points_exp"])
    if raw.empty or SOURCE_COLUMN not in raw.columns:
        return pd.DataFrame(columns=["season", "week", "player_key", "points_exp"])
    out = pd.DataFrame(
        {
            "season": pd.to_numeric(raw.get("season"), errors="coerce"),
            "week": pd.to_numeric(raw.get("week"), errors="coerce"),
            "player_key": raw["player_id"].astype(str),
            "points_exp": pd.to_numeric(raw[SOURCE_COLUMN], errors="coerce"),
        }
    ).dropna(subset=["season", "week", "player_key"])
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    # A player can appear once per game; a mid-season move gives two rows in a
    # week, and the opportunities add rather than average.
    return out.groupby(["season", "week", "player_key"], as_index=False)[
        "points_exp"
    ].sum()


def attach_expected(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach expected points and the luck term to a weekly panel.

    A week a player did not play is zero expected points, not an unknown number
    of them -- the same convention the panel already uses for actual points, and
    the reason it has to be stated is that the feed simply has no row there.
    """
    seasons = sorted(panel["season"].unique().tolist())
    expected = load_expected_points(seasons)
    frame = panel.copy()
    if expected.empty:
        frame["points_exp"] = np.nan
    else:
        frame = frame.merge(
            expected, on=["season", "week", "player_key"], how="left"
        )
    played = pd.to_numeric(frame["played"], errors="coerce").fillna(0)
    frame["points_exp"] = np.where(
        played.eq(0), 0.0, pd.to_numeric(frame["points_exp"], errors="coerce")
    )
    frame["points_luck"] = (
        pd.to_numeric(frame["points"], errors="coerce") - frame["points_exp"]
    )
    return frame
