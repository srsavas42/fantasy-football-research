"""Where a back's carries come from, as a weekly count the panel can aggregate.

The efficiency layer predicts yards per carry from a player's own prior rate,
his age, snaps and team change, and nothing about the *situation* he runs in. A
back given third-and-one and goal-line work has a structurally low yards per
carry, and a back running first-and-ten between the twenties a high one, and the
layer is handed both as the same statistic and asked to tell them apart.

The season panel is assembled from ``load_player_stats``, which counts carries
and yards but not the down and distance they happened on. That lives only in
play-by-play, so this module is the one place the season-average path reads it.

Measured by ``scripts/screen_situational_usage.py`` and validated by
``scripts/validate_short_yardage.py``: the share of a back's carries with two or
fewer yards to go, taken from the prior season, is worth -1.12% MAE and -1.03%
CRPS on yards per carry across 2023/2024/2025, unanimously, and -5.60% MAE on
the backs who actually run in short yardage against +0.46% on the backs who do
not. A covariate that names a role should help the players who hold it and leave
the rest alone, and this one does.

Only that one cut is built. Goal-line share was screened alongside it and left
out on purpose: carries inside the five persist at 2.2% year over year and
goal-to-go carries at 1.8%, against short-yardage share's 16.3%. Goal-line work
is handed out by a season's circumstances rather than held as a trait, so last
season's cannot forecast this season's touchdowns however real the within-season
effect is -- the descriptive correlation against next-season rushing touchdown
rate was +0.022 at p = 0.47.

Coverage is a real constraint and is left as missing rather than filled. Seasons
whose play-by-play the feed will not serve produce no rows here, the ratio comes
out missing, and the design matrix's median fill plus its missingness indicator
handle it -- which is the correct behaviour, because a zero would say the back
never ran in short yardage rather than that nobody looked.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# Two yards or fewer to go. The cut is not tuned: it is the conventional
# definition of short yardage, and picking it from the same data it is scored on
# is how a feature earns a fold win by chance.
SHORT_YARDAGE_DISTANCE = 2

CARRY_CONTEXT_COLUMNS = ("rush_short_yardage_att",)


def weekly_carry_context(
    seasons: Iterable[int], *, cache_dir=None
) -> pd.DataFrame:
    """Per player-week short-yardage carry counts from play-by-play.

    Returns an empty frame with the right columns when no season can be served,
    so a caller can merge unconditionally.
    """
    frames = []
    for season in sorted({int(s) for s in seasons}):
        pbp = _load_pbp(season, cache_dir=cache_dir)
        if pbp is None or pbp.empty:
            continue
        frames.append(_season_carry_context(pbp, season))
    if not frames:
        return pd.DataFrame(
            columns=["season", "week", "player_id", *CARRY_CONTEXT_COLUMNS]
        )
    return pd.concat(frames, ignore_index=True)


def _load_pbp(season: int, *, cache_dir=None) -> pd.DataFrame | None:
    from ffmodel.data import ingest

    try:
        return ingest.load_pbp([season], cache_dir=cache_dir)
    except Exception:
        # A season the feed will not serve is missing coverage, not a failure
        # that should stop a build. The column comes out NaN and the design
        # matrix flags it.
        return None


def _season_carry_context(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    needed = {"rush_attempt", "rusher_player_id", "ydstogo", "week"}
    if not needed.issubset(pbp.columns):
        missing = sorted(needed - set(pbp.columns))
        raise ValueError(f"{season} play-by-play is missing columns: {missing}")
    if "season_type" in pbp:
        pbp = pbp[pbp["season_type"].astype(str).eq("REG")]
    run = (
        pd.to_numeric(pbp["rush_attempt"], errors="coerce").eq(1)
        & pbp["rusher_player_id"].notna()
    )
    distance = pd.to_numeric(pbp.loc[run, "ydstogo"], errors="coerce")
    frame = pd.DataFrame({
        "season": season,
        "week": pd.to_numeric(pbp.loc[run, "week"], errors="coerce"),
        "player_id": pbp.loc[run, "rusher_player_id"].astype(str),
        "rush_short_yardage_att": distance.le(SHORT_YARDAGE_DISTANCE).astype(float),
    })
    frame = frame[frame["week"].notna()]
    return (
        frame.groupby(["season", "week", "player_id"], as_index=False)[
            "rush_short_yardage_att"
        ]
        .sum()
        .astype({"week": int})
    )


def merge_carry_context(
    player_weeks: pd.DataFrame, context: pd.DataFrame
) -> pd.DataFrame:
    """Attach the carry-context counts to a player-week panel.

    A player-week the context does not cover keeps ``NaN`` -- see the module
    docstring on why that is not zero.
    """
    out = player_weeks.copy()
    for column in CARRY_CONTEXT_COLUMNS:
        if column in out:
            out = out.drop(columns=[column])
    if context is None or context.empty or "player_id" not in out:
        for column in CARRY_CONTEXT_COLUMNS:
            out[column] = np.nan
        return out
    keys = ["season", "week", "player_id"]
    joined = context.copy()
    joined["player_id"] = joined["player_id"].astype(str)
    out["_join_player_id"] = out["player_id"].astype(str)
    joined = joined.rename(columns={"player_id": "_join_player_id"})
    for key in ("season", "week"):
        out[key] = pd.to_numeric(out[key], errors="coerce")
        joined[key] = pd.to_numeric(joined[key], errors="coerce")
    out = out.merge(
        joined, on=["season", "week", "_join_player_id"], how="left"
    ).drop(columns=["_join_player_id"])
    # A week the context covers but where the back took no short-yardage carry
    # is a real zero; only weeks outside its coverage stay missing.
    covered = out["season"].isin(set(pd.to_numeric(context["season"], errors="coerce")))
    for column in CARRY_CONTEXT_COLUMNS:
        out.loc[covered & out[column].isna(), column] = 0.0
    return out
