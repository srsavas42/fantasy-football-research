"""Games a player started and did not finish.

The panel records ``played`` as one 0/1 flag: did he have a stat line. A receiver who
is hurt at halftime and a receiver who played every snap are both ``played = 1``, so
the absence state the games model reads sees no difference between them, and the next
week looks like any other. It is not. Among skill players who took under 60% of their
usual snaps, 42% miss the next game (7% otherwise), 31% are on the next injury report
(8%), and those who do play take 0.46 of the snaps against 0.72.

``left_early`` labels the game from the snap share alone:

    played, and his snap share was under ``THRESHOLD`` x his average over the previous
    ``WINDOW`` club games, from a baseline of at least ``MIN_BASELINE``.

**Weeks 1-2 have no such average**, so a player hurt in the opener is never flagged. For
those games the baseline is what the depth chart and last season say he should play: the
median snap share of players in his position and depth slot in earlier seasons, and his
own average over the last six games of the previous season. With both, the lower of the
two is used with ``EARLY_THRESHOLD_BOTH``; with one, the stricter
``EARLY_THRESHOLD_ONE_SOURCE`` applies. Both estimates use earlier seasons only. Among
week 1-2 games, players flagged this way miss the next game 41% of the time against 4.5%
(38% against 5.7% on the depth chart alone at the stricter threshold).

It is a label of the *game that was played*, so it is only ever read lagged:
``partial_prev`` is whether the previous game was cut short, and ``partial_recent``
counts the last three. The label is noisy -- a blowout, a rotation or an ejection also
cuts snaps -- and the lag keeps it honest: nothing about the game being projected is
used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A game is cut short when his snaps fall under this share of his recent average.
THRESHOLD = 0.6

#: Below this average snap share a player has no role to cut short.
MIN_BASELINE = 0.5

#: Club games the baseline averages over, and the fewest of them it needs.
WINDOW = 4
MIN_PRIOR = 2

#: Weeks 1-2 baseline (see the module docstring). When only one of the two sources
#: exists the threshold is stricter, because one noisy estimate is easier to fall under.
EARLY_BASELINE = True
EARLY_THRESHOLD_BOTH = 0.65
EARLY_THRESHOLD_ONE_SOURCE = 0.5
PRIOR_GAMES = 6          # last games of the previous season that form his own average
DEPTH_CAP = 3            # depth slots beyond this are pooled

#: Columns this module adds. ``left_early`` describes the row's own game and is not a
#: feature for that row's projection; the two below it are.
LABEL = "left_early"
LABEL_EARLY = "left_early_depth"
FEATURES = ("partial_prev", "partial_prev_early", "partial_recent")


def _by_player(frame: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    return frame.groupby(["player_key", "season"], sort=False)


def _early_expectation(rows: pd.DataFrame, work: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """What he should have played in a game with no in-season history, and the threshold.

    Earlier seasons only, both for the depth-slot median and for his own last games, so
    the label for a given season never sees that season or a later one.
    """
    nan = pd.Series(np.nan, index=work.index)
    if "depth_rank" not in rows.columns or "position" not in rows.columns:
        depth = nan
    else:
        slot = pd.to_numeric(rows["depth_rank"], errors="coerce").clip(upper=DEPTH_CAP)
        frame = pd.DataFrame(
            {"season": work["season"], "position": rows["position"].astype(str),
             "slot": slot, "snap": work["snap"]}
        )
        depth = nan.copy()
        have = frame.dropna(subset=["snap", "slot"])
        for season in sorted(frame["season"].unique()):
            prior = have[have["season"] < season]
            if prior.empty:
                continue
            medians = prior.groupby(["position", "slot"])["snap"].median()
            here = frame["season"] == season
            keyed = pd.MultiIndex.from_arrays([frame.loc[here, "position"], frame.loc[here, "slot"]])
            depth.loc[here] = medians.reindex(keyed).to_numpy()

    last = (
        work.dropna(subset=["snap"])
        .groupby(["player_key", "season"], sort=False)["snap"]
        .apply(lambda s: s.tail(PRIOR_GAMES).mean())
    )
    last.index = pd.MultiIndex.from_arrays(
        [last.index.get_level_values(0), last.index.get_level_values(1) + 1]
    )
    keyed = pd.MultiIndex.from_arrays([work["player_key"], work["season"]])
    own = pd.Series(last.reindex(keyed).to_numpy(), index=work.index)

    both = depth.notna() & own.notna()
    # With both estimates he must be under the lower of the two; with one, under that one.
    expected = pd.Series(np.where(both, np.minimum(depth, own), depth.fillna(own)), index=work.index)
    threshold = pd.Series(
        np.where(both, EARLY_THRESHOLD_BOTH, EARLY_THRESHOLD_ONE_SOURCE), index=work.index
    )
    return expected, threshold


def add_partial_game(frame: pd.DataFrame) -> pd.DataFrame:
    """Label games cut short and attach the lagged flags, on the frame's own index."""
    out = frame.copy()
    if "snap_share" not in out.columns:
        for column in (LABEL, LABEL_EARLY, "partial_prev", "partial_prev_early", "partial_recent"):
            out[column] = 0.0
        return out

    order = out.sort_values(["player_key", "season", "week"], kind="mergesort").index
    work = out.loc[order, ["player_key", "season", "played", "snap_share"]].copy()
    played = pd.to_numeric(work["played"], errors="coerce").fillna(0.0) == 1
    snap = pd.to_numeric(work["snap_share"], errors="coerce").where(played)
    work["snap"] = snap

    grouped = work.groupby(["player_key", "season"], sort=False)["snap"]
    baseline = grouped.transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_PRIOR).mean()
    )
    cut = played & (baseline >= MIN_BASELINE) & (snap < THRESHOLD * baseline)
    cut_early = pd.Series(False, index=work.index)
    if EARLY_BASELINE:
        early_base, early_threshold = _early_expectation(out.loc[order], work)
        use = baseline.isna() & early_base.notna()
        cut_early = played & use & (early_base >= MIN_BASELINE) & (snap < early_threshold * early_base)
    work["cut_season"] = cut.astype(float)
    work[LABEL_EARLY] = cut_early.astype(float)
    work[LABEL] = (cut | cut_early).astype(float)

    key = ["player_key", "season"]
    # The two labels are read apart: a game measured against his own recent snaps is a
    # stronger sign than one measured against a depth-chart average, and the games model
    # fits them differently.
    work["partial_prev"] = work.groupby(key, sort=False)["cut_season"].shift(1).fillna(0.0)
    work["partial_prev_early"] = work.groupby(key, sort=False)[LABEL_EARLY].shift(1).fillna(0.0)
    work["partial_recent"] = work.groupby(key, sort=False)["cut_season"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).sum()
    ).fillna(0.0)

    for column in (LABEL, LABEL_EARLY, "partial_prev", "partial_prev_early", "partial_recent"):
        out[column] = work[column].reindex(out.index).to_numpy(float)
    return out
