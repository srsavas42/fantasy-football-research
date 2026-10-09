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

#: Columns this module adds. ``left_early`` describes the row's own game and is not a
#: feature for that row's projection; the two below it are.
LABEL = "left_early"
FEATURES = ("partial_prev", "partial_recent")


def _by_player(frame: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    return frame.groupby(["player_key", "season"], sort=False)


def add_partial_game(frame: pd.DataFrame) -> pd.DataFrame:
    """Label games cut short and attach the lagged flags, on the frame's own index."""
    out = frame.copy()
    if "snap_share" not in out.columns:
        out[LABEL] = 0.0
        out["partial_prev"] = 0.0
        out["partial_recent"] = 0.0
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
    work[LABEL] = cut.astype(float)

    lab = work.groupby(["player_key", "season"], sort=False)[LABEL]
    work["partial_prev"] = lab.shift(1).fillna(0.0)
    work["partial_recent"] = lab.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).sum()
    ).fillna(0.0)

    for column in (LABEL, "partial_prev", "partial_recent"):
        out[column] = work[column].reindex(out.index).to_numpy(float)
    return out
