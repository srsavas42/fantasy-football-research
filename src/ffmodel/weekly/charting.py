"""Next Gen Stats: tracking-derived efficiency, and the half of the panel it
declines to measure.

Every column here is an "over expected" metric of the same family as expected
fantasy points, and the family's first member came back a null because it was a
restatement of usage the model already read. These are not. Regressed on the
nine usage features already in the design, the tracking columns give up almost
nothing -- R² of 0.005 to 0.095, against 0.80 for expected points. Separation,
yards after catch above expectation, and completion percentage over expected are
*athlete and scheme* quantities, and the box score does not contain them.

So they are worth a ladder run, and the thing that makes it hard is not
redundancy but coverage. The league publishes tracking summaries only for
players clearing a volume threshold, which means the missingness is selection on
exactly the variable that matters:

    NGS rushing     covered rows: 16.0 carries/wk, 15.3 pts   uncovered: 4.5, 6.1
    NGS receiving   covered rows:  7.9 targets/wk, 13.7 pts   uncovered: 2.7, 7.7
    NGS passing     covered rows: 33.3 attempts/wk, 16.6 pts  uncovered: 5.1, 2.7

Half of running back weeks and nearly two thirds of receiver weeks are simply
not measured, and they are the low-volume half. The design fills a missing
feature with the training median, which here reads as "league-average efficiency
for a player nobody tracked" -- a defensible fill precisely because the volume
that determined the missingness is already a feature. An explicit `_tracked`
flag is carried alongside so the fit can price the fill rather than swallow it.

Three families, keyed to the positions they describe. A quarterback has no
receiving separation and a receiver has no time to throw; those columns arrive
all-missing for the wrong position, land on the median, and standardize to a
constant that the ridge ignores. That is the intended behaviour of fitting by
position, not a defect to be worked around.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ffmodel.data import ingest

# Only "over expected" and tracking-native columns are taken. Anything the feed
# reports that is also in the box score (yards, attempts, touchdowns) is already
# in the panel from a source with full coverage, and taking it here would import
# the coverage problem for nothing.
CHARTING_FAMILIES = {
    "rushing": (
        "rush_yards_over_expected_per_att",
        "rush_pct_over_expected",
        "efficiency",
    ),
    "receiving": (
        "avg_yac_above_expectation",
        "avg_expected_yac",
        "avg_separation",
    ),
    "passing": (
        "completion_percentage_above_expectation",
        "expected_completion_percentage",
        "avg_time_to_throw",
    ),
}

CHARTING_COLUMNS = tuple(c for cols in CHARTING_FAMILIES.values() for c in cols)
TRACKED_COLUMNS = tuple(f"{family}_tracked" for family in CHARTING_FAMILIES)


def load_charting(seasons: Iterable[int]) -> pd.DataFrame:
    """Tracking summaries per player-week, keyed like the panel.

    Week 0 rows are the feed's season-to-date summaries rather than a week, and
    including them would put a season average on the panel's week zero -- which
    does not exist -- or worse, merge cleanly onto nothing and look fine.
    """
    seasons = sorted({int(s) for s in seasons})
    frames = []
    for family, columns in CHARTING_FAMILIES.items():
        try:
            raw = ingest.load_nextgen_stats(seasons, stat_type=family)
        except Exception:
            continue
        if raw.empty or "player_gsis_id" not in raw.columns:
            continue
        present = [c for c in columns if c in raw.columns]
        if not present:
            continue
        rows = raw[raw["week"] > 0]
        out = rows[["season", "week", "player_gsis_id", *present]].rename(
            columns={"player_gsis_id": "player_key"}
        )
        out = out.drop_duplicates(subset=["season", "week", "player_key"])
        out[f"{family}_tracked"] = 1.0
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["season", "week", "player_key"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["season", "week", "player_key"], how="outer")
    merged["season"] = merged["season"].astype(int)
    merged["week"] = merged["week"].astype(int)
    merged["player_key"] = merged["player_key"].astype(str)
    return merged


def attach_charting(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach tracking columns to a weekly panel.

    A week with no tracking row is left missing rather than zeroed. Zero
    separation and zero completion percentage over expected are both real,
    extreme values in these units, and the honest-zero convention the panel uses
    for counting stats is wrong for every column in this file.
    """
    frame = panel.copy()
    charting = load_charting(sorted(frame["season"].unique().tolist()))
    if charting.empty:
        return frame
    frame = frame.merge(charting, on=["season", "week", "player_key"], how="left")
    for column in TRACKED_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame
