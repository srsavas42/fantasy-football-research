"""One athletic score per player, from RAS where supplied and the combine otherwise.

Draft slot is otherwise the only signal a rookie carries. The individual
measurables do add to it, but unevenly and in position-specific directions —
weight predicts more carries for a back and fewer targets for a tight end — so
feeding eight raw drills per position invites the model to fit noise.

A single position-normalized composite avoids that. Each drill is converted to a
percentile against the same position in earlier classes, oriented so a higher
percentile is always more athletic, and averaged over whatever the player
actually recorded. The result is put on the familiar 0-10 scale.

Relative Athletic Score does the same job with one advantage this cannot
reproduce: it incorporates pro-day testing, so it covers players the combine
never measured. It is published by a third party and is not part of nflverse,
so it is not fetched. Drop it at ``config.RAS_SCORES_PATH`` and it takes
precedence, with the composite remaining the fallback.
``athletic_score_is_ras`` records which one a row actually used.

Measured result, and a caution. On rookie seasons 2015-2024, the composite's
rank correlation with realized volume after controlling for draft pick is 0.080
for backs, 0.016 for receivers and -0.014 for tight ends: no incremental signal
over draft slot at any position. Excluding size does not rescue it. The reason
is visible in the raw measurables, where the two effects that do survive that
control point in opposite directions — weight is worth +0.187 to a back's
carries and -0.220 to a tight end's targets — so any average over position-
normalized percentiles cancels them against each other.

The composite is therefore shipped as the slot RAS will occupy and as a
degradation path, not as a validated predictor. Nothing consumes it
automatically. Real RAS may behave differently, since its pro-day coverage
reaches the untested players a combine-only composite cannot describe at all,
and that is the population worth testing it on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ffmodel.config import RAS_SCORES_PATH
from ffmodel.features.combine import COMBINE_DRILLS, COMBINE_SIZE

# Drill orientation: True where a larger number is more athletic. Times are
# inverted so every percentile points the same way. Size is included to stay
# comparable with RAS, which grades it; the raw weight column remains available
# separately for the position-specific directional signal it carries.
_ORIENTATION = {
    **{f"combine_{drill}": larger_is_better for drill, larger_is_better in COMBINE_DRILLS.items()},
    **{f"combine_{column}": True for column in COMBINE_SIZE},
}

ATHLETIC_FEATURES = (
    "athletic_score",
    "athletic_score_is_ras",
    "athletic_metrics_used",
)

_SCORE_SCALE = 10.0


def _expanding_percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    """Percentile of each value against the same position in classes up to its own.

    Scoring against every class ever recorded would let a player's percentile
    depend on athletes who had not yet been measured when they were drafted.
    """
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    seasons = pd.to_numeric(frame["season"], errors="coerce")
    for position, group in frame.groupby(frame["position"].astype(str)):
        group_values = values.loc[group.index]
        group_seasons = seasons.loc[group.index]
        for season in sorted(group_seasons.dropna().unique()):
            baseline = group_values[group_seasons.le(season)].dropna()
            target = group_seasons.eq(season) & group_values.notna()
            if baseline.empty or not target.any():
                continue
            reference = baseline.to_numpy(dtype=float)
            out.loc[group.index[target.loc[group.index]]] = [
                float((reference <= value).mean())
                for value in group_values[target.loc[group.index]]
            ]
    return out


def combine_athletic_score(features: pd.DataFrame) -> pd.DataFrame:
    """Position-normalized 0-10 composite over whatever drills were recorded."""
    if features.empty:
        return pd.DataFrame(
            columns=["season", "position", "player_id", "athletic_score", "athletic_metrics_used"]
        )
    frame = features.copy()
    percentiles = []
    for column, larger_is_better in _ORIENTATION.items():
        if column not in frame.columns:
            continue
        oriented = pd.to_numeric(frame[column], errors="coerce")
        if not larger_is_better:
            oriented = -oriented
        frame[f"_p_{column}"] = _expanding_percentile(
            frame.assign(**{column: oriented}), column
        )
        percentiles.append(f"_p_{column}")
    if not percentiles:
        frame["athletic_score"] = np.nan
        frame["athletic_metrics_used"] = 0.0
    else:
        block = frame[percentiles]
        frame["athletic_metrics_used"] = block.notna().sum(axis=1).astype(float)
        # Mean over recorded drills only: a player who ran three events is scored
        # on those three rather than penalised for the ones they skipped.
        frame["athletic_score"] = block.mean(axis=1, skipna=True) * _SCORE_SCALE
    keep = ["season", "position", "player_id", "player_name", "athletic_score", "athletic_metrics_used"]
    return frame[[column for column in keep if column in frame.columns]]


def load_ras_scores(path: Path | None = None) -> pd.DataFrame:
    """Read a hand-supplied RAS table, if one exists.

    Expected columns: ``player_name`` and ``ras``, plus any of ``season``,
    ``position``, ``gsis_id``, ``pfr_player_id``, ``espn_id`` to identify the
    player. Absence is normal — the composite covers it.
    """
    path = Path(path or RAS_SCORES_PATH)
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "ras"])
    table = pd.read_csv(path)
    table.columns = [str(column).strip().lower() for column in table.columns]
    if "ras" not in table.columns:
        raise ValueError(f"{path} must contain a 'ras' column")
    table["ras"] = pd.to_numeric(table["ras"], errors="coerce")

    from ffmodel.data.identity import is_gsis_id, resolve_player_ids

    native = table.get("gsis_id", pd.Series(pd.NA, index=table.index)).astype("string")
    table["player_id"] = native.where(is_gsis_id(native))
    try:
        bridged = resolve_player_ids(table)
    except Exception:
        bridged = pd.Series(pd.NA, index=table.index, dtype="string")
    table["player_id"] = table["player_id"].where(table["player_id"].notna(), bridged)
    return table.dropna(subset=["ras"])


def merge_athletic_score(
    rows: pd.DataFrame,
    combine_features: pd.DataFrame,
    *,
    ras: pd.DataFrame | None = None,
    key: str = "player_key",
) -> pd.DataFrame:
    """Attach one athletic score per row, preferring RAS over the composite."""
    out = rows.copy()
    for column in ATHLETIC_FEATURES:
        out[column] = np.nan
    if key not in out.columns:
        return out

    lookup = out[key].astype("string")
    composite = combine_athletic_score(combine_features)
    if not composite.empty and "player_id" in composite.columns:
        keyed = composite.dropna(subset=["player_id"]).drop_duplicates("player_id")
        keyed = keyed.set_index("player_id")
        out["athletic_score"] = lookup.map(keyed["athletic_score"]).astype(float)
        out["athletic_metrics_used"] = (
            lookup.map(keyed["athletic_metrics_used"]).astype(float).fillna(0.0)
        )
    else:
        out["athletic_metrics_used"] = 0.0
    out["athletic_score_is_ras"] = 0.0

    if ras is None:
        ras = load_ras_scores()
    if not ras.empty and "player_id" in ras.columns:
        keyed = ras.dropna(subset=["player_id"]).drop_duplicates("player_id")
        supplied = lookup.map(keyed.set_index("player_id")["ras"]).astype(float)
        # RAS wins where present: it also reflects pro-day testing, which the
        # combine feed does not carry at all.
        out["athletic_score"] = supplied.where(supplied.notna(), out["athletic_score"])
        out["athletic_score_is_ras"] = supplied.notna().astype(float)
    return out
