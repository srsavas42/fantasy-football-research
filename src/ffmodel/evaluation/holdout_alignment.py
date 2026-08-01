"""Align projected players to realized outcomes without dropping the failures.

A holdout evaluation that inner-joins projections to realized stat rows silently
discards every projected player who never recorded one. Those are not missing
observations: a player projected onto a roster who produced nothing produced
zero, and that is the outcome the projection got wrong. Excluding them grades
the model only on players whose role materialised, which flatters exactly the
role-collapse behaviour a preseason projection most needs to get right.

Synthetic replacement buckets are the one class that must still be dropped. They
are a modelling device standing in for volume earned by players absent from the
point-in-time roster, not people, so they have no realized counterpart to score.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

REPLACEMENT_FLAGS = ("is_replacement_player", "is_replacement_qb")


def real_player_mask(rows: pd.DataFrame) -> np.ndarray:
    """Rows corresponding to actual players rather than replacement buckets."""
    mask = np.ones(len(rows), dtype=bool)
    for flag in REPLACEMENT_FLAGS:
        if flag in rows.columns:
            values = pd.to_numeric(rows[flag], errors="coerce").fillna(0.0).to_numpy()
            mask &= values <= 0
    return mask


def align_projection_to_outcomes(
    projected_rows: pd.DataFrame,
    realized: pd.DataFrame,
    *,
    outcome: str = "actual",
    keys: Sequence[str] = ("player_key", "team"),
) -> pd.DataFrame:
    """Every real projected player, with an outcome and its origin.

    Returns one row per real projected player carrying ``sample_index`` (the
    position of that player in the posterior sample matrix), the outcome, and
    ``realized_row`` recording whether the outcome was observed or imputed as a
    zero. Keeping that flag lets a caller report how much of a result rests on
    players who never appeared.
    """
    keys = list(keys)
    missing = [key for key in keys if key not in projected_rows.columns]
    if missing:
        raise ValueError(f"projected rows are missing join keys: {missing}")
    if outcome not in realized.columns:
        raise ValueError(f"realized rows must contain an '{outcome}' column")

    frame = projected_rows.reset_index(drop=True)
    frame = frame.assign(sample_index=np.arange(len(frame)))
    frame = frame[real_player_mask(frame)].copy()

    merged = frame.merge(
        realized[keys + [outcome]].drop_duplicates(keys), on=keys, how="left"
    )
    merged["realized_row"] = merged[outcome].notna()
    # No stat row means no production, which is a zero rather than a gap.
    merged[outcome] = pd.to_numeric(merged[outcome], errors="coerce").fillna(0.0)
    return merged.reset_index(drop=True)
