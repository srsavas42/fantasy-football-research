"""Leak-free multi-season state features for season-average pathways."""

from __future__ import annotations

import numpy as np
import pandas as pd


HISTORY_ALPHA = 0.50

PRODUCTION_HISTORY_INPUTS = (
    "prior_availability",
    "prior_snap_share",
    "prior_pass_role",
    "prior_target_role",
    "prior_carry_role",
    "prior_qb_attempts_per_snap",
    "prior_target_per_snap",
    "prior_carry_per_snap",
)

EFFICIENCY_HISTORY_INPUTS = (
    "prior_pass_quality_signal",
    "prior_pass_td_rate",
    "prior_rec_quality_signal",
    "prior_rush_quality_signal",
    "prior_rush_epa_per_carry",
)

PLAYER_PATHWAY_FEATURES = (
    *(
        feature
        for name in (*PRODUCTION_HISTORY_INPUTS, *EFFICIENCY_HISTORY_INPUTS)
        for feature in (f"{name}_3yr", f"{name}_trend")
    ),
    "prior_role_quality_signal",
    "prior_role_quality_signal_3yr",
    "prior_role_quality_signal_trend",
    "prior_pass_room_quality_advantage",
    "prior_rec_room_quality_advantage",
    "prior_rush_room_quality_advantage",
    "prior_role_room_quality_advantage",
)


def _numeric(rows: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return pd.to_numeric(
        rows.get(name, pd.Series(default, index=rows.index)), errors="coerce"
    )


def _add_history_columns(
    rows: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    groups: tuple[str, ...],
) -> pd.DataFrame:
    out = rows.copy()
    out["_history_order"] = np.arange(len(out))
    out = out.sort_values([*groups, "season", "_history_order"])
    groupers = [out[name] for name in groups]
    previous_season = out.groupby(list(groups), dropna=False)["season"].shift(1)
    consecutive = out["season"].sub(previous_season).eq(1)
    for name in columns:
        values = _numeric(out, name)
        out[f"{name}_3yr"] = values.groupby(groupers, dropna=False).transform(
            lambda series: series.ewm(
                alpha=HISTORY_ALPHA,
                adjust=True,
                ignore_na=True,
                min_periods=1,
            ).mean()
        )
        previous = values.groupby(groupers, dropna=False).shift(1)
        out[f"{name}_trend"] = (values - previous).where(consecutive)
    return out.sort_values("_history_order").drop(columns="_history_order")


def _room_advantage(
    rows: pd.DataFrame,
    signal: str,
    weight: str,
) -> pd.Series:
    quality = _numeric(rows, signal)
    role = _numeric(rows, weight).where(lambda values: values > 0)
    valid = quality.notna() & role.notna()
    weighted = (quality * role).where(valid, 0.0)
    support = role.where(valid, 0.0)
    groupers = [rows["season"], rows["team"], rows["position"]]
    weighted_total = weighted.groupby(groupers, dropna=False).transform("sum")
    support_total = support.groupby(groupers, dropna=False).transform("sum")
    other_support = support_total - support
    other_quality = (weighted_total - weighted).div(other_support.where(other_support > 0))
    return (quality - other_quality).where(valid & other_support.gt(0))


# Trend features a model actually regresses on. A trend is a difference against
# the player's previous season, so it only exists where that row is in the same
# frame. If one of these comes out wholly missing the caller has featurized a
# projection season on its own, and the consuming model will quietly fill every
# row with a training median — fitting on a varying feature and serving a
# constant.
CONSUMED_TREND_FEATURES = ("prior_snap_share_trend",)


def add_player_pathway_features(
    rows: pd.DataFrame, *, require_trends: bool = True
) -> pd.DataFrame:
    """Add multi-year production and relative-efficiency states.

    Input columns are already lagged to the response season, so every derived
    value for season ``Y`` uses information observed no later than ``Y-1``.

    Raises if a trend a model consumes is missing for every row, which is what
    happens when a projection season is featurized alone. ``build_season_average_data``
    builds history and the projection together and then filters, so the
    production path is unaffected; the guard exists for any other caller, where
    the failure is otherwise silent and looks like a well-behaved model.
    ``require_trends=False`` opts out for callers that do not consume them.
    """
    out = rows.copy().reset_index(drop=True)
    out = _add_history_columns(
        out,
        PRODUCTION_HISTORY_INPUTS + EFFICIENCY_HISTORY_INPUTS,
        groups=("player_key",),
    )

    rec = _numeric(out, "prior_rec_quality_signal")
    rush = _numeric(out, "prior_rush_quality_signal")
    role_quality = rec.copy()
    quarterback = out["position"].eq("QB")
    role_quality.loc[quarterback] = _numeric(
        out, "prior_pass_quality_signal"
    ).loc[quarterback]
    running_back = out["position"].eq("RB")
    role_quality.loc[running_back] = pd.concat(
        [rec[running_back], rush[running_back]], axis=1
    ).mean(axis=1, skipna=True)
    out["prior_role_quality_signal"] = role_quality
    out = _add_history_columns(
        out, ("prior_role_quality_signal",), groups=("player_key",)
    )

    relative_specs = {
        "pass": ("prior_pass_quality_signal", "prior_pass_role"),
        "rec": ("prior_rec_quality_signal", "prior_target_role"),
        "rush": ("prior_rush_quality_signal", "prior_carry_role"),
        "role": ("prior_role_quality_signal", "prior_snap_share"),
    }
    for stream, (signal, weight) in relative_specs.items():
        out[f"prior_{stream}_room_quality_advantage"] = _room_advantage(
            out, signal, weight
        )
    if require_trends and len(out):
        absent = [
            name
            for name in CONSUMED_TREND_FEATURES
            if name in out and out[name].isna().all()
        ]
        if absent:
            seasons = out["season"].nunique() if "season" in out else 0
            raise ValueError(
                f"{', '.join(absent)} is missing for every row. A trend needs the "
                f"player's previous season in the same frame, and this frame holds "
                f"{seasons} season(s). Featurize history together with the "
                "projection season and filter afterwards, or pass "
                "require_trends=False if no consumer reads these."
            )
    return out
