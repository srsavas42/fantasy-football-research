"""Walk-forward screens for indirect efficiency-to-volume pathways.

These deterministic models mirror the responses used by the production
playing-time stack. They are feature gates, not replacements for the Bayesian
models: a stable winner must still transfer to the matching posterior layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ffmodel.data.wikipedia_coaching import team_identity
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.features.season_pathways import (
    add_player_pathway_features as _canonical_player_pathway_features,
)
from ffmodel.models.season_availability import AVAILABILITY_FEATURES
from ffmodel.models.season_opportunity import (
    CARRY_ELIGIBILITY_FEATURES,
    QB_PROPENSITY_FEATURES,
    SNAP_FEATURES,
)


HISTORY_ALPHA = 0.50
MODEL_CONFIGURATIONS = (
    "baseline",
    "production_history",
    "efficiency_1yr",
    "efficiency_state",
    "combined",
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


def add_player_pathway_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper around the production pathway feature builder."""
    return _canonical_player_pathway_features(rows)


@dataclass(frozen=True)
class _Task:
    name: str
    response: str
    valid: str
    kind: str
    base: tuple[str, ...]
    production: tuple[str, ...]
    efficiency_1yr: tuple[str, ...]
    efficiency_state: tuple[str, ...]
    weight: str | None = None


ROLE_BASE = tuple(dict.fromkeys((*AVAILABILITY_FEATURES, *SNAP_FEATURES)))
TARGET_BASE = (
    "prior_target_per_snap",
    "prior_target_role",
    "prior_snap_share",
    "prior_availability",
    "age",
    "experience",
    "team_change",
    "cold_start",
    "roster_active",
    "roster_reserve",
    "depth_rank",
)
CARRY_RATE_BASE = tuple(
    dict.fromkeys((*CARRY_ELIGIBILITY_FEATURES, "prior_availability"))
)

PLAYER_TASKS = (
    _Task(
        "availability",
        "observed_availability",
        "_valid_availability",
        "continuous",
        AVAILABILITY_FEATURES,
        (
            "prior_availability_3yr",
            "prior_availability_trend",
            "prior_snap_share_3yr",
            "prior_snap_share_trend",
        ),
        ("prior_role_quality_signal",),
        (
            "prior_role_quality_signal",
            "prior_role_quality_signal_3yr",
            "prior_role_quality_signal_trend",
            "prior_role_room_quality_advantage",
        ),
        "team_games",
    ),
    _Task(
        "conditional_snap_share",
        "_conditional_snap_share",
        "_valid_snap",
        "continuous",
        SNAP_FEATURES,
        (
            "prior_snap_share_3yr",
            "prior_snap_share_trend",
            "prior_availability_3yr",
        ),
        ("prior_role_quality_signal",),
        (
            "prior_role_quality_signal",
            "prior_role_quality_signal_3yr",
            "prior_role_quality_signal_trend",
            "prior_role_room_quality_advantage",
        ),
        "offense_snaps",
    ),
    _Task(
        "material_role_retention",
        "_material_role",
        "_valid_retention",
        "binary",
        ROLE_BASE,
        (
            "prior_snap_share_3yr",
            "prior_snap_share_trend",
            "prior_availability_3yr",
        ),
        ("prior_role_quality_signal",),
        (
            "prior_role_quality_signal",
            "prior_role_quality_signal_3yr",
            "prior_role_quality_signal_trend",
            "prior_role_room_quality_advantage",
        ),
    ),
    _Task(
        "material_role_gain",
        "_material_role",
        "_valid_role_gain",
        "binary",
        ROLE_BASE,
        (
            "prior_snap_share_3yr",
            "prior_snap_share_trend",
            "prior_availability_3yr",
        ),
        ("prior_role_quality_signal",),
        (
            "prior_role_quality_signal",
            "prior_role_quality_signal_3yr",
            "prior_role_quality_signal_trend",
            "prior_role_room_quality_advantage",
        ),
    ),
    _Task(
        "target_role_1pg",
        "_target_role_1pg",
        "_valid_target_role",
        "binary",
        TARGET_BASE,
        (
            "prior_target_per_snap_3yr",
            "prior_target_per_snap_trend",
            "prior_target_role_3yr",
            "prior_target_role_trend",
        ),
        ("prior_rec_quality_signal",),
        (
            "prior_rec_quality_signal",
            "prior_rec_quality_signal_3yr",
            "prior_rec_quality_signal_trend",
            "prior_rec_room_quality_advantage",
        ),
    ),
    _Task(
        "carry_eligibility",
        "_carry_eligible",
        "_valid_carry_eligibility",
        "binary",
        CARRY_ELIGIBILITY_FEATURES,
        (
            "prior_carry_per_snap_3yr",
            "prior_carry_per_snap_trend",
            "prior_carry_role_3yr",
            "prior_carry_role_trend",
        ),
        ("prior_rush_epa_per_carry",),
        (
            "prior_rush_epa_per_carry",
            "prior_rush_epa_per_carry_3yr",
            "prior_rush_epa_per_carry_trend",
            "prior_rush_room_quality_advantage",
        ),
    ),
    _Task(
        "targets_per_snap",
        "_targets_per_snap",
        "_valid_target_rate",
        "continuous",
        TARGET_BASE,
        (
            "prior_target_per_snap_3yr",
            "prior_target_per_snap_trend",
            "prior_target_role_3yr",
        ),
        ("prior_rec_quality_signal",),
        (
            "prior_rec_quality_signal",
            "prior_rec_quality_signal_3yr",
            "prior_rec_quality_signal_trend",
            "prior_rec_room_quality_advantage",
        ),
        "offense_snaps",
    ),
    _Task(
        "carries_per_snap",
        "_carries_per_snap",
        "_valid_carry_rate",
        "continuous",
        CARRY_RATE_BASE,
        (
            "prior_carry_per_snap_3yr",
            "prior_carry_per_snap_trend",
            "prior_carry_role_3yr",
        ),
        ("prior_rush_epa_per_carry",),
        (
            "prior_rush_epa_per_carry",
            "prior_rush_epa_per_carry_3yr",
            "prior_rush_epa_per_carry_trend",
            "prior_rush_room_quality_advantage",
        ),
        "offense_snaps",
    ),
    _Task(
        "qb_attempts_per_snap",
        "_qb_attempts_per_snap",
        "_valid_qb_rate",
        "continuous",
        QB_PROPENSITY_FEATURES,
        (
            "prior_qb_attempts_per_snap_3yr",
            "prior_qb_attempts_per_snap_trend",
            "prior_pass_role_3yr",
        ),
        ("prior_pass_quality_signal", "prior_pass_td_rate"),
        (
            "prior_pass_quality_signal",
            "prior_pass_td_rate",
            "prior_pass_quality_signal_3yr",
            "prior_pass_quality_signal_trend",
            "prior_pass_room_quality_advantage",
        ),
        "offense_snaps",
    ),
)


def _prepare_player_responses(rows: pd.DataFrame) -> pd.DataFrame:
    out = add_player_pathway_features(rows)
    named = _numeric(out, "is_replacement_player", 0).ne(1)
    team_games = _numeric(out, "team_games").clip(lower=1)
    availability = _numeric(out, "observed_availability")
    snap_share = _numeric(out, "snap_share")
    snap_observed = _numeric(out, "snap_counts_observed", 0).gt(0)
    offense_snaps = _numeric(out, "offense_snaps")
    targets = _numeric(out, "targets", 0).fillna(0)
    carries = _numeric(out, "rush_att", 0).fillna(0)
    passes = _numeric(out, "pass_att", 0).fillna(0)

    out["_conditional_snap_share"] = (snap_share / availability).clip(0, 1)
    out["_material_role"] = snap_share.ge(0.25).astype(int)
    out["_target_role_1pg"] = (targets / team_games).ge(1.0).astype(int)
    out["_carry_eligible"] = carries.gt(0).astype(int)
    out["_targets_per_snap"] = (targets / offense_snaps).clip(0, 1)
    out["_carries_per_snap"] = (carries / offense_snaps).clip(0, 1)
    out["_qb_attempts_per_snap"] = (passes / offense_snaps).clip(0, 1)

    out["_valid_availability"] = named & availability.notna() & team_games.gt(0)
    out["_valid_snap"] = (
        named & snap_observed & snap_share.notna() & availability.gt(0) & offense_snaps.gt(0)
    )
    prior_snap = _numeric(out, "prior_snap_share")
    returning = _numeric(out, "cold_start", 1).eq(0)
    out["_valid_retention"] = out["_valid_snap"] & prior_snap.ge(0.25)
    out["_valid_role_gain"] = (
        out["_valid_snap"] & returning & prior_snap.lt(0.25) & prior_snap.notna()
    )
    receiver = out["position"].isin(("RB", "WR", "TE"))
    out["_valid_target_role"] = named & receiver & team_games.gt(0)
    out["_valid_carry_eligibility"] = named
    out["_valid_target_rate"] = named & receiver & snap_observed & offense_snaps.gt(0)
    out["_valid_carry_rate"] = named & snap_observed & offense_snaps.gt(0)
    out["_valid_qb_rate"] = (
        named & out["position"].eq("QB") & snap_observed & offense_snaps.gt(0)
    )
    return out


def _feature_set(task: _Task, model: str) -> tuple[str, ...]:
    if model == "baseline":
        return task.base
    if model == "production_history":
        return tuple(dict.fromkeys((*task.base, *task.production)))
    if model == "efficiency_1yr":
        return tuple(dict.fromkeys((*task.base, *task.efficiency_1yr)))
    if model == "efficiency_state":
        return tuple(dict.fromkeys((*task.base, *task.efficiency_state)))
    if model == "combined":
        return tuple(
            dict.fromkeys((*task.base, *task.production, *task.efficiency_state))
        )
    raise ValueError(f"unknown configuration: {model}")


def _matrix_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    *,
    categoricals: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    left_columns = [np.ones(len(train), dtype=float)]
    right_columns = [np.ones(len(test), dtype=float)]
    for name in features:
        left = _numeric(train, name)
        right = _numeric(test, name)
        fill = float(left.median()) if left.notna().any() else 0.0
        filled = left.fillna(fill)
        mean = float(filled.mean())
        scale = float(filled.std(ddof=0))
        scale = scale if scale > 1e-8 else 1.0
        left_columns.append((filled.to_numpy(dtype=float) - mean) / scale)
        right_columns.append((right.fillna(fill).to_numpy(dtype=float) - mean) / scale)
    for name in categoricals:
        categories = sorted(train[name].dropna().astype(str).unique())
        for category in categories[:-1]:
            left_columns.append(train[name].astype(str).eq(category).to_numpy(dtype=float))
            right_columns.append(test[name].astype(str).eq(category).to_numpy(dtype=float))
    return np.column_stack(left_columns), np.column_stack(right_columns)


def _weights(rows: pd.DataFrame, name: str | None) -> np.ndarray:
    if name is None:
        return np.ones(len(rows), dtype=float)
    values = _numeric(rows, name).fillna(0).clip(lower=0)
    positive = values[values > 0]
    center = float(positive.median()) if len(positive) else 1.0
    return np.clip(values.to_numpy(dtype=float) / max(center, 1.0), 0.25, 5.0)


def _ridge_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    response: str,
    features: tuple[str, ...],
    *,
    weight: str | None,
    alpha: float,
    categoricals: tuple[str, ...],
) -> np.ndarray:
    X, Z = _matrix_pair(train, test, features, categoricals=categoricals)
    y = _numeric(train, response).to_numpy(dtype=float)
    w = _weights(train, weight)
    root = np.sqrt(w)
    penalty = np.eye(X.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(
        (X * root[:, None]).T @ (X * root[:, None]) + penalty,
        (X * root[:, None]).T @ (y * root),
    )
    return np.clip(Z @ beta, 0.0, 1.0)


def _logistic_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    response: str,
    features: tuple[str, ...],
    *,
    alpha: float,
    categoricals: tuple[str, ...],
) -> np.ndarray:
    X, Z = _matrix_pair(train, test, features, categoricals=categoricals)
    y = _numeric(train, response).to_numpy(dtype=float)
    if np.unique(y).size < 2:
        return np.full(len(test), float(y.mean()))
    beta = np.zeros(X.shape[1], dtype=float)
    beta[0] = np.log(np.clip(y.mean(), 1e-4, 1 - 1e-4) / np.clip(1 - y.mean(), 1e-4, 1))
    penalty = np.eye(X.shape[1]) * alpha
    penalty[0, 0] = 0.0
    for _ in range(50):
        eta = np.clip(X @ beta, -20, 20)
        probability = 1.0 / (1.0 + np.exp(-eta))
        variance = np.clip(probability * (1.0 - probability), 1e-5, None)
        gradient = X.T @ (y - probability) - penalty @ beta
        hessian = (X * variance[:, None]).T @ X + penalty
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -20, 20))), 1e-6, 1 - 1e-6)


def player_pathway_fold_metrics(
    data: SeasonAverageData,
    *,
    ridge_alpha: float = 100.0,
    logistic_alpha: float = 20.0,
    min_training_seasons: int = 2,
) -> pd.DataFrame:
    """Cross-fit indirect player-layer efficiency and production states."""
    rows = _prepare_player_responses(data.player_rows)
    records = []
    for holdout in sorted(rows["season"].unique()):
        earlier = rows[rows["season"] < holdout]
        if earlier["season"].nunique() < min_training_seasons:
            continue
        current = rows[rows["season"] == holdout]
        for task in PLAYER_TASKS:
            train = earlier[earlier[task.valid]].copy()
            test = current[current[task.valid]].copy()
            if len(train) < 30 or len(test) < 10:
                continue
            observed = _numeric(test, task.response).to_numpy(dtype=float)
            for model in MODEL_CONFIGURATIONS:
                features = _feature_set(task, model)
                if task.kind == "binary":
                    predicted = _logistic_predict(
                        train,
                        test,
                        task.response,
                        features,
                        alpha=logistic_alpha,
                        categoricals=("position",),
                    )
                    primary = np.square(predicted - observed)
                    secondary = -(
                        observed * np.log(predicted)
                        + (1.0 - observed) * np.log(1.0 - predicted)
                    )
                    metric = "brier"
                    secondary_metric = "log_loss"
                else:
                    predicted = _ridge_predict(
                        train,
                        test,
                        task.response,
                        features,
                        weight=task.weight,
                        alpha=ridge_alpha,
                        categoricals=("position",),
                    )
                    error = predicted - observed
                    primary = np.abs(error)
                    secondary = np.square(error)
                    metric = "mae"
                    secondary_metric = "rmse"
                records.append(
                    {
                        "family": "player",
                        "task": task.name,
                        "model": model,
                        "season": int(holdout),
                        "n": int(len(test)),
                        "metric": metric,
                        "score": float(primary.mean()),
                        "score_sum": float(primary.sum()),
                        "secondary_metric": secondary_metric,
                        "secondary_score": float(
                            np.sqrt(secondary.mean())
                            if secondary_metric == "rmse"
                            else secondary.mean()
                        ),
                        "secondary_sum": float(secondary.sum()),
                    }
                )
    return pd.DataFrame(records)


def add_team_efficiency_features(
    team_rows: pd.DataFrame, player_weeks: pd.DataFrame
) -> pd.DataFrame:
    """Attach prior-year team efficiency aggregated from complete weekly stats."""
    weeks = player_weeks.copy()
    weeks["team"] = [
        team_identity(team, int(season)).franchise_code
        for team, season in zip(weeks["team"], weeks["season"])
    ]
    totals = (
        weeks.groupby(["season", "team"], dropna=False)
        .agg(
            pass_att=("pass_att", "sum"),
            pass_yds=("pass_yds", "sum"),
            pass_epa=("pass_epa", lambda values: values.sum(min_count=1)),
            pass_first_downs=("pass_first_downs", lambda values: values.sum(min_count=1)),
            rush_att=("rush_att", "sum"),
            rush_yds=("rush_yds", "sum"),
            rush_epa=("rush_epa", lambda values: values.sum(min_count=1)),
            rush_first_downs=("rush_first_downs", lambda values: values.sum(min_count=1)),
        )
        .reset_index()
    )
    for numerator, denominator, output in (
        ("pass_yds", "pass_att", "team_pass_yards_per_attempt"),
        ("pass_epa", "pass_att", "team_pass_epa_per_attempt"),
        ("pass_first_downs", "pass_att", "team_pass_first_down_rate"),
        ("rush_yds", "rush_att", "team_rush_yards_per_carry"),
        ("rush_epa", "rush_att", "team_rush_epa_per_carry"),
        ("rush_first_downs", "rush_att", "team_rush_first_down_rate"),
    ):
        totals[output] = _numeric(totals, numerator).div(
            _numeric(totals, denominator).where(lambda values: values > 0)
        )
    keep = [
        "season",
        "team",
        *[name for name in totals if name.startswith("team_")],
    ]
    prior = totals[keep].copy()
    prior["season"] += 1
    prior = prior.rename(
        columns={name: f"prior_{name}" for name in keep if name.startswith("team_")}
    )
    out = team_rows.merge(prior, on=["season", "team"], how="left")
    history = (
        "prior_opportunity_plays_per_game",
        "prior_pass_rate",
        "prior_sack_rate",
        "prior_target_rate",
        *tuple(name for name in out if name.startswith("prior_team_")),
    )
    return _add_history_columns(out, history, groups=("team",))


TEAM_TASKS = {
    "opportunity_plays": {
        "response": "opportunity_plays_per_game",
        "base": ("prior_opportunity_plays_per_game",),
        "production": (
            "prior_opportunity_plays_per_game_3yr",
            "prior_opportunity_plays_per_game_trend",
        ),
        "efficiency": (
            "prior_team_pass_epa_per_attempt",
            "prior_team_pass_first_down_rate",
            "prior_team_rush_epa_per_carry",
            "prior_team_rush_first_down_rate",
        ),
    },
    "pass_rate": {
        "response": "pass_rate",
        "base": ("prior_pass_rate",),
        "production": ("prior_pass_rate_3yr", "prior_pass_rate_trend"),
        "efficiency": (
            "prior_team_pass_epa_per_attempt",
            "prior_team_pass_first_down_rate",
            "prior_team_rush_epa_per_carry",
            "prior_team_rush_first_down_rate",
        ),
    },
    "sack_rate": {
        "response": "sack_rate",
        "base": ("prior_sack_rate",),
        "production": ("prior_sack_rate_3yr", "prior_sack_rate_trend"),
        "efficiency": (
            "prior_team_pass_epa_per_attempt",
            "prior_team_pass_yards_per_attempt",
            "prior_team_pass_first_down_rate",
        ),
    },
    "target_rate": {
        "response": "target_rate",
        "base": ("prior_target_rate",),
        "production": ("prior_target_rate_3yr", "prior_target_rate_trend"),
        "efficiency": (
            "prior_team_pass_epa_per_attempt",
            "prior_team_pass_yards_per_attempt",
            "prior_team_pass_first_down_rate",
        ),
    },
}


def team_pathway_fold_metrics(
    data: SeasonAverageData,
    player_weeks: pd.DataFrame,
    *,
    ridge_alpha: float = 30.0,
    min_training_seasons: int = 2,
) -> pd.DataFrame:
    rows = add_team_efficiency_features(data.team_rows, player_weeks)
    records = []
    for holdout in sorted(rows["season"].unique()):
        train = rows[rows["season"] < holdout].copy()
        test = rows[rows["season"] == holdout].copy()
        if train["season"].nunique() < min_training_seasons:
            continue
        for task, spec in TEAM_TASKS.items():
            observed = _numeric(test, spec["response"])
            valid = observed.notna()
            current = test[valid].copy()
            y = observed[valid].to_numpy(dtype=float)
            for model in MODEL_CONFIGURATIONS:
                if model == "baseline":
                    features = spec["base"]
                elif model == "production_history":
                    features = (*spec["base"], *spec["production"])
                elif model == "efficiency_1yr":
                    features = (*spec["base"], *spec["efficiency"])
                elif model == "efficiency_state":
                    features = (
                        *spec["base"],
                        *spec["efficiency"],
                        *(f"{name}_3yr" for name in spec["efficiency"]),
                        *(f"{name}_trend" for name in spec["efficiency"]),
                    )
                else:
                    features = (
                        *spec["base"],
                        *spec["production"],
                        *spec["efficiency"],
                        *(f"{name}_3yr" for name in spec["efficiency"]),
                        *(f"{name}_trend" for name in spec["efficiency"]),
                    )
                predicted = _ridge_predict(
                    train,
                    current,
                    spec["response"],
                    tuple(dict.fromkeys(features)),
                    weight="games",
                    alpha=ridge_alpha,
                    categoricals=("team",),
                )
                # Rate responses are bounded; plays per game is not.
                if task == "opportunity_plays":
                    X, Z = _matrix_pair(
                        train,
                        current,
                        tuple(dict.fromkeys(features)),
                        categoricals=("team",),
                    )
                    weights = _weights(train, "games")
                    root = np.sqrt(weights)
                    penalty = np.eye(X.shape[1]) * ridge_alpha
                    penalty[0, 0] = 0.0
                    beta = np.linalg.solve(
                        (X * root[:, None]).T @ (X * root[:, None]) + penalty,
                        (X * root[:, None]).T
                        @ (_numeric(train, spec["response"]).to_numpy() * root),
                    )
                    predicted = np.clip(Z @ beta, 1.0, None)
                error = predicted - y
                records.append(
                    {
                        "family": "team",
                        "task": task,
                        "model": model,
                        "season": int(holdout),
                        "n": int(len(current)),
                        "metric": "mae",
                        "score": float(np.abs(error).mean()),
                        "score_sum": float(np.abs(error).sum()),
                        "secondary_metric": "rmse",
                        "secondary_score": float(np.sqrt(np.square(error).mean())),
                        "secondary_sum": float(np.square(error).sum()),
                    }
                )
    return pd.DataFrame(records)


def _weighted_room_rate(
    rows: pd.DataFrame, rate: str, exposure: str
) -> pd.Series:
    values = _numeric(rows, rate)
    weight = _numeric(rows, exposure).where(lambda x: x > 0)
    valid = values.notna() & weight.notna()
    numerator = (values * weight).where(valid, 0.0).groupby(
        [rows["season"], rows["team"], rows["position"]], dropna=False
    ).transform("sum")
    denominator = weight.where(valid, 0.0).groupby(
        [rows["season"], rows["team"], rows["position"]], dropna=False
    ).transform("sum")
    return numerator.div(denominator.where(denominator > 0))


def early_competition_rows(player_rows: pd.DataFrame) -> pd.DataFrame:
    """Team-position rows for forecasting additions before the offseason."""
    players = player_rows[
        _numeric(player_rows, "is_replacement_player", 0).ne(1)
    ].copy()
    arriving = _numeric(players, "team_change", 0).eq(1) | _numeric(
        players, "cold_start", 1
    ).eq(1)
    target_claim = _numeric(players, "prior_target_role").where(
        _numeric(players, "prior_target_role").gt(0)
    ).combine_first(_numeric(players, "draft_target_prior").where(
        _numeric(players, "draft_target_prior").gt(0)
    )).fillna(0.0)
    carry_claim = _numeric(players, "prior_carry_role").where(
        _numeric(players, "prior_carry_role").gt(0)
    ).combine_first(_numeric(players, "draft_carry_prior").where(
        _numeric(players, "draft_carry_prior").gt(0)
    )).fillna(0.0)
    players["_incoming_target_claim"] = target_claim.where(arriving, 0.0)
    players["_incoming_carry_claim"] = carry_claim.where(arriving, 0.0)
    players["_room_rec_epa"] = _weighted_room_rate(
        players, "shrunk_rec_epa_per_target", "targets"
    )
    players["_room_rec_ypt"] = _weighted_room_rate(
        players, "shrunk_rec_yards_per_target", "targets"
    )
    players["_room_rush_epa"] = _weighted_room_rate(
        players, "shrunk_rush_epa_per_carry", "rush_att"
    )
    players["_room_rush_ypc"] = _weighted_room_rate(
        players, "shrunk_rush_yards_per_carry", "rush_att"
    )
    group = ["season", "team", "position"]
    room = players.groupby(group, dropna=False).agg(
        incoming_target_claim=("_incoming_target_claim", "sum"),
        incoming_carry_claim=("_incoming_carry_claim", "sum"),
        room_target_share=("target_share", "sum"),
        room_target_leader=("target_share", "max"),
        room_carry_share=("carry_share", "sum"),
        room_carry_leader=("carry_share", "max"),
        room_player_count=("player_key", "nunique"),
        room_rec_epa=("_room_rec_epa", "first"),
        room_rec_ypt=("_room_rec_ypt", "first"),
        room_rush_epa=("_room_rush_epa", "first"),
        room_rush_ypc=("_room_rush_ypc", "first"),
    ).reset_index()
    prior_columns = [name for name in room if name not in group]
    prior = room.copy()
    prior["season"] += 1
    prior = prior.rename(columns={name: f"prior_{name}" for name in prior_columns})
    out = room[[*group, "incoming_target_claim", "incoming_carry_claim"]].merge(
        prior, on=group, how="left"
    )
    out["major_target_addition"] = out["incoming_target_claim"].ge(0.15).astype(int)
    out["major_carry_addition"] = out["incoming_carry_claim"].ge(0.15).astype(int)
    history = tuple(name for name in out if name.startswith("prior_room_"))
    return _add_history_columns(out, history, groups=("team", "position"))


COMPETITION_TASKS = {
    "incoming_target_claim": {
        "kind": "continuous",
        "positions": ("RB", "WR", "TE"),
        "base": (
            "prior_incoming_target_claim",
            "prior_room_target_share",
            "prior_room_target_leader",
            "prior_room_player_count",
        ),
        "production": (
            "prior_room_target_share_3yr",
            "prior_room_target_share_trend",
            "prior_room_target_leader_3yr",
        ),
        "efficiency": ("prior_room_rec_epa", "prior_room_rec_ypt"),
    },
    "major_target_addition": {
        "kind": "binary",
        "positions": ("RB", "WR", "TE"),
        "base": (
            "prior_incoming_target_claim",
            "prior_room_target_share",
            "prior_room_target_leader",
            "prior_room_player_count",
        ),
        "production": (
            "prior_room_target_share_3yr",
            "prior_room_target_share_trend",
            "prior_room_target_leader_3yr",
        ),
        "efficiency": ("prior_room_rec_epa", "prior_room_rec_ypt"),
    },
    "incoming_carry_claim": {
        "kind": "continuous",
        "positions": ("QB", "RB", "WR", "TE"),
        "base": (
            "prior_incoming_carry_claim",
            "prior_room_carry_share",
            "prior_room_carry_leader",
            "prior_room_player_count",
        ),
        "production": (
            "prior_room_carry_share_3yr",
            "prior_room_carry_share_trend",
            "prior_room_carry_leader_3yr",
        ),
        "efficiency": ("prior_room_rush_epa", "prior_room_rush_ypc"),
    },
    "major_carry_addition": {
        "kind": "binary",
        "positions": ("QB", "RB", "WR", "TE"),
        "base": (
            "prior_incoming_carry_claim",
            "prior_room_carry_share",
            "prior_room_carry_leader",
            "prior_room_player_count",
        ),
        "production": (
            "prior_room_carry_share_3yr",
            "prior_room_carry_share_trend",
            "prior_room_carry_leader_3yr",
        ),
        "efficiency": ("prior_room_rush_epa", "prior_room_rush_ypc"),
    },
}


def competition_pathway_fold_metrics(
    data: SeasonAverageData,
    *,
    ridge_alpha: float = 30.0,
    logistic_alpha: float = 10.0,
    min_training_seasons: int = 2,
) -> pd.DataFrame:
    rows = early_competition_rows(data.player_rows)
    records = []
    for holdout in sorted(rows["season"].unique()):
        earlier = rows[rows["season"] < holdout]
        current = rows[rows["season"] == holdout]
        if earlier["season"].nunique() < min_training_seasons:
            continue
        for task, spec in COMPETITION_TASKS.items():
            train = earlier[earlier["position"].isin(spec["positions"])].copy()
            test = current[current["position"].isin(spec["positions"])].copy()
            observed = _numeric(test, task).to_numpy(dtype=float)
            for model in MODEL_CONFIGURATIONS:
                if model == "baseline":
                    features = spec["base"]
                elif model == "production_history":
                    features = (*spec["base"], *spec["production"])
                elif model == "efficiency_1yr":
                    features = (*spec["base"], *spec["efficiency"])
                elif model == "efficiency_state":
                    features = (
                        *spec["base"],
                        *spec["efficiency"],
                        *(f"{name}_3yr" for name in spec["efficiency"]),
                        *(f"{name}_trend" for name in spec["efficiency"]),
                    )
                else:
                    features = (
                        *spec["base"],
                        *spec["production"],
                        *spec["efficiency"],
                        *(f"{name}_3yr" for name in spec["efficiency"]),
                        *(f"{name}_trend" for name in spec["efficiency"]),
                    )
                features = tuple(dict.fromkeys(features))
                if spec["kind"] == "binary":
                    predicted = _logistic_predict(
                        train,
                        test,
                        task,
                        features,
                        alpha=logistic_alpha,
                        categoricals=("position", "team"),
                    )
                    primary = np.square(predicted - observed)
                    secondary = -(
                        observed * np.log(predicted)
                        + (1 - observed) * np.log(1 - predicted)
                    )
                    metric, secondary_metric = "brier", "log_loss"
                else:
                    predicted = _ridge_predict(
                        train,
                        test,
                        task,
                        features,
                        weight=None,
                        alpha=ridge_alpha,
                        categoricals=("position", "team"),
                    )
                    error = predicted - observed
                    primary = np.abs(error)
                    secondary = np.square(error)
                    metric, secondary_metric = "mae", "rmse"
                records.append(
                    {
                        "family": "competition",
                        "task": task,
                        "model": model,
                        "season": int(holdout),
                        "n": int(len(test)),
                        "metric": metric,
                        "score": float(primary.mean()),
                        "score_sum": float(primary.sum()),
                        "secondary_metric": secondary_metric,
                        "secondary_score": float(
                            np.sqrt(secondary.mean())
                            if secondary_metric == "rmse"
                            else secondary.mean()
                        ),
                        "secondary_sum": float(secondary.sum()),
                    }
                )
    return pd.DataFrame(records)


def pathway_summary(folds: pd.DataFrame, *, recent_start: int = 2019) -> pd.DataFrame:
    """Pool fold scores and count wins against the matching baseline."""
    if folds.empty:
        return pd.DataFrame()
    records = []
    for (family, task), task_folds in folds.groupby(["family", "task"]):
        baseline = task_folds[task_folds["model"].eq("baseline")].set_index("season")
        for model, group in task_folds.groupby("model"):
            group = group.set_index("season")
            n = int(group["n"].sum())
            common = group.index.intersection(baseline.index)
            recent = common[common >= recent_start]
            metric = group["metric"].iloc[0]
            secondary_metric = group["secondary_metric"].iloc[0]
            records.append(
                {
                    "family": family,
                    "task": task,
                    "model": model,
                    "n": n,
                    "metric": metric,
                    "score": float(group["score_sum"].sum() / n),
                    "secondary_metric": secondary_metric,
                    "secondary_score": float(
                        np.sqrt(group["secondary_sum"].sum() / n)
                        if secondary_metric == "rmse"
                        else group["secondary_sum"].sum() / n
                    ),
                    "fold_wins": int(
                        (group.loc[common, "score"] < baseline.loc[common, "score"]).sum()
                    ),
                    "folds": int(len(common)),
                    "recent_wins": int(
                        (group.loc[recent, "score"] < baseline.loc[recent, "score"]).sum()
                    ),
                    "recent_folds": int(len(recent)),
                }
            )
    return pd.DataFrame(records).sort_values(
        ["family", "task", "score"]
    ).reset_index(drop=True)
