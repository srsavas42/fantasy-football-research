"""Deterministic efficiency ablation for the production QB volume layers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.season_availability import QB_WORKLOAD_FEATURES
from ffmodel.models.season_opportunity import QB_PROPENSITY_FEATURES


QB_EFFICIENCY_FEATURES = ("prior_pass_quality_signal", "prior_pass_td_rate")


def qb_layer_efficiency_fold_metrics(
    data: SeasonAverageData,
    *,
    workload_alpha: float = 300.0,
    propensity_alpha: float = 100.0,
    min_training_seasons: int = 2,
) -> pd.DataFrame:
    """Cross-fit workload and attempts-per-snap layers with efficiency toggles."""
    rows = data.player_rows
    records = []
    configurations = {
        "production_proxy": ((), ()),
        "workload_efficiency": (QB_EFFICIENCY_FEATURES, ()),
        "propensity_efficiency": ((), QB_EFFICIENCY_FEATURES),
        "both_efficiency": (QB_EFFICIENCY_FEATURES, QB_EFFICIENCY_FEATURES),
    }
    for holdout in sorted(rows["season"].unique()):
        train = rows[rows["season"] < holdout].copy()
        if train["season"].nunique() < min_training_seasons:
            continue
        test = rows[rows["season"] == holdout].copy()
        team = data.team_rows[data.team_rows["season"] == holdout].set_index(
            ["season", "team"]
        )["prior_pass_attempts_per_game"]
        keys = pd.MultiIndex.from_frame(test[["season", "team"]])
        team_rate = team.reindex(keys).to_numpy(dtype=float)
        named_qb = test["position"].eq("QB") & pd.to_numeric(
            test.get("is_replacement_player", pd.Series(0, index=test.index)),
            errors="coerce",
        ).fillna(0).ne(1)
        observed = pd.to_numeric(test["pass_att"], errors="coerce") / pd.to_numeric(
            test["team_games"], errors="coerce"
        )
        for label, (workload_extra, propensity_extra) in configurations.items():
            workload = _workload_share(
                train,
                test,
                tuple(QB_WORKLOAD_FEATURES) + workload_extra,
                alpha=workload_alpha,
            )
            propensity = _pass_propensity(
                train,
                test,
                tuple(QB_PROPENSITY_FEATURES) + propensity_extra,
                alpha=propensity_alpha,
            )
            score = (workload * propensity).fillna(0.0)
            share = pd.Series(0.0, index=test.index)
            for _, group in test.groupby(["season", "team"], dropna=False):
                values = score.loc[group.index]
                if values.sum() > 0:
                    share.loc[group.index] = values / values.sum()
                else:
                    share.loc[group.index] = workload.loc[group.index]
            predicted = share.to_numpy(dtype=float) * team_rate
            error = predicted[named_qb.to_numpy()] - observed[named_qb].to_numpy(
                dtype=float
            )
            records.append(
                {
                    "season": int(holdout),
                    "model": label,
                    "n": int(named_qb.sum()),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "absolute_error": float(np.abs(error).sum()),
                    "squared_error": float(np.square(error).sum()),
                }
            )
    return pd.DataFrame(records)


def qb_layer_efficiency_metrics(folds: pd.DataFrame) -> pd.DataFrame:
    """Pool QB layer folds and compare their stability with production proxy."""
    if folds.empty:
        return pd.DataFrame()
    baseline = folds[folds["model"].eq("production_proxy")].set_index("season")
    records = []
    for model, group in folds.groupby("model"):
        group = group.set_index("season")
        n = int(group["n"].sum())
        common = group.index.intersection(baseline.index)
        recent = common[common >= 2019]
        records.append(
            {
                "model": model,
                "n": n,
                "mae": float(group["absolute_error"].sum() / n),
                "rmse": float(np.sqrt(group["squared_error"].sum() / n)),
                "fold_wins": int((group.loc[common, "mae"] < baseline.loc[common, "mae"]).sum()),
                "folds": int(len(common)),
                "recent_wins": int((group.loc[recent, "mae"] < baseline.loc[recent, "mae"]).sum()),
                "recent_folds": int(len(recent)),
            }
        )
    return pd.DataFrame(records)


def _role(rows: pd.DataFrame) -> pd.Series:
    snaps = pd.to_numeric(rows["prior_qb_snap_share"], errors="coerce")
    passing = pd.to_numeric(rows["prior_pass_role"], errors="coerce")
    draft = pd.to_numeric(rows["draft_pass_prior"], errors="coerce")
    return (
        snaps.where(snaps > 0)
        .combine_first(passing.where(passing > 0))
        .combine_first(draft.where(draft > 0))
        .fillna(0.02)
        .clip(1e-5, 1.0)
    )


def _matrix_pair(
    train: pd.DataFrame, test: pd.DataFrame, features: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    train_columns = [np.ones(len(train), dtype=float)]
    test_columns = [np.ones(len(test), dtype=float)]
    for name in features:
        left = pd.to_numeric(
            train.get(name, pd.Series(np.nan, index=train.index)), errors="coerce"
        )
        right = pd.to_numeric(
            test.get(name, pd.Series(np.nan, index=test.index)), errors="coerce"
        )
        fill = float(left.median()) if left.notna().any() else 0.0
        filled = left.fillna(fill)
        mean = float(filled.mean())
        scale = float(filled.std(ddof=0))
        scale = scale if scale > 1e-8 else 1.0
        train_columns.append((filled.to_numpy(dtype=float) - mean) / scale)
        test_columns.append((right.fillna(fill).to_numpy(dtype=float) - mean) / scale)
    return np.column_stack(train_columns), np.column_stack(test_columns)


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float, weight=None) -> np.ndarray:
    if weight is None:
        weight = np.ones(len(y), dtype=float)
    root = np.sqrt(np.asarray(weight, dtype=float))
    weighted = X * root[:, None]
    penalty = np.eye(X.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        weighted.T @ weighted + penalty,
        weighted.T @ (y * root),
    )


def _workload_share(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    *,
    alpha: float,
) -> pd.Series:
    left = train[train["position"].eq("QB")].copy()
    right = test[test["position"].eq("QB")].copy()
    X, Z = _matrix_pair(left, right, features)
    observed = pd.to_numeric(
        left["observed_qb_workload_share"], errors="coerce"
    ).fillna(0).clip(1e-5, 1.0)
    availability = pd.to_numeric(
        left["observed_availability"], errors="coerce"
    ).fillna(0.75).clip(0.03, 1.0)
    adjusted = (observed / availability).clip(lower=1e-5)
    adjusted /= adjusted.groupby([left["season"], left["team"]]).transform("sum")
    response = np.log(adjusted.to_numpy(dtype=float)) - np.log(
        _role(left).to_numpy(dtype=float)
    )
    coefficient = _ridge(X, response, alpha)
    score = (
        np.log(_role(right).to_numpy(dtype=float))
        + np.log(
            pd.to_numeric(right["prior_availability"], errors="coerce")
            .fillna(0.75)
            .clip(0.03, 1.0)
            .to_numpy(dtype=float)
        )
        + Z @ coefficient
    )
    output = pd.Series(0.0, index=test.index)
    scored = right.assign(_score=score)
    for _, group in scored.groupby(["season", "team"], dropna=False):
        values = np.exp(group["_score"] - group["_score"].max())
        output.loc[group.index] = values / values.sum()
    return output


def _pass_propensity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    *,
    alpha: float,
) -> pd.Series:
    left = train[train["position"].eq("QB")].copy()
    right = test[test["position"].eq("QB")].copy()
    snaps = pd.to_numeric(left["offense_snaps"], errors="coerce").fillna(0)
    attempts = pd.to_numeric(left["pass_att"], errors="coerce").fillna(0)
    observed = pd.to_numeric(
        left["snap_counts_observed"], errors="coerce"
    ).fillna(0).gt(0)
    valid = observed & snaps.gt(0)
    left = left[valid]
    snaps = snaps[valid]
    attempts = attempts[valid]
    X, Z = _matrix_pair(left, right, features)
    rate = ((attempts + 0.5) / (snaps + 1.0)).clip(1e-4, 1 - 1e-4)
    response = np.log(rate / (1.0 - rate)).to_numpy(dtype=float)
    median = max(float(snaps.median()), 1.0)
    weight = np.clip(snaps.to_numpy(dtype=float) / median, 0.25, 5.0)
    coefficient = _ridge(X, response, alpha, weight)
    eta = Z @ coefficient
    propensity = 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20)))
    return pd.Series(propensity, index=right.index)
