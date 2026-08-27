"""Walk-forward scoring for the weekly layer.

One rule, enforced here rather than trusted to each estimator: a fold fits on
seasons strictly before its holdout. Nothing in a holdout season reaches a fit,
including the holdout's own earlier weeks. That is stricter than a weekly model
strictly needs -- in real use week 10 is fitted with weeks 1-9 in hand -- and it
is chosen deliberately, because a fold that refits every week would let a
hyperparameter chosen once leak across the boundary in a way that is very hard
to audit. The cost is that the model is evaluated slightly pessimistically. The
benefit is that a result cannot be an artefact of the fitting schedule.

Every estimator is scored on two populations, and both are reported:

``panel``
    Every rostered player-week. Roughly 43% of these are players who did not
    play, most of them third-stringers nobody rosters.

``relevant``
    The rows a manager is plausibly deciding about, defined in
    :func:`~ffmodel.weekly.features.relevant_population` from lagged columns
    only. This is the headline population, and the reason is the season layer's:
    scoring on a fringe-heavy mixture flatters a model that is good at
    forecasting zero and hides what it does on the players who matter.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import (
    empirical_crps,
    interval_coverage,
    ordering_metrics,
    pit_calibration,
)
from ffmodel.weekly.features import relevant_population

DEFAULT_DRAWS = 1000


def score(
    observed: np.ndarray,
    samples: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    top_k: int = 24,
) -> dict[str, float]:
    """MAE, RMSE, CRPS, coverage, bias and ordering for one block of rows."""
    observed = np.asarray(observed, float)
    samples = np.asarray(samples, float)
    mean = samples.mean(axis=1)
    error = mean - observed
    out = {
        "n": int(len(observed)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, 0.8)["coverage"]),
        "coverage_95": float(interval_coverage(observed, samples, 0.95)["coverage"]),
    }
    ordering = ordering_metrics(mean, observed, groups=groups, k=top_k)
    for key in ("spearman", "concordance", "within_group_spearman", "within_group_top_k"):
        if key in ordering:
            out[key] = float(ordering[key])
    out["pit_shape"] = pit_calibration(observed, samples)["shape"]
    out["pit_deviation"] = float(pit_calibration(observed, samples)["deviation"])
    return out


def walk_forward(
    frame: pd.DataFrame,
    estimators: Sequence,
    *,
    target: str,
    holdouts: Iterable[int],
    eligible: Callable[[pd.DataFrame], pd.Series] | None = None,
    draws: int = DEFAULT_DRAWS,
    seed: int = 20260827,
    min_train_seasons: int = 2,
) -> dict:
    """Fit each estimator on prior seasons, score it on each holdout season."""
    holdouts = sorted({int(h) for h in holdouts})
    frame = frame.copy()
    usable = np.isfinite(pd.to_numeric(frame[target], errors="coerce"))
    if eligible is not None:
        usable &= eligible(frame).to_numpy(bool)
    frame = frame[usable]

    results: dict[str, list] = {"folds": []}
    for holdout in holdouts:
        train = frame[frame["season"] < holdout]
        test = frame[frame["season"] == holdout]
        if train["season"].nunique() < min_train_seasons or test.empty:
            continue
        relevant = relevant_population(test).to_numpy(bool)
        fold: dict[str, object] = {
            "holdout": int(holdout),
            "train_seasons": sorted(int(s) for s in train["season"].unique()),
            "n_test": int(len(test)),
            "n_relevant": int(relevant.sum()),
            "estimators": {},
        }
        observed = pd.to_numeric(test[target], errors="coerce").to_numpy(float)
        position = test["position"].astype(str).to_numpy()
        train_target = pd.to_numeric(train[target], errors="coerce").to_numpy(float)

        week = pd.to_numeric(test["week"], errors="coerce").to_numpy(float)
        drafted = (
            pd.to_numeric(test["adp_drafted"], errors="coerce").eq(1).to_numpy()
            if "adp_drafted" in test.columns
            else None
        )
        for estimator in estimators:
            fitted = estimator.fit(train, train_target)
            samples = fitted.predict_samples(test, draws=draws, seed=seed + holdout)
            entry = {
                "panel": score(observed, samples, groups=position),
                "relevant": score(
                    observed[relevant], samples[relevant], groups=position[relevant]
                ),
            }
            # The rest-of-season question is not one question. Week 1 is the
            # draft and week 9 is the waiver wire, and an estimator can be good
            # at one and poor at the other; pooling hides that.
            for label, window in (("early", (1, 4)), ("mid", (5, 10)), ("late", (11, 18))):
                inside = relevant & (week >= window[0]) & (week <= window[1])
                if inside.sum() >= 50:
                    entry[f"relevant_{label}"] = score(
                        observed[inside], samples[inside], groups=position[inside]
                    )
            # The population where the draft board is a real forecast rather
            # than an extrapolation. Beating ADP on players it declined to rank
            # is close to free -- it has nothing to say about them -- so the
            # honest comparison is here, which is the split the season layer's
            # ADP work also had to make.
            if drafted is not None:
                inside = relevant & drafted
                if inside.sum() >= 50:
                    entry["drafted"] = score(
                        observed[inside], samples[inside], groups=position[inside]
                    )
                    for label, window in (
                        ("early", (1, 4)),
                        ("mid", (5, 10)),
                        ("late", (11, 18)),
                    ):
                        block = inside & (week >= window[0]) & (week <= window[1])
                        if block.sum() >= 50:
                            entry[f"drafted_{label}"] = score(
                                observed[block], samples[block], groups=position[block]
                            )
            fold["estimators"][estimator.name] = entry
        results["folds"].append(fold)

    results["pooled"] = _pool(results["folds"])
    results["target"] = target
    results["draws"] = int(draws)
    return results


def _pool(folds: list[dict]) -> dict:
    """Row-count-weighted means across folds, per estimator and population."""
    pooled: dict[str, dict[str, dict[str, float]]] = {}
    if not folds:
        return pooled
    names = list(folds[0]["estimators"].keys())
    populations = sorted(
        {p for f in folds for e in f["estimators"].values() for p in e}
    )
    for name in names:
        pooled[name] = {}
        for population in populations:
            blocks = [
                f["estimators"][name][population]
                for f in folds
                if name in f["estimators"] and population in f["estimators"][name]
            ]
            total = sum(b["n"] for b in blocks)
            if not total:
                continue
            keys = [
                k
                for k in blocks[0]
                if k != "n" and isinstance(blocks[0][k], (int, float))
            ]
            pooled[name][population] = {
                "n": int(total),
                **{
                    key: float(
                        sum(b[key] * b["n"] for b in blocks if key in b) / total
                    )
                    for key in keys
                },
            }
    return pooled


def report(results: dict, population: str = "relevant") -> pd.DataFrame:
    """Pooled table, one row per estimator, in ladder order."""
    rows = []
    for name, block in results.get("pooled", {}).items():
        entry = block.get(population)
        if entry is None:
            continue
        rows.append({"estimator": name, **entry})
    return pd.DataFrame(rows)
