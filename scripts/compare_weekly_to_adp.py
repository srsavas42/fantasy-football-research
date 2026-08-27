"""Does the weekly model beat a naive draft board on every metric?

Not "on average", and not "on the metric that flatters it". The bar is every
accuracy metric, on the population where ADP is a real forecast, in every fold
and at every point in the season -- and where it is not met, this says so rather
than quietly reporting the pooled number.

Read from the walk-forward JSON, so it cannot disagree with the run it describes.

Metrics are split by direction, which is the part a naive comparison gets wrong.
MAE, RMSE and CRPS are losses and want to fall. Coverage is neither -- it wants
to sit *at* nominal, so an arm that covers 0.98 against a nominal 0.80 is worse
than one covering 0.81, not better, and the comparison is on distance from
target. Ordering metrics want to rise.

    python scripts/compare_weekly_to_adp.py --model hurdle+context/position
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASELINE = "adp-curve"

# The arm that actually ships for each response, which is not the last rung of
# either ladder: the rest-of-season simulator is built last and loses to the
# direct regression, so comparing it against ADP would be comparing something
# nobody runs.
SHIPPED = {
    "next_week": "hurdle+context+adp+news/position",
    "rest_of_season": "direct-total+phase+adp",
}

# name -> (direction, target). "lower" and "higher" are self-explanatory;
# "target" means the metric is scored on absolute distance from a nominal value.
METRICS = {
    "mae": ("lower", None),
    "rmse": ("lower", None),
    "crps": ("lower", None),
    "coverage_80": ("target", 0.80),
    "coverage_95": ("target", 0.95),
    "within_group_spearman": ("higher", None),
    "within_group_top_k": ("higher", None),
}

# Below this a difference is not worth claiming either way. The season layer's
# ADP work used the same floor.
MATERIAL = 0.0025


def _better(metric: str, model: float, baseline: float) -> tuple[bool, float]:
    """Is the model better, and by what relative margin?"""
    direction, target = METRICS[metric]
    if direction == "lower":
        return model < baseline, (baseline - model) / abs(baseline) if baseline else 0.0
    if direction == "higher":
        return model > baseline, (model - baseline) / abs(baseline) if baseline else 0.0
    model_gap, baseline_gap = abs(model - target), abs(baseline - target)
    return model_gap < baseline_gap, (baseline_gap - model_gap) / max(baseline_gap, 1e-9)


def compare(results: dict, model: str, population: str) -> pd.DataFrame:
    pooled = results.get("pooled", {})
    if model not in pooled or BASELINE not in pooled:
        return pd.DataFrame()
    left = pooled[model].get(population)
    right = pooled[BASELINE].get(population)
    if not left or not right:
        return pd.DataFrame()
    rows = []
    for metric in METRICS:
        if metric not in left or metric not in right:
            continue
        won, margin = _better(metric, left[metric], right[metric])
        rows.append(
            {
                "metric": metric,
                "model": round(left[metric], 4),
                "adp": round(right[metric], 4),
                "margin": f"{margin:+.2%}",
                "verdict": "win" if won and abs(margin) >= MATERIAL
                else ("LOSS" if not won and abs(margin) >= MATERIAL else "tie"),
            }
        )
    return pd.DataFrame(rows)


def per_fold(results: dict, model: str, population: str, metric: str) -> pd.DataFrame:
    rows = []
    for fold in results.get("folds", []):
        block = fold["estimators"]
        if model not in block or BASELINE not in block:
            continue
        if population not in block[model] or population not in block[BASELINE]:
            continue
        left = block[model][population][metric]
        right = block[BASELINE][population][metric]
        won, margin = _better(metric, left, right)
        rows.append(
            {
                "holdout": fold["holdout"],
                "model": round(left, 4),
                "adp": round(right, 4),
                "margin": f"{margin:+.2%}",
                "verdict": "win" if won else "LOSS",
            }
        )
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("scripts/validation_runs/weekly_ladder.json")
    )
    parser.add_argument("--model", default=None, help="defaults to the last rung")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text("utf-8"))
    summary: dict[str, object] = {}
    losses: list[str] = []

    for task in ("next_week", "rest_of_season"):
        if task not in payload:
            continue
        results = payload[task]
        names = list(results.get("pooled", {}).keys())
        default = SHIPPED.get(task, names[-1])
        model = args.model if args.model in names else (
            default if default in names else names[-1]
        )
        print(f"\n{'=' * 70}\n{task}: {model}  vs  {BASELINE}\n{'=' * 70}")

        task_summary = {}
        for population in (
            "drafted",
            "drafted_early",
            "drafted_mid",
            "drafted_late",
            "relevant",
        ):
            table = compare(results, model, population)
            if table.empty:
                continue
            n = results["pooled"][model][population]["n"]
            print(f"\n-- {population} (n={n}) --")
            print(table.to_string(index=False))
            task_summary[population] = table.to_dict("records")
            for row in table[table["verdict"] == "LOSS"].itertuples():
                losses.append(f"{task}/{population}/{row.metric} ({row.margin})")

        print(f"\n-- per fold, drafted, CRPS --")
        folds = per_fold(results, model, "drafted", "crps")
        if not folds.empty:
            print(folds.to_string(index=False))
            task_summary["folds_drafted_crps"] = folds.to_dict("records")
        summary[task] = {"model": model, **task_summary}

    print(f"\n{'=' * 70}")
    if losses:
        print(f"NOT uniformly better than ADP. {len(losses)} metric(s) lost:")
        for entry in losses:
            print(f"  - {entry}")
    else:
        print("Every compared metric beats the ADP baseline by the material margin.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "losses": losses}, indent=2, default=str),
            "utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
