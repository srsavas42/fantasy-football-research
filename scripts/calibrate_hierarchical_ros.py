"""Can the hierarchical rest-of-season model's intervals be repaired after the fact?

The structured estimator is a viable *mean* — with the full feature surface it
reaches MAE 30.28 against the shipped direct regression's 30.32, a dead heat —
and it is rejected on calibration alone: 80% coverage of **0.575**, against 0.790
for the control. Since the shipped projection carries p10/p50/p90 and the league
agent reads all three, an interval covering 57.5% of outcomes inside a band that
claims 80% is not a cosmetic defect. It would make every waiver decision
overconfident.

Adding features does not touch it — the full arm covers 0.575 against the
stripped arm's 0.579 — which says the failure is not an information problem.
Each layer's variance is *composed*: a Beta concentration read off realized play
counts, a persistent level SD read off the covariance between distinct weeks,
and a week-noise scale backed out so the residual pool is not counted twice.
Every one is a point estimate plugged in as though known, and the product is too
narrow.

So the question this asks is whether the *shape* is right and only the width is
wrong. If so, one scalar per fold fixes it and the structured model becomes
usable; if the shape is wrong too, no single factor will do it and the
architecture needs more than a patch.

**The factor is estimated out of sample or it proves nothing.** For holdout Y the
model is fitted on seasons before Y-1, predicts Y-1, and the width that would
have made Y-1 cover nominally is solved there; the model is then refitted on
everything before Y and that stored factor applied to Y. Y never informs its own
calibration. This is the nesting ``fit_blend_weights`` already uses.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.weekly.evaluate import score
from ffmodel.weekly.features import relevant_population
from ffmodel.weekly.restofseason import (
    TARGET,
    DirectTotal,
    HierarchicalSeason,
    add_rest_of_season_target,
)

NOMINAL = 0.80
GRID = np.round(np.arange(1.0, 3.01, 0.05), 2)


def widen(samples: np.ndarray, factor: float) -> np.ndarray:
    """Stretch each row's draws about their own median.

    The median rather than the mean because the totals are right-skewed and a
    scale about the mean would shift the centre as it widens. Clipped at zero:
    a rest-of-season total cannot be negative, and letting the stretched tail
    go below it would buy coverage with impossible outcomes.
    """
    centre = np.median(samples, axis=1, keepdims=True)
    return np.maximum(centre + factor * (samples - centre), 0.0)


def coverage(samples: np.ndarray, observed: np.ndarray, level: float = NOMINAL) -> float:
    lo = np.quantile(samples, (1.0 - level) / 2.0, axis=1)
    hi = np.quantile(samples, 1.0 - (1.0 - level) / 2.0, axis=1)
    return float(np.mean((observed >= lo) & (observed <= hi)))


def solve_factor(samples: np.ndarray, observed: np.ndarray) -> float:
    """The width whose 80% band covers 80% of the inner season."""
    best, gap = 1.0, float("inf")
    for factor in GRID:
        got = abs(coverage(widen(samples, factor), observed) - NOMINAL)
        if got < gap:
            best, gap = float(factor), got
    return best


def build(kind: str):
    if kind == "hierarchical":
        return HierarchicalSeason(
            name="aggregated-weekly", persistent=True,
            use_team=True, use_phase=True, use_adp=True, use_role=True,
        )
    return DirectTotal(name="direct-total+everything", use_team=True, use_phase=True,
                       use_adp=True, use_role=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=Path(".cache/weekly_features_2016_2025.pkl"))
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--draws", type=int, default=600)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/hierarchical_calibration.json"))
    args = parser.parse_args(argv)

    frame = add_rest_of_season_target(pd.read_pickle(args.features))
    frame = frame[np.isfinite(pd.to_numeric(frame[TARGET], errors="coerce"))]

    rows, factors = [], {}
    for holdout in args.holdouts:
        inner = holdout - 1
        # --- the factor, solved on a season the outer fit has not seen scored
        inner_train = frame[frame["season"] < inner]
        inner_test = frame[frame["season"] == inner]
        model = build("hierarchical").fit(
            inner_train, inner_train[TARGET].to_numpy(float)
        )
        keep = relevant_population(inner_test).to_numpy(bool)
        inner_test = inner_test[keep]
        inner_samples = model.predict_samples(inner_test, draws=args.draws, seed=inner)
        inner_observed = pd.to_numeric(inner_test[TARGET], errors="coerce").to_numpy(float)
        factor = solve_factor(inner_samples, inner_observed)
        factors[holdout] = factor
        print(f"  {holdout}: inner {inner} raw coverage "
              f"{coverage(inner_samples, inner_observed):.3f} -> factor {factor:.2f}", flush=True)

        # --- the holdout, with that factor applied unseen
        train = frame[frame["season"] < holdout]
        test = frame[frame["season"] == holdout]
        test = test[relevant_population(test).to_numpy(bool)]
        observed = pd.to_numeric(test[TARGET], errors="coerce").to_numpy(float)
        position = test["position"].astype(str).to_numpy()

        for kind in ("hierarchical", "direct"):
            fitted = build(kind).fit(train, train[TARGET].to_numpy(float))
            samples = fitted.predict_samples(test, draws=args.draws, seed=holdout)
            arms = {kind: samples}
            if kind == "hierarchical":
                arms["hierarchical+calibrated"] = widen(samples, factor)
            for name, block in arms.items():
                got = score(observed, block, groups=position)
                rows.append({"holdout": holdout, "arm": name, "factor":
                             factor if "calibrated" in name else 1.0, **got})
        print(f"  {holdout} done", flush=True)

    table = pd.DataFrame(rows)
    metrics = [c for c in ("mae", "crps", "coverage_80", "coverage_95",
                           "within_group_spearman") if c in table.columns]
    pooled = (table.assign(**{m: table[m] * table["n"] for m in metrics})
                   .groupby("arm", as_index=False).sum(numeric_only=True))
    for m in metrics:
        pooled[m] = pooled[m] / pooled["n"]

    order = ["direct", "hierarchical", "hierarchical+calibrated"]
    print(f"\n=== pooled over {args.holdouts} (relevant population) ===")
    print(pooled.set_index("arm").loc[[o for o in order if o in set(pooled['arm'])]]
          [["n", *metrics]].round(4).to_string())

    print("\n=== per fold ===")
    print(table.pivot_table(index="holdout", columns="arm",
                            values=["coverage_80", "crps"]).round(3).to_string())
    print(f"\nfactors solved out of sample: {factors}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"factors": factors, "per_fold": rows,
             "pooled": pooled.to_dict("records")}, indent=2, default=str), "utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
