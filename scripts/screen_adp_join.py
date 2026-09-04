"""Blend or feature: two ways to join the pipeline to the draft board.

``project_season.py`` ships a *mixture*: each posterior draw comes from either
the pipeline or the ADP rank curve, at a fixed 0.316 weight on the pipeline. The
obvious alternative is a *stack* -- fit ``points ~ a + b*pipeline + c*ADP`` and
let the data choose both weights, with an intercept free to absorb bias in
either source. A mixture cannot do that: it cannot reweight, cannot shift, and
its spread is the union of two distributions rather than a fitted one.

Fitted leave-one-season-out across the three scored holdouts, where each
season's pipeline mean already comes from a pipeline that never saw it.

    2023: stack = +5.8  + 0.441*pipeline + 0.538*ADP
    2024: stack = +13.8 + 0.311*pipeline + 0.605*ADP
    2025: stack = +11.3 + 0.408*pipeline + 0.561*ADP

    arm                         MAE    vs ADP
    ADP alone                 56.74     +0.0%
    pipeline alone            58.11     +2.4%
    blend (0.316 mixture)     54.57     -3.8%
    stack (ADP as feature)    54.91     -3.2%

**The mixture wins, narrowly.** So the answer to "would it be better as a
feature" is no on point accuracy -- but the margin is a third of a point of MAE,
and the stack is doing something the mixture cannot, which shows up directionally
rather than in the average.

    arm               corr    slope
    pipeline        +0.256   +0.392
    blend           +0.256   +1.242
    stack           +0.236   +0.865

The slope is how much real deviation a unit of the arm's disagreement buys. The
raw pipeline overstates itself 2.5-fold, which is the known result and the reason
a weight exists at all. The *blend* overcorrects: at slope 1.242 its disagreement
is undersized by about a quarter, meaning it should be pushed further from the
board than it is -- which is the same finding as the weight sweep, where the
optimum sat at 0.40 against a shipped 0.316. The stack, with weights the data
chose, lands at 0.865, the closest of the three to correctly sized.

So the two constructions fail in opposite directions and neither is clearly
better. The stack's fitted weight on the pipeline, 0.31 to 0.44 across folds,
independently reproduces both the 0.316 shipped weight and the 0.392
disagreement slope, which is a useful consistency check on all three.

Worth noting what the stack found that the mixture structurally cannot: every
fold wants a positive intercept, +5.8 to +13.8 points, with the two slopes
summing to slightly under one. Both forecasts are biased low on this population
and a mixture has no way to say so.

    python scripts/screen_adp_join.py
"""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path

import numpy as np
from scipy import stats

BLEND_WEIGHT = 0.316


def load(paths):
    seasons = {}
    for path in paths:
        blob = json.loads(Path(path).read_text("utf-8"))
        for season, fold in blob["folds"].items():
            seasons[int(season)] = {
                k: np.array(v, dtype=float) for k, v in fold["rows"].items()
            }
    return seasons


def main() -> int:
    seasons = load([f"reports/mva_{y}.json" for y in (2023, 2024, 2025)])
    years = sorted(seasons)
    parts = []
    print("stack fitted leave-one-season-out:")
    for test in years:
        train = [y for y in years if y != test]
        design = np.column_stack([
            np.ones(sum(len(seasons[y]["observed"]) for y in train)),
            np.concatenate([seasons[y]["model_mean"] for y in train]),
            np.concatenate([seasons[y]["adp_mean"] for y in train]),
        ])
        target = np.concatenate([seasons[y]["observed"] for y in train])
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        block = seasons[test]
        parts.append({
            "observed": block["observed"],
            "adp": block["adp_mean"],
            "model": block["model_mean"],
            "stack": beta[0] + beta[1] * block["model_mean"] + beta[2] * block["adp_mean"],
            "blend": block["adp_mean"]
            + BLEND_WEIGHT * (block["model_mean"] - block["adp_mean"]),
        })
        print(f"  {test}: {beta[0]:+.1f} + {beta[1]:.3f}*pipeline + {beta[2]:.3f}*ADP")

    pull = lambda key: np.concatenate([p[key] for p in parts])
    observed, adp = pull("observed"), pull("adp")
    print(f"\n  {'arm':24} {'MAE':>8} {'vs ADP':>8}")
    base = np.abs(observed - adp).mean()
    for name, key in (("ADP alone", "adp"), ("pipeline alone", "model"),
                      ("blend (0.316 mixture)", "blend"), ("stack (ADP as feature)", "stack")):
        mae = np.abs(observed - pull(key)).mean()
        print(f"  {name:24} {mae:8.2f} {(mae - base) / base:+8.1%}")

    print(f"\n  {'arm':24} {'corr':>8} {'slope':>8} {'p':>10}")
    for name, key in (("pipeline", "model"), ("blend", "blend"), ("stack", "stack")):
        gap, error = pull(key) - adp, observed - adp
        r, p = stats.pearsonr(gap, error)
        print(f"  {name:24} {r:+8.3f} {np.polyfit(gap, error, 1)[0]:+8.3f} {p:10.2g}")
    print("\n  slope 1.0 is a correctly sized disagreement; below it is overstated,")
    print("  above it is too timid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
