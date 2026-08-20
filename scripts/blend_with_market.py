"""Can the model and the draft board together beat either alone?

Four hypotheses for the pipeline's deficit against a rank curve have been tested
and eliminated, and the one that survives — that its component projections are
simply less accurate — is not a defect anyone can go and fix. That makes the
question worth asking differently. Two forecasts that make *different* mistakes
combine into a better one even when neither dominates, and nothing about the
pipeline losing to ADP implies their errors are the same errors.

This measures the ceiling and then tries to reach it honestly.

**The ceiling** is set by how correlated the two error series are. Perfectly
correlated forecasts cannot help each other; independent ones combine to beat
both. The correlation is reported before any blending, because it says in advance
whether the exercise can work.

**Two ways to combine, and they are not the same.** Averaging paired draws
(`w * model + (1-w) * curve`) produces a distribution *narrower* than either
input, which flatters MAE and can wreck CRPS. Mixing (drawing from the model with
probability `w`, from the curve otherwise) produces one *wider* than either. Both
give the same mean, so MAE cannot distinguish them and CRPS decides. Reporting
only one would be choosing the answer.

**The weight is chosen out of sample.** For each holdout the blend weight comes
from the earlier holdouts only, so the reported score never uses a weight fitted
to the season it scores. The first holdout has no earlier one and is reported
separately, marked, and excluded from the honest average — an in-sample weight on
the season it is scoring would be the most flattering number here and the least
meaningful.

Undrafted players get the model unchanged: the curve has no rank for them and
nothing to say.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage

_spec = importlib.util.spec_from_file_location(
    "_adp_only", Path(__file__).with_name("benchmark_adp_only.py")
)
_adp_only = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adp_only)

WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def score(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    mean = samples.mean(axis=1)
    return {
        "n": int(len(observed)),
        "mae": float(np.abs(mean - observed).mean()),
        "rmse": float(np.sqrt(np.mean((mean - observed) ** 2))),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, 0.80)["coverage"]),
        "coverage_95": float(interval_coverage(observed, samples, 0.95)["coverage"]),
    }


def average(model: np.ndarray, curve: np.ndarray, w: float) -> np.ndarray:
    """Paired-draw convex combination. Narrower than either input."""
    return w * model + (1.0 - w) * curve


def mixture(model: np.ndarray, curve: np.ndarray, w: float, rng) -> np.ndarray:
    """Draw from one component or the other. Wider than either input."""
    take = rng.random(model.shape) < w
    return np.where(take, model, curve)


def load_holdout(out_dir: Path, label: str, holdout: int):
    base = out_dir / f"{label}_{holdout}"
    rows = pd.read_parquet(base.with_suffix(".rows.parquet"))
    samples = np.load(base.with_suffix(".samples.npz"))["samples"].astype(float)
    meta = json.loads(base.with_suffix(".meta.json").read_text())
    return rows, samples, meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--label", default="shipping")
    parser.add_argument(
        "--out-dir", type=Path, default=Path(".cache/holdout-predictions")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp2")
    )
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path("scripts/validation_runs/market_blend.json")

    pool = _adp_only.prepare(args.cache_dir, args.scoring)
    folds: dict[int, dict] = {}
    configs = set()
    for holdout in args.holdouts:
        rows, samples, meta = load_holdout(args.out_dir, args.label, holdout)
        configs.add(json.dumps(meta["config"], sort_keys=True))
        named = (
            pd.to_numeric(rows.get("is_replacement_player"), errors="coerce")
            .fillna(0)
            .ne(1)
            .to_numpy()
        )
        drafted = pd.to_numeric(rows["adp_drafted"], errors="coerce").eq(1).to_numpy()
        finite = np.isfinite(rows["observed"].to_numpy(float)) & np.isfinite(
            samples
        ).all(axis=1)
        mask = named & drafted & finite
        block = rows[mask].reset_index(drop=True)
        model = samples[mask]
        curve = _adp_only.project(
            pool[pool.season.lt(holdout)], block, model.shape[1], seed=4000 + holdout
        )
        folds[holdout] = {
            "observed": block["observed"].to_numpy(float),
            "model": model,
            "curve": curve,
        }

    if len(configs) != 1:
        raise SystemExit(
            "the exported holdouts were produced under different model "
            "configurations; re-export before blending"
        )
    config = json.loads(configs.pop())

    print(f"\nmodel configuration exported: {config}\n")

    # The ceiling. Two forecasts that miss the same players the same way cannot
    # help each other, whatever weight is chosen.
    print("  error correlation between the model and the curve, drafted pool:")
    correlations = {}
    for holdout, f in folds.items():
        a = f["model"].mean(axis=1) - f["observed"]
        b = f["curve"].mean(axis=1) - f["observed"]
        correlations[holdout] = float(np.corrcoef(a, b)[0, 1])
        print(f"    {holdout}  {correlations[holdout]:+.3f}   (n={len(a)})")

    def curve_for(holdout: int, w: float, how: str) -> np.ndarray:
        f = folds[holdout]
        if how == "average":
            return average(f["model"], f["curve"], w)
        return mixture(f["model"], f["curve"], w, np.random.default_rng(7))

    # Weight chosen on earlier holdouts only.
    report: dict[str, object] = {"config": config, "correlation": correlations}
    ordered = sorted(folds)
    for how in ("average", "mixture"):
        print(f"\n  {how.upper()}\n")
        print(f"    {'holdout':>8s} {'w':>5s} {'model MAE':>10s} {'curve MAE':>10s} "
              f"{'blend MAE':>10s} {'model CRPS':>11s} {'curve CRPS':>11s} "
              f"{'blend CRPS':>11s}")
        entries = []
        for i, holdout in enumerate(ordered):
            earlier = ordered[:i]
            if earlier:
                totals = []
                for w in WEIGHTS:
                    value = sum(
                        empirical_crps(
                            folds[h]["observed"], curve_for(h, w, how)
                        ).mean()
                        * len(folds[h]["observed"])
                        for h in earlier
                    )
                    totals.append(value)
                w = float(WEIGHTS[int(np.argmin(totals))])
                honest = True
            else:
                w = float("nan")
                honest = False
            f = folds[holdout]
            entry = {
                "holdout": holdout,
                "weight": w,
                "honest": honest,
                "model": score(f["observed"], f["model"]),
                "curve": score(f["observed"], f["curve"]),
            }
            if honest:
                entry["blend"] = score(f["observed"], curve_for(holdout, w, how))
                print(
                    f"    {holdout:>8d} {w:>5.2f} {entry['model']['mae']:>10.2f} "
                    f"{entry['curve']['mae']:>10.2f} {entry['blend']['mae']:>10.2f} "
                    f"{entry['model']['crps']:>11.2f} {entry['curve']['crps']:>11.2f} "
                    f"{entry['blend']['crps']:>11.2f}"
                )
            else:
                print(
                    f"    {holdout:>8d} {'--':>5s} {entry['model']['mae']:>10.2f} "
                    f"{entry['curve']['mae']:>10.2f} {'--':>10s} "
                    f"{entry['model']['crps']:>11.2f} {entry['curve']['crps']:>11.2f} "
                    f"{'--':>11s}   no earlier holdout to pick w on"
                )
            entries.append(entry)
        honest_entries = [e for e in entries if e["honest"]]
        if honest_entries:
            weight = sum(e["blend"]["n"] for e in honest_entries)
            for metric in ("mae", "crps"):
                pooled = {
                    which: sum(e[which][metric] * e[which]["n"] for e in honest_entries)
                    / weight
                    for which in ("model", "curve", "blend")
                }
                print(
                    f"\n    pooled {metric.upper():5s} model {pooled['model']:.2f}, "
                    f"curve {pooled['curve']:.2f}, blend {pooled['blend']:.2f}"
                    f"   blend vs curve {(pooled['blend'] - pooled['curve']) / pooled['curve']:+.2%}"
                    f", vs model {(pooled['blend'] - pooled['model']) / pooled['model']:+.2%}"
                )
        report[how] = entries

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
