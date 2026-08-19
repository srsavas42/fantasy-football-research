"""Did preseason consensus add anything the play-by-play history did not?

Reads the paired scoring walk-forward runs and reports the change per holdout
and pooled, on the metrics the gate uses. Three checks stand between a number
here and a claim about the feature.

**The arms must share their frames.** A candidate scored on one cache against a
baseline scored on another is not a controlled comparison, and it has happened
here: two caches with identical row counts differed in sixty-nine of two
hundred eighty-nine columns. Mismatched fingerprints exit non-zero rather than
printing a table.

**The feature must have reached the design matrix.** A flag that sets a name on
a list is not the same as a column the model saw; ``_matrix`` drops names it
cannot find, and an arm can be enabled, raise nothing, and fit exactly the
baseline. The fitted feature names are checked, not assumed.

**A change has to clear a floor to be a finding.** The argmin of a noisy sweep
is not a result. Anything inside :data:`MATERIAL` is reported as no movement,
because a verdict rule without a floor announces 0.01% as a discovery -- which
this branch has done twice.

One thing this cannot settle. ADP is a forecast, so a model that reads it is
partly following the market, and a win here does not mean the model found
something the market had not. It means the two together beat the history alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MATERIAL = 0.0025
# Lower is better for all of these, so a negative delta is a gain.
METRICS = ("mae", "crps")
COVERAGE = ("coverage_80", "coverage_95")
ADP_PREFIX = "adp_"


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"no run at {path}")
    return json.loads(path.read_text())


def folds(report: dict) -> list[str]:
    return sorted(k for k in report if not k.startswith("_"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="adpoff")
    parser.add_argument("--candidate", default="adpon")
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--runs", type=Path, default=Path("scripts/validation_runs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    base = load(args.runs / f"scoring_{args.baseline}.json")
    cand = load(args.runs / f"scoring_{args.candidate}.json")

    base_frames = base.get("_frames", {})
    cand_frames = cand.get("_frames", {})
    if base_frames != cand_frames:
        print("the two arms were not scored on the same frames:")
        print(f"  {args.baseline:10s} {base_frames}")
        print(f"  {args.candidate:10s} {cand_frames}")
        return 2

    shared = [f for f in folds(base) if f in set(folds(cand))]
    if not shared:
        raise SystemExit("the two runs share no completed holdout yet")
    missing = sorted(set(folds(base)) ^ set(folds(cand)))
    if missing:
        print(f"note: {missing} finished in only one arm and is excluded\n")

    # Was the feature actually in the design? Only the candidate should carry
    # it, and it should carry it in every room the flag names.
    saw: dict[str, list[str]] = {}
    for label, report in ((args.baseline, base), (args.candidate, cand)):
        seen: set[str] = set()
        recorded = False
        for fold in shared:
            features = report[fold].get("volume_features")
            if features is None:
                continue
            recorded = True
            for names in features.values():
                seen.update(n for n in names if n.startswith(ADP_PREFIX))
        saw[label] = sorted(seen) if recorded else ["<not recorded>"]
    print(f"ADP columns in the fitted design:")
    for label, names in saw.items():
        print(f"  {label:10s} {names or 'none'}")
    if saw[args.candidate] == []:
        print(
            "\nthe candidate fitted no ADP column: the arm is the baseline and "
            "any delta below is noise, not a null result"
        )
        return 3
    print()

    rows = []
    header = (
        f"  {'holdout':>8s} {'base MAE':>9s} {'adp MAE':>9s} {'d MAE':>8s} "
        f"{'base CRPS':>10s} {'adp CRPS':>9s} {'d CRPS':>8s} "
        f"{'cov95 b':>8s} {'cov95 a':>8s}"
    )
    print(f"{args.scoring.upper()} total fantasy points\n")
    print(header)
    totals = {metric: [0.0, 0.0] for metric in METRICS}
    weight = 0.0
    for fold in shared:
        b = base[fold][args.scoring]
        c = cand[fold][args.scoring]
        n = float(b.get("n", 1))
        entry: dict[str, object] = {"holdout": fold, "n": int(n)}
        for metric in METRICS:
            entry[f"base_{metric}"] = float(b[metric])
            entry[f"adp_{metric}"] = float(c[metric])
            entry[f"delta_{metric}"] = (float(c[metric]) - float(b[metric])) / float(
                b[metric]
            )
            totals[metric][0] += float(b[metric]) * n
            totals[metric][1] += float(c[metric]) * n
        for name in COVERAGE:
            entry[f"base_{name}"] = float(b[name])
            entry[f"adp_{name}"] = float(c[name])
        weight += n
        rows.append(entry)
        print(
            f"  {fold:>8s} {entry['base_mae']:>9.3f} {entry['adp_mae']:>9.3f} "
            f"{entry['delta_mae']:>+7.2%} {entry['base_crps']:>10.3f} "
            f"{entry['adp_crps']:>9.3f} {entry['delta_crps']:>+7.2%} "
            f"{entry['base_coverage_95']:>8.3f} {entry['adp_coverage_95']:>8.3f}"
        )

    pooled = {
        metric: (totals[metric][1] - totals[metric][0]) / totals[metric][0]
        for metric in METRICS
    }
    wins = {
        metric: sum(1 for r in rows if r[f"delta_{metric}"] < 0) for metric in METRICS
    }
    print(
        f"\n  pooled (n-weighted): MAE {pooled['mae']:+.2%}, "
        f"CRPS {pooled['crps']:+.2%}"
    )
    print(
        f"  holdouts improved:   MAE {wins['mae']}/{len(rows)}, "
        f"CRPS {wins['crps']}/{len(rows)}"
    )

    def verdict() -> str:
        gains = [m for m in METRICS if pooled[m] < -MATERIAL]
        losses = [m for m in METRICS if pooled[m] > MATERIAL]
        if gains and not losses:
            swept = all(wins[m] == len(rows) for m in gains)
            return (
                f"ADP helps on {', '.join(gains)}"
                + (", winning every holdout" if swept else ", but not on every holdout")
            )
        if losses and not gains:
            return f"ADP costs {', '.join(losses)}: the history already carries it"
        if gains and losses:
            return f"mixed: gains {gains}, losses {losses}"
        return (
            f"nothing moves beyond {MATERIAL:.2%}: consensus adds nothing the "
            "play-by-play history does not already carry"
        )

    line = verdict()
    print(f"\n  {line}")

    payload = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "scoring": args.scoring,
        "frames": base_frames,
        "adp_columns_fitted": saw,
        "material_threshold": MATERIAL,
        "folds": rows,
        "pooled": pooled,
        "holdouts_improved": wins,
        "verdict": line,
    }
    output = args.output or args.runs / "adp_ablation.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
