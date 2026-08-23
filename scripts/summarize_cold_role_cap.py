"""Read the cap selection out loud, including where its criterion disagrees.

The selection is by mean CRPS, fixed before the numbers. Two things about that
need reporting alongside the winner rather than after someone asks.

The criterion was pre-registered without a tie-breaking rule, so ``argmin``
names a winner however small the margin. Every other comparison in this package
goes through a materiality floor -- 0.25% -- and this one did not. What that
floor would have chosen is printed as an observation, clearly not as the
selection, because it is being applied after seeing the numbers.

Coverage is printed beside CRPS at every candidate. The defect that motivated
the cold-role work was a coverage defect, so if the two measures point different
ways that is a judgement call for a person, not something to resolve silently by
having picked the convenient measure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MATERIAL = 0.0025


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=Path("scripts/validation_runs/cold_role_cap.json"),
    )
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))

    folds = sorted(k for k in report if k.isdigit())
    print(f"criterion: {report['criterion']}")
    print(f"incumbent: {report['incumbent']}")
    if report.get("dropped_seasons"):
        print(f"dropped seasons: {report['dropped_seasons']}")
    print()

    print("INNER FOLDS — what each candidate scored\n")
    for fold in folds:
        entry = report[fold]
        inner = entry["inner"]
        best = min(v["crps"] for v in inner.values())
        print(
            f"  holdout {fold}, selected on {entry['inner_season']}   "
            f"uncapped ratios "
            + ", ".join(f"{k}={v:.2f}" for k, v in entry["inner_ratios"].items())
        )
        print(
            f"    {'cap':>6s} {'crps':>9s} {'vs best':>9s} {'cov gap':>9s} "
            f"{'ppr cov80':>10s} {'ppr cov95':>10s}"
        )
        for cap, values in inner.items():
            gap = (values["crps"] - best) / best
            flag = " <- picked" if cap == entry["chosen_cap"] else ""
            within = "" if gap > MATERIAL else " *"
            print(
                f"    {cap:>6s} {values['crps']:>9.4f} {gap:>+8.3%}{within:2s} "
                f"{values['coverage_gap']:>9.4f} "
                f"{values['scores']['ppr']['coverage_80']:>10.3f} "
                f"{values['scores']['ppr']['coverage_95']:>10.3f}{flag}"
            )
        # What a materiality floor would have chosen. Applied after the fact and
        # labelled as such: it is an observation about the procedure, not the
        # procedure's output.
        tied = [c for c, v in inner.items() if (v["crps"] - best) / best <= MATERIAL]
        smallest = min(tied, key=lambda c: float("inf") if c == "None" else float(c))
        print(
            f"    * within {MATERIAL:.2%} of best: {tied}"
            f"   -> smallest of those would be {smallest}"
        )
        print()

    print("OUTER FOLDS — incumbent against the inner fold's pick\n")
    print(
        f"  {'fold':>6s} {'picked':>7s} {'inc crps':>9s} {'sel crps':>9s} "
        f"{'delta':>8s} {'inc cov95':>10s} {'sel cov95':>10s}"
    )
    deltas = []
    for fold in folds:
        entry = report[fold]
        inc = float(np.mean([entry["outer_incumbent"][s]["crps"] for s in ("standard", "half_ppr", "ppr")]))
        sel = float(np.mean([entry["outer_selected"][s]["crps"] for s in ("standard", "half_ppr", "ppr")]))
        deltas.append((sel - inc) / inc)
        print(
            f"  {fold:>6s} {entry['chosen_cap']:>7s} {inc:>9.4f} {sel:>9.4f} "
            f"{(sel - inc) / inc:>+7.2%} "
            f"{entry['outer_incumbent']['ppr']['coverage_95']:>10.3f} "
            f"{entry['outer_selected']['ppr']['coverage_95']:>10.3f}"
        )
    pooled = float(np.mean(deltas))
    wins = sum(1 for d in deltas if d < 0)
    print(
        f"\n  mean delta {pooled:+.2%} over {len(deltas)} folds, "
        f"selected better on {wins}/{len(deltas)}"
    )
    verdict = (
        "material" if abs(pooled) > MATERIAL else f"below the {MATERIAL:.2%} floor"
    )
    print(f"  that is {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
