"""Judge interval coverage in counts against a binomial reference, not by eye.

    python scripts/coverage_calibration.py scripts/validation_runs/wf_calibrated.json

A coverage *rate* on a single fold is a poor statistic when the event being
counted is rare. A 95% interval over 84 quarterback rows expects about four
misses, with a standard deviation of two. A fold that records 1.000 has zero
misses, which reads as "the intervals cover everything" and looks like
over-widening — but P(0 | n=84, p=0.05) is 0.0135, so seeing it in at least one
of three folds has probability 0.04. That is unusual and entirely ordinary
noise, and a per-fold rate cannot tell you which.

This reports misses per fold with the binomial z, and pools the folds, where the
sample is large enough for the answer to be stable. Reading the calibration
work of 2026-08-03 that way: pooled 95% misses went from 45 to 13 against 12.7
expected, and the fold that looked like over-widening was one draw from a
distribution centred exactly where it should be.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffmodel.evaluation.acceptance import COVERAGE_NOMINAL


def _binomial_interval(n: int, p: float, level: float = 0.95) -> tuple[int, int]:
    """Central ``level`` interval for Binomial(n, p), by exact summation.

    The probability mass is built in log space. ``math.comb`` is exact but its
    result for the target stream's n is an integer too large to convert to a
    float, so the direct product overflows before it can be scaled down.
    """
    if n <= 0:
        return 0, 0

    def log_pmf(k: int) -> float:
        return (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * math.log(p)
            + (n - k) * math.log1p(-p)
        )

    pmf = [math.exp(log_pmf(k)) for k in range(n + 1)]
    tail = (1.0 - level) / 2
    cumulative, low = 0.0, 0
    for k, mass in enumerate(pmf):
        cumulative += mass
        if cumulative >= tail:
            low = k
            break
    cumulative, high = 0.0, n
    for k in range(n, -1, -1):
        cumulative += pmf[k]
        if cumulative >= tail:
            high = k
            break
    return low, high


def report(run: dict, stream: str) -> str:
    lines: list[str] = []
    folds = sorted(f for f in run if isinstance(run[f], dict) and stream in run[f])
    if not folds:
        return f"{stream}: not present in this run"
    for metric, nominal in sorted(COVERAGE_NOMINAL.items()):
        if metric not in run[folds[0]][stream]:
            continue
        lines.append(f"\n{stream}/{metric} — nominal {nominal:.0%}")
        lines.append(
            f"  {'fold':>6s} {'n':>5s} {'misses':>7s} {'expected':>9s} {'z':>7s}"
        )
        total_n = total_missed = 0
        for fold in folds:
            payload = run[fold][stream]
            n = int(payload.get("n", 0))
            if not n:
                continue
            missed = int(round(n * (1.0 - float(payload[metric]))))
            expected = n * (1.0 - nominal)
            sd = math.sqrt(n * (1.0 - nominal) * nominal)
            z = (missed - expected) / sd if sd else float("nan")
            lines.append(
                f"  {fold:>6s} {n:>5d} {missed:>7d} {expected:>9.1f} {z:>+7.2f}"
            )
            total_n += n
            total_missed += missed
        if not total_n:
            continue
        expected = total_n * (1.0 - nominal)
        sd = math.sqrt(total_n * (1.0 - nominal) * nominal)
        low, high = _binomial_interval(total_n, 1.0 - nominal)
        inside = low <= total_missed <= high
        lines.append(
            f"  {'POOLED':>6s} {total_n:>5d} {total_missed:>7d} {expected:>9.1f} "
            f"{(total_missed - expected) / sd:>+7.2f}"
            f"   95% range [{low}, {high}] — {'calibrated' if inside else 'MISCALIBRATED'}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--streams",
        nargs="*",
        default=None,
        help="streams to report; default is every stream carrying an n",
    )
    args = parser.parse_args(argv)

    run = json.loads(args.run.read_text(encoding="utf-8"))
    streams = args.streams
    if not streams:
        first = next(f for f in run if isinstance(run[f], dict))
        streams = sorted(
            name
            for name, value in run[first].items()
            if isinstance(value, dict) and "n" in value
        )
    print(args.run.stem)
    for stream in streams:
        print(report(run, stream))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
