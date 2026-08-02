"""Run the acceptance gate over two walk-forward JSON outputs.

    python scripts/compare_validation_runs.py \
        scripts/validation_runs/wf_postseason.json \
        scripts/validation_runs/wf_meanpreserving.json

Exits non-zero when the candidate is not acceptable, so it can gate a promotion
in CI rather than in prose. ``--protected`` names streams that may not regress
beyond ``--protected-tolerance`` whatever they buy elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffmodel.evaluation.acceptance import compare_runs, format_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--protected",
        nargs="*",
        default=["pass_qb", "qb_workload"],
        help="streams a change may not damage in exchange for gains elsewhere",
    )
    parser.add_argument("--protected-tolerance", type=float, default=0.005)
    args = parser.parse_args(argv)

    report = compare_runs(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
        protected=args.protected,
        protected_tolerance=args.protected_tolerance,
    )
    print(
        format_report(
            report, baseline=args.baseline.stem, candidate=args.candidate.stem
        )
    )
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
