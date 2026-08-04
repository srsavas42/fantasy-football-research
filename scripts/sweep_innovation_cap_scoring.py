"""Does the innovation cap own the tail deficit on total fantasy points?

Two findings collide here.

1. The 95% intervals on total points are about four points too narrow, the
   deficit is in the volume layer, and it is shape rather than width: a global
   2x stretch lands cov95 on nominal while destroying cov80.
2. `innovation_cap` was promoted at 0.25 against measured role churn of 1.43
   for targets and at least 2.00 for carries, so the allocators represent
   roughly a sixth of the season-to-season role movement the data shows.

The selection that promoted 0.25 used mean distance from nominal coverage over
the carry and target streams as its criterion. That statistic was later shown to
be uninterpretable on exactly those streams: 49.6% of carry rows and 27.9% of
target rows are zero, every interval containing zero covers them, and the
population rate therefore cannot reach nominal however good the model is. A
criterion that reads guaranteed coverage as over-wide intervals rewards
narrowing, which is what it did.

So the cap is a live suspect for the missing tail mass, and this measures it
where it matters -- on the published output rather than on an intermediate
stream, and against a proper scoring rule as well as coverage.

The sweep is cheap for the same reason the original was: `role_innovation_scale`
is consumed at prediction time and enters no likelihood, so one posterior serves
every candidate.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.efficiency_posterior import score_fantasy_points_posterior
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.base import calibrate_innovation_scale
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline

CANDIDATES: list[float | None] = [0.25, 0.50, 1.00, 1.50, 2.00, None]
SCORING_FORMATS = ("standard", "half_ppr", "ppr")


def apply_cap(volume_model, train_rows: pd.DataFrame, cap: float | None) -> dict[str, float]:
    """Set each allocator's scale as though it had been fitted under ``cap``."""
    applied: dict[str, float] = {}
    for stream, model in (
        ("target", volume_model.target_model),
        ("carry", volume_model.carry_model),
    ):
        prepared = model._prepare(train_rows)
        measured = model._estimate_role_innovation(prepared)
        target = measured if cap is None else min(measured, float(cap))
        if model.calibrated_innovation:
            allocation, mask = model._innovation_rooms(prepared)
            scale = calibrate_innovation_scale(
                allocation, mask, target, seed=model.innovation_calibration_seed
            )
        else:
            scale = target
        model.role_innovation_scale = float(scale)
        applied[stream] = float(scale)
    return applied


def binomial_z(coverage: float, n: int, nominal: float) -> float:
    misses = (1.0 - coverage) * n
    expected = (1.0 - nominal) * n
    return (misses - expected) / np.sqrt(n * nominal * (1.0 - nominal))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # In-window by default. This sweep is a diagnostic and its output must not
    # be used to pick a value: 2025 is the one season no choice in this package
    # has seen, and a sweep is exactly the shape of thing that spends it.
    parser.add_argument("--holdout", type=int, default=2024)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025"))
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, default=Path("scripts/validation_runs/cap_scoring_2025.json")
    )
    args = parser.parse_args(argv)

    pr = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    tr = pd.read_pickle(args.cache_dir / "team_rows.pkl")
    train = SeasonAverageData(
        tr[tr.season < args.holdout].copy(), pr[pr.season < args.holdout].copy()
    )
    test = SeasonAverageData(
        tr[tr.season == args.holdout].copy(), pr[pr.season == args.holdout].copy()
    )

    pipeline = SeasonAverageScoringPipeline()
    sample_kwargs = {"draws": args.draws, "tune": args.draws, "chains": 4}
    pipeline.fit(
        train,
        volume_sample_kwargs=sample_kwargs,
        efficiency_sample_kwargs=sample_kwargs,
    )
    print("fitted", flush=True)

    report: dict[str, object] = {"holdout": args.holdout, "caps": {}}
    print(f"\nTOTAL FANTASY POINTS BY INNOVATION CAP, holdout {args.holdout}\n")
    print(
        f"  {'cap':>6s} {'tgt scale':>9s} {'car scale':>9s} {'scoring':>9s} "
        f"{'cov80':>7s} {'z80':>7s} {'cov95':>7s} {'z95':>7s} {'MAE':>8s} {'CRPS':>8s}"
    )
    for cap in CANDIDATES:
        scales = apply_cap(pipeline.volume_model, train.player_rows, cap)
        prediction = pipeline.predict_samples(test, seed=42)
        entry: dict[str, object] = {"scales": scales}
        for scoring in SCORING_FORMATS:
            summary = score_fantasy_points_posterior(prediction, scoring=scoring)
            entry[scoring] = summary
            n = int(summary["n"])
            print(
                f"  {str(cap):>6s} {scales['target']:>9.4f} {scales['carry']:>9.4f} "
                f"{scoring:>9s} "
                f"{summary['coverage_80']:>7.3f} "
                f"{binomial_z(summary['coverage_80'], n, 0.80):>+7.2f} "
                f"{summary['coverage_95']:>7.3f} "
                f"{binomial_z(summary['coverage_95'], n, 0.95):>+7.2f} "
                f"{summary['mae']:>8.3f} {summary['crps']:>8.3f}",
                flush=True,
            )
        report["caps"][str(cap)] = entry
        print("", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
