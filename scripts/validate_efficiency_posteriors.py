"""Resumable walk-forward validation for posterior season efficiency.

The script fits each response independently so an interrupted ten-model run
can resume at the next target. Point accuracy is compared with the accepted
efficiency-v1 ridge/prior choice, while CRPS, interval coverage, and sampler
quality evaluate the new distributional contract.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import tempfile

import numpy as np

from ffmodel.evaluation.efficiency_posterior import (
    add_point_baseline_metrics,
    score_efficiency_posterior,
)
from ffmodel.evaluation.efficiency_season_average import (
    add_walk_forward_volume_features,
)
from ffmodel.evaluation.posterior_comparison import (
    atomic_write_json,
    combined_fingerprint,
    ensure_manifest,
    file_fingerprint,
    frame_fingerprint,
    load_json,
)
from ffmodel.features.season_average import build_season_average_data
from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    EFFICIENCY_MODEL_SPECS,
    POSTERIOR_MEAN_MODE,
    SeasonAveragePosteriorEfficiencyPipeline,
)


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    files = (
        Path(__file__),
        root / "src/ffmodel/features/season_average.py",
        root / "src/ffmodel/features/season_efficiency.py",
        root / "src/ffmodel/evaluation/efficiency_posterior.py",
        root / "src/ffmodel/evaluation/efficiency_season_average.py",
        root / "src/ffmodel/evaluation/metrics.py",
        root / "src/ffmodel/models/base.py",
        root / "src/ffmodel/models/efficiency_season_average.py",
    )
    return combined_fingerprint(
        {path.relative_to(root).as_posix(): file_fingerprint(path) for path in files}
    )


def _diagnostics(pipeline, target: str) -> dict[str, object]:
    result = pipeline.diagnostics()[target]
    return {key: value for key, value in result.items() if key != "summary"}


def _collect(output_dir: Path) -> list[dict[str, object]]:
    return [
        load_json(path)
        for path in sorted(output_dir.glob("holdout-*/**/metrics.json"))
    ]


def _weighted(records, name: str) -> float:
    values = np.asarray([float(record[name]) for record in records], dtype=float)
    weights = np.asarray(
        [float(record["opportunities"]) for record in records], dtype=float
    )
    return float(np.average(values, weights=np.clip(weights, 1.0, None)))


def _report(args) -> dict[str, object]:
    records = _collect(args.output_dir)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["target"]].append(record)
    pooled = {}
    gates = {}
    for target, target_records in grouped.items():
        target_records = sorted(target_records, key=lambda record: record["season"])
        metrics = {
            name: _weighted(target_records, name)
            for name in (
                "posterior_weighted_mae",
                "posterior_weighted_crps",
                "accepted_point_weighted_mae",
                "coverage_80",
                "coverage_95",
            )
        }
        metrics["relative_improvement"] = (
            metrics["accepted_point_weighted_mae"]
            - metrics["posterior_weighted_mae"]
        ) / metrics["accepted_point_weighted_mae"]
        metrics["fold_wins"] = sum(
            record["posterior_weighted_mae"]
            < record["accepted_point_weighted_mae"]
            for record in target_records
        )
        metrics["folds"] = len(target_records)
        pooled[target] = metrics
        expected = set(args.holdouts)
        scored = {int(record["season"]) for record in target_records}
        sampler_passed = all(record["diagnostics"]["passed"] for record in target_records)
        mean_mode = POSTERIOR_MEAN_MODE[target]
        if mean_mode == "posterior":
            point_accuracy_passed = bool(
                metrics["relative_improvement"] > 0
                and metrics["fold_wins"] >= min(2, len(expected))
            )
        else:
            point_accuracy_passed = bool(
                abs(metrics["relative_improvement"]) <= 1e-8
                and all(
                    abs(
                        record["posterior_weighted_mae"]
                        - record["accepted_point_weighted_mae"]
                    )
                    <= 1e-10
                    for record in target_records
                )
            )
        gates[target] = {
            "complete": scored == expected,
            "sampler_passed": sampler_passed,
            "point_accuracy_passed": point_accuracy_passed,
            "coverage_passed": bool(
                0.70 <= metrics["coverage_80"] <= 0.90
                and 0.90 <= metrics["coverage_95"] <= 0.99
            ),
        }
        gates[target]["passed"] = all(gates[target].values())
    report = {
        "configuration": {
            "seasons": args.seasons,
            "holdouts": args.holdouts,
            "targets": args.targets,
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "nuts_sampler": args.nuts_sampler,
            "ridge_alpha": args.ridge_alpha,
        },
        "folds": records,
        "pooled": pooled,
        "gates": gates,
    }
    atomic_write_json(args.output_dir / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2014, 2025)))
    parser.add_argument("--holdouts", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=tuple(EFFICIENCY_MODEL_BY_TARGET),
        default=[spec.target for spec in EFFICIENCY_MODEL_SPECS],
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--nuts-sampler", choices=("pymc", "nutpie"), default="nutpie")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=500.0)
    parser.add_argument("--volume-alpha", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/season-average-validation/efficiency-v2-posterior"),
    )
    args = parser.parse_args(argv)
    if args.score_only:
        args.resume = True
    if args.nuts_sampler == "nutpie" and "NUMBA_CACHE_DIR" not in os.environ:
        # Numba-generated function names can exceed Windows' path limit when
        # rooted under this intentionally descriptive project directory.
        cache = Path(tempfile.gettempdir()) / "ffmodel-numba"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(cache)

    data = build_season_average_data(
        args.seasons, source="nflverse", roster_mode="point_in_time"
    )
    rows = add_walk_forward_volume_features(
        data, include_efficiency=True, alpha=args.volume_alpha
    )
    experiment = {
        "seasons": args.seasons,
        "holdouts": args.holdouts,
        "targets": args.targets,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "nuts_sampler": args.nuts_sampler,
        "seed": args.seed,
        "ridge_alpha": args.ridge_alpha,
        "volume_alpha": args.volume_alpha,
        "source_fingerprint": _source_fingerprint(),
        "rows_fingerprint": frame_fingerprint(
            rows, keys=("season", "team", "player_key")
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(
        args.output_dir / "experiment.json", experiment, resume=args.resume
    )

    fit_kwargs = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "nuts_sampler": args.nuts_sampler,
        "seed": args.seed,
    }
    for holdout in args.holdouts:
        train = rows[rows["season"] < holdout].copy().reset_index(drop=True)
        test = rows[rows["season"] == holdout].copy().reset_index(drop=True)
        if train.empty or test.empty:
            raise ValueError(f"holdout {holdout} has an empty train or test fold")
        for target_index, target in enumerate(args.targets):
            target_dir = args.output_dir / f"holdout-{holdout}" / target
            posterior_dir = target_dir / "posterior"
            metrics_path = target_dir / "metrics.json"
            if args.resume and posterior_dir.exists():
                pipeline = SeasonAveragePosteriorEfficiencyPipeline.load(posterior_dir)
                print(f"resumed {holdout} {target}", flush=True)
            else:
                if args.score_only:
                    raise FileNotFoundError(
                        f"score-only posterior is missing: {posterior_dir}"
                    )
                print(f"fitting {holdout} {target}", flush=True)
                pipeline = SeasonAveragePosteriorEfficiencyPipeline(
                    ridge_alpha=args.ridge_alpha
                ).fit(
                    train,
                    targets=[target],
                    **{
                        **fit_kwargs,
                        "seed": args.seed + target_index + 100 * (holdout - min(args.holdouts)),
                    },
                )
                pipeline.save(posterior_dir)

            model = pipeline.models[target]
            record = score_efficiency_posterior(
                model, test, draws=args.draws * args.chains, seed=args.seed
            )
            record = add_point_baseline_metrics(
                record,
                train_rows=train,
                test_rows=test,
                model=model,
                ridge_alpha=args.ridge_alpha,
            )
            record["season"] = int(holdout)
            record["diagnostics"] = _diagnostics(pipeline, target)
            atomic_write_json(metrics_path, record)
            report = _report(args)
            print(
                f"scored {holdout} {target}: posterior={record['posterior_weighted_mae']:.6f} "
                f"benchmark={record['accepted_point_weighted_mae']:.6f}",
                flush=True,
            )

    report = _report(args)
    print(report["gates"], flush=True)


if __name__ == "__main__":
    main()
