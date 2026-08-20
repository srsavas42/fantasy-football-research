"""Walk-forward screen for injury-informed season availability.

The challenger differs only in its leakage-safe injury-history and current
injury recovery features.  It is intentionally evaluated at the availability
layer first, before it is allowed to change the accepted volume pipeline.

Example:

    python scripts/validate_injury_availability.py --draws 300 --tune 300 \
        --chains 2 --nuts-sampler nutpie --report-json reports/injury.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.features.season_average import build_season_average_data
from ffmodel.features.season_injury import INJURY_AVAILABILITY_FEATURES
from ffmodel.models.season_availability import SeasonAvailabilityModel


def _distribution_metrics(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    mean = samples.mean(axis=1)
    return {
        "mae": float(np.abs(observed - mean).mean()),
        "rmse": float(np.sqrt(np.mean((observed - mean) ** 2))),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
    }


def _evaluate(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    injury_features: bool,
    fit_kwargs: dict[str, object],
    seed: int,
) -> dict[str, object]:
    model = SeasonAvailabilityModel(
        extra_features=INJURY_AVAILABILITY_FEATURES if injury_features else ()
    ).fit(train, **fit_kwargs)
    prediction = model.predict_samples(test, seed=seed)
    observed = pd.to_numeric(
        prediction.rows["observed_availability"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    samples = prediction.availability
    injury_exposed = (
        pd.to_numeric(
            prediction.rows.get("current_injury_reported", pd.Series(0, index=prediction.rows.index)),
            errors="coerce",
        ).fillna(0).gt(0)
        | pd.to_numeric(
            prediction.rows.get("prior_injury_report_weeks_3yr", pd.Series(0, index=prediction.rows.index)),
            errors="coerce",
        ).fillna(0).gt(0)
    ).to_numpy()
    output: dict[str, object] = {
        "metrics": _distribution_metrics(observed, samples),
        "feature_names": model.feature_names,
        "n_players": int(len(prediction.rows)),
        "n_injury_exposed": int(injury_exposed.sum()),
    }
    if injury_exposed.any():
        output["injury_exposed_metrics"] = _distribution_metrics(
            observed[injury_exposed], samples[injury_exposed]
        )
    return output


def _delta(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, float]:
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    return {
        name: float(candidate_metrics[name] - baseline_metrics[name])
        for name in candidate_metrics
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(range(2014, 2025)),
    )
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--nuts-sampler", choices=("pymc", "nutpie"), default="nutpie")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="use prebuilt walk-forward frames instead of a fresh "
                             "nflverse pull, so the result lines up with every "
                             "other run on this branch")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)

    if args.nuts_sampler == "nutpie" and "NUMBA_CACHE_DIR" not in os.environ:
        cache = Path(tempfile.gettempdir()) / "ffmodel-numba"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(cache)

    if args.cache_dir is not None:
        # Use the frames every other comparison on this branch uses. A fresh
        # pull differs from the cache in more than the passage of time --
        # sixty-nine of two hundred eighty-nine columns differed between two
        # caches here with identical row counts -- so a result measured on its
        # own private build cannot be lined up against anything else.
        player_rows = pd.read_pickle(Path(args.cache_dir) / "player_rows.pkl")
        team_rows = pd.read_pickle(Path(args.cache_dir) / "team_rows.pkl")
        missing = [c for c in INJURY_AVAILABILITY_FEATURES if c not in player_rows]
        if missing:
            raise SystemExit(
                f"{args.cache_dir} is missing {missing}; these frames predate "
                "the injury features and the challenger would silently fit the "
                "baseline"
            )
        data = SimpleNamespace(player_rows=player_rows, team_rows=team_rows)
    else:
        data = build_season_average_data(
            args.seasons,
            source="nflverse",
            roster_mode="point_in_time",
        )
    rows = data.player_rows.copy()
    fit_kwargs = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "nuts_sampler": args.nuts_sampler,
    }
    report: dict[str, object] = {
        "seasons": args.seasons,
        "holdouts": args.holdouts,
        "fit": fit_kwargs,
        "injury_feature_contract": list(INJURY_AVAILABILITY_FEATURES),
        "folds": {},
    }
    for offset, holdout in enumerate(args.holdouts):
        train = rows[rows["season"].lt(holdout)].copy()
        test = rows[rows["season"].eq(holdout)].copy()
        if train.empty or test.empty:
            raise ValueError(f"holdout {holdout} needs both training and test rows")
        baseline = _evaluate(
            train=train,
            test=test,
            injury_features=False,
            fit_kwargs=fit_kwargs,
            seed=args.seed + 2 * offset,
        )
        challenger = _evaluate(
            train=train,
            test=test,
            injury_features=True,
            fit_kwargs=fit_kwargs,
            seed=args.seed + 2 * offset + 1,
        )
        fold = {
            "baseline": baseline,
            "injury_challenger": challenger,
            "delta_challenger_minus_baseline": _delta(challenger, baseline),
        }
        report["folds"][str(holdout)] = fold
        delta = fold["delta_challenger_minus_baseline"]
        print(
            f"{holdout}: CRPS baseline={baseline['metrics']['crps']:.4f} "
            f"challenger={challenger['metrics']['crps']:.4f} "
            f"delta={delta['crps']:+.4f}; "
            f"MAE delta={delta['mae']:+.4f}; "
            f"80% coverage delta={delta['coverage_80']:+.4f}"
        )

    deltas = pd.DataFrame(
        [
            fold["delta_challenger_minus_baseline"]
            for fold in report["folds"].values()
        ]
    )
    report["mean_delta_challenger_minus_baseline"] = {
        name: float(value) for name, value in deltas.mean().items()
    }
    print("mean delta (challenger - baseline): " + json.dumps(
        report["mean_delta_challenger_minus_baseline"], sort_keys=True
    ))
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
