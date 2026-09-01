"""Walk-forward metrics at every layer: availability, volume, efficiency, totals.

Fits the shipping ``SeasonAverageScoringPipeline`` per holdout -- the same
configuration ``project_season.py`` ships, reserve/IR split and the reserve
efficiency flag both on by default -- and scores each layer against what it is
actually trying to predict, not just the final point total a good volume call
and a bad efficiency call can cancel inside of.

    availability   observed_availability vs the hurdle's games-active draws
    volume         targets / carries / pass attempts vs the roster allocation
    efficiency     each response's own rate vs its draws, on the same
                   exposure-and-position population the pipeline fits it on
    totals         fantasy points vs the coherent stat-line draws, overall and
                   on the drafted and full-season populations

Folds run one per process; a full pipeline fit at moderate draws is heavy
enough that this container has died mid-run on less.

    python scripts/validate_full_pipeline.py --holdouts 2023 --report-json a.json
    python scripts/validate_full_pipeline.py --merge a.json b.json c.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.efficiency_posterior import score_fantasy_points_posterior
from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.efficiency_season_average import EFFICIENCY_MODEL_SPECS
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline

MATERIAL = 0.0025


def _metrics(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    mean = samples.mean(axis=1)
    return {
        "mae": float(np.abs(observed - mean).mean()),
        "rmse": float(np.sqrt(np.mean((observed - mean) ** 2))),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
        "n": int(len(observed)),
    }


def _availability_metrics(prediction, exposure_floor: int) -> dict[str, object]:
    rows = prediction.volume.player_rows
    observed = pd.to_numeric(
        rows["observed_availability"], errors="coerce"
    ).to_numpy(dtype=float)
    samples = np.asarray(prediction.volume.availability, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(samples).all(axis=1)
    out = {"overall": _metrics(observed[valid], samples[valid])}
    reserve = pd.to_numeric(
        rows.get("roster_reserve"), errors="coerce"
    ).fillna(0).gt(0).to_numpy() & valid
    if reserve.sum() >= 5:
        out["reserve"] = _metrics(observed[reserve], samples[reserve])
    ir = pd.to_numeric(
        rows.get("roster_injured_reserve"), errors="coerce"
    ).fillna(0).gt(0).to_numpy() & valid
    if ir.sum() >= 5:
        out["injured_reserve"] = _metrics(observed[ir], samples[ir])
    return out


def _volume_metrics(prediction) -> dict[str, dict[str, object]]:
    rows = prediction.player_rows.reset_index(drop=True)
    out = {}
    for stream, count_col, samples in (
        ("carry", "rush_att", prediction.volume.carries),
        ("target", "targets", prediction.volume.targets),
        ("pass", "pass_att", prediction.volume.pass_attempts),
    ):
        observed = pd.to_numeric(rows.get(count_col), errors="coerce").to_numpy(dtype=float)
        samples = np.asarray(samples, dtype=float)
        valid = np.isfinite(observed) & np.isfinite(samples).all(axis=1)
        if valid.sum() >= 20:
            out[stream] = _metrics(observed[valid], samples[valid])
    return out


def _efficiency_metrics(prediction, exposure_floor: int) -> dict[str, dict[str, object]]:
    rows = prediction.player_rows.reset_index(drop=True)
    out = {}
    for spec in EFFICIENCY_MODEL_SPECS:
        if spec.target not in prediction.efficiency.rates:
            continue
        observed = pd.to_numeric(rows.get(spec.target), errors="coerce").to_numpy(dtype=float)
        samples = np.asarray(prediction.efficiency.rates[spec.target], dtype=float)
        exposure = pd.to_numeric(rows.get(spec.exposure), errors="coerce").fillna(0)
        floor = min(spec.min_exposure, exposure_floor)
        eligible = (
            exposure.ge(floor).to_numpy()
            & rows["position"].astype(str).str.upper().isin(spec.positions).to_numpy()
        )
        valid = eligible & np.isfinite(observed) & np.isfinite(samples).all(axis=1)
        if valid.sum() >= 20:
            out[spec.target] = _metrics(observed[valid], samples[valid])
    return out


def _totals_metrics(prediction) -> dict[str, dict[str, object]]:
    rows = prediction.player_rows.reset_index(drop=True)
    out = {}
    for scoring in ("ppr",):
        out[f"{scoring}_overall"] = score_fantasy_points_posterior(prediction, scoring=scoring)
        drafted = pd.to_numeric(rows.get("adp_drafted"), errors="coerce").eq(1).to_numpy()
        if drafted.sum() >= 30:
            out[f"{scoring}_drafted"] = score_fantasy_points_posterior(
                prediction, scoring=scoring, subset=drafted
            )
        played = pd.to_numeric(rows.get("games"), errors="coerce")
        slate = pd.to_numeric(rows.get("team_games"), errors="coerce")
        full = (played / slate.replace(0, float("nan"))).ge(0.94).to_numpy()
        if full.sum() >= 30:
            out[f"{scoring}_full_season"] = score_fantasy_points_posterior(
                prediction, scoring=scoring, subset=full
            )
    return out


def _run_fold(holdout: int, cache_dir: Path, *, draws: int, tune: int, chains: int, seed: int):
    player_rows = pd.read_pickle(cache_dir / "player_rows.pkl")
    team_rows = pd.read_pickle(cache_dir / "team_rows.pkl")
    player_rows = player_rows[player_rows.season.lt(2026)]
    team_rows = team_rows[team_rows.season.lt(2026)]
    train = SeasonAverageData(
        team_rows[team_rows.season.lt(holdout)].copy(),
        player_rows[player_rows.season.lt(holdout)].copy(),
    )
    test = SeasonAverageData(
        team_rows[team_rows.season.eq(holdout)].copy(),
        player_rows[player_rows.season.eq(holdout)].copy(),
    )
    if train.player_rows.empty or test.player_rows.empty:
        raise SystemExit(f"holdout {holdout} has no train or test rows")

    pipeline = SeasonAverageScoringPipeline()
    sample_kwargs = {"draws": draws, "tune": tune, "chains": chains}
    started = time.perf_counter()
    pipeline.fit(
        train, volume_sample_kwargs=sample_kwargs, efficiency_sample_kwargs=sample_kwargs
    )
    prediction = pipeline.predict_samples(test, seed=seed)
    elapsed = round(time.perf_counter() - started, 1)

    exposure_floor = pipeline.efficiency_model.exposure_floor or 5
    fold = {
        "seconds": elapsed,
        "availability": _availability_metrics(prediction, exposure_floor),
        "volume": _volume_metrics(prediction),
        "efficiency": _efficiency_metrics(prediction, exposure_floor),
        "totals": _totals_metrics(prediction),
    }
    del pipeline, prediction
    gc.collect()
    return fold


def _print_layer(name: str, folds: dict, holdouts: list[int]):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    keys = sorted({k for h in holdouts for k in folds[str(h)][name]})
    for key in keys:
        rows = []
        for h in holdouts:
            block = folds[str(h)][name].get(key)
            if block:
                rows.append(block)
        if not rows:
            continue
        n = int(np.mean([r["n"] for r in rows]))
        mae = np.mean([r["mae"] for r in rows])
        crps = np.mean([r["crps"] for r in rows])
        cov = np.mean([r.get("coverage_80", np.nan) for r in rows])
        print(f"  {key:24s} n~{n:<6} MAE {mae:>9.4f}  CRPS {crps:>9.4f}  cov80 {cov:>6.3f}")


def _print_totals(folds: dict, holdouts: list[int]):
    print(f"\n{'=' * 78}\ntotals (fantasy points)\n{'=' * 78}")
    keys = sorted({k for h in holdouts for k in folds[str(h)]["totals"]})
    for key in keys:
        rows = [folds[str(h)]["totals"][key] for h in holdouts if key in folds[str(h)]["totals"]]
        if not rows:
            continue
        n = int(np.mean([r["n"] for r in rows]))
        mae = np.mean([r["mae"] for r in rows])
        crps = np.mean([r["crps"] for r in rows])
        cov80 = np.mean([r["coverage_80"] for r in rows])
        cov95 = np.mean([r["coverage_95"] for r in rows])
        print(
            f"  {key:20s} n~{n:<6} MAE {mae:>7.2f}  CRPS {crps:>7.2f}  "
            f"cov80 {cov80:>6.3f}  cov95 {cov95:>6.3f}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", type=Path, default=Path("reports/full_pipeline.json"))
    parser.add_argument("--merge", type=Path, nargs="+", default=None)
    args = parser.parse_args(argv)

    if args.merge:
        folds: dict = {}
        for path in args.merge:
            folds.update(json.loads(path.read_text("utf-8"))["folds"])
        holdouts = sorted(int(k) for k in folds)
        _print_layer("availability", folds, holdouts)
        _print_layer("volume", folds, holdouts)
        _print_layer("efficiency", folds, holdouts)
        _print_totals(folds, holdouts)
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps({"holdouts": holdouts, "folds": folds}, indent=2, default=str), "utf-8"
        )
        print(f"\nwrote {args.report_json}")
        return 0

    report: dict[str, object] = {"holdouts": args.holdouts, "folds": {}}
    for holdout in args.holdouts:
        fold = _run_fold(
            holdout, args.cache_dir,
            draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed,
        )
        report["folds"][str(holdout)] = fold
        print(f"holdout {holdout} done in {fold['seconds']}s", flush=True)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, default=str), "utf-8")
    print(f"wrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
