"""Choose the role-innovation cap on an inner fold, cheaply.

``innovation_cap`` bounds how much season-to-season role churn the target and
carry allocators will represent. It defaults to 0.50 and it *binds on every
fit*: the measured dispersion is 1.43 for targets and 2.00 for carries, so the
cap is not a safety rail that occasionally catches an outlier — it is the
operative parameter, and its value has never been validated. Worse, the carry
measurement of exactly 2.00 is the estimator's own internal clip, so the true
figure is unknown and only known to be larger.

The sweep is cheap because of where the parameter lives. ``role_innovation_scale``
is consumed in ``predict_share_samples`` and enters no likelihood, so a single
posterior serves every candidate: fit once, then re-predict per cap. That turns
what would be one full pipeline fit per candidate into one per fold.

Selection is nested, for the same reason the exposure floor's is — searching
candidates on the seasons they are scored against would be selecting on the test
set:

    for each outer holdout H:
        fit on seasons < H-1, sweep caps against H-1, pick
        fit on seasons < H,   score H with the winner

    python scripts/select_innovation_cap.py --caps 0.25 0.35 0.50 0.75 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _walkforward_data import DEFAULT_CACHE, HOLDOUTS, load_frames

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.base import calibrate_innovation_scale
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

STREAMS = {"target": ("targets", "targets_per_team_game"),
           "carry": ("rush_att", "carries_per_team_game")}


def _apply_cap(pipeline, train_rows: pd.DataFrame, cap: float | None) -> dict[str, float]:
    """Set each allocator's scale as though it had been fitted under ``cap``."""
    applied: dict[str, float] = {}
    for stream, model in (
        ("target", pipeline.target_model),
        ("carry", pipeline.carry_model),
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
        applied[f"{stream}_measured"] = float(measured)
    return applied


def _score(pipeline, test: SeasonAverageData) -> dict[str, dict[str, float]]:
    prediction = pipeline.predict_samples(test, seed=42)
    rows = prediction.player_rows
    named = (
        pd.to_numeric(rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
                      errors="coerce").fillna(0).ne(1).to_numpy()
    )
    games = pd.to_numeric(rows["team_games"], errors="coerce").to_numpy(float)
    out: dict[str, dict[str, float]] = {}
    for stream, (column, attribute) in STREAMS.items():
        observed = (
            pd.to_numeric(rows[column], errors="coerce").to_numpy(float)[named]
            / games[named]
        )
        samples = getattr(prediction, attribute)[named]
        out[stream] = {
            "n": int(len(observed)),
            "mae": float(np.abs(observed - samples.mean(axis=1)).mean()),
            "crps": float(empirical_crps(observed, samples).mean()),
            "cov80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
            "cov95": float(interval_coverage(observed, samples, level=0.95)["coverage"]),
        }
    return out


def _coverage_penalty(scores: dict[str, dict[str, float]]) -> float:
    """Mean absolute distance from nominal, over both levels and both streams.

    The cap governs how wide the allocation is, so it is selected on width
    rather than on point accuracy. Carry's 80% intervals contain 88% of
    outcomes at the current value, which no MAE criterion would notice.
    """
    gaps = []
    for stream in STREAMS:
        gaps.append(abs(scores[stream]["cov80"] - 0.80))
        gaps.append(abs(scores[stream]["cov95"] - 0.95))
    return float(np.mean(gaps))


def _fit(player_rows: pd.DataFrame, team_rows: pd.DataFrame, seasons, sample_kwargs):
    data = SeasonAverageData(
        team_rows[team_rows.season.isin(seasons)].copy(),
        player_rows[player_rows.season.isin(seasons)].copy(),
    )
    return SeasonAverageVolumePipeline(postseason_role_features=True).fit(
        data, **sample_kwargs
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?", default="innovation_cap")
    parser.add_argument("--caps", nargs="+", type=float,
                        default=[0.25, 0.35, 0.50, 0.75, 1.0])
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--holdouts", nargs="+", type=int, default=list(HOLDOUTS))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/validation_runs"))
    args = parser.parse_args(argv)

    player_rows, team_rows = load_frames(args.cache_dir)
    sample_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    # ``None`` is uncapped, and is always a candidate so the sweep can report
    # that the bound is doing nothing useful.
    candidates: list[float | None] = [None, *sorted(set(args.caps))]

    report: dict[str, object] = {"candidates": [str(c) for c in candidates]}
    for holdout in args.holdouts:
        started = time.perf_counter()
        inner_season = holdout - 1
        inner_train = sorted(s for s in player_rows.season.unique() if s < inner_season)
        inner_pipeline = _fit(player_rows, team_rows, inner_train, sample_kwargs)
        inner_rows = player_rows[player_rows.season.isin(inner_train)]
        inner_test = SeasonAverageData(
            team_rows[team_rows.season == inner_season].copy(),
            player_rows[player_rows.season == inner_season].copy(),
        )

        inner: dict[str, dict] = {}
        for cap in candidates:
            applied = _apply_cap(inner_pipeline, inner_rows, cap)
            scores = _score(inner_pipeline, inner_test)
            inner[str(cap)] = {"scales": applied, "scores": scores,
                               "coverage_penalty": _coverage_penalty(scores)}
            print(f"[{holdout}] inner cap={cap}: penalty="
                  f"{inner[str(cap)]['coverage_penalty']:.4f} "
                  + " ".join(f"{s} cov80={scores[s]['cov80']:.3f}" for s in STREAMS),
                  flush=True)

        chosen_name = min(inner, key=lambda k: inner[k]["coverage_penalty"])
        chosen = None if chosen_name == "None" else float(chosen_name)
        print(f"[{holdout}] inner fold picks cap={chosen}", flush=True)

        outer_train = sorted(s for s in player_rows.season.unique() if s < holdout)
        outer_pipeline = _fit(player_rows, team_rows, outer_train, sample_kwargs)
        outer_rows = player_rows[player_rows.season.isin(outer_train)]
        outer_test = SeasonAverageData(
            team_rows[team_rows.season == holdout].copy(),
            player_rows[player_rows.season == holdout].copy(),
        )
        incumbent_scales = _apply_cap(outer_pipeline, outer_rows, 0.50)
        incumbent = _score(outer_pipeline, outer_test)
        selected_scales = _apply_cap(outer_pipeline, outer_rows, chosen)
        selected = _score(outer_pipeline, outer_test)

        report[str(holdout)] = {
            "inner_season": inner_season,
            "inner": inner,
            "chosen_cap": chosen_name,
            "outer_incumbent": incumbent,
            "outer_incumbent_scales": incumbent_scales,
            "outer_selected": selected,
            "outer_selected_scales": selected_scales,
            "seconds": time.perf_counter() - started,
        }
        print(f"[{holdout}] done in {report[str(holdout)]['seconds']:.0f}s", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.label}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
