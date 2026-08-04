"""Total-season scoring walk-forward, self-contained (no cached checkpoints).

``validate_season_scoring_posteriors.py`` loads pre-fitted posteriors from
``.cache/season-average-validation/...``. A fresh checkout has none, so the gate
cannot be re-run there at all. This fits both layers per holdout from data
instead, which makes the comparison reproducible but means its absolute levels
are not comparable to the numbers in docs/season-scoring-v1-validation.md — only
paired comparisons within this harness are.

    python scripts/validate_scoring_walkforward.py coupled
    python scripts/validate_scoring_walkforward.py uncoupled --no-couple-gate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _walkforward_data import add_common_arguments, gate_override, load_frames

from ffmodel.evaluation.efficiency_posterior import score_fantasy_points_posterior
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline

SCORING_FORMATS = ("standard", "half_ppr", "ppr")


def main(argv=None) -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("scripts/validation_runs")
    )
    args = parser.parse_args(argv)

    player_rows, team_rows = load_frames(args.cache_dir)
    coupling = gate_override(args)
    sample_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}

    report: dict[str, object] = {}
    for holdout in args.holdouts:
        started = time.perf_counter()
        train = SeasonAverageData(
            team_rows[team_rows.season < holdout].copy(),
            player_rows[player_rows.season < holdout].copy(),
        )
        test = SeasonAverageData(
            team_rows[team_rows.season == holdout].copy(),
            player_rows[player_rows.season == holdout].copy(),
        )
        pipeline = SeasonAverageScoringPipeline(
            efficiency_exposure_floor=args.efficiency_exposure_floor,
        )
        if args.volume_feature_estimator is not None:
            pipeline.volume_feature_estimator = args.volume_feature_estimator
            # These cross-fits only contribute a posterior mean to a covariate,
            # so they do not need the budget of the models being validated.
            pipeline.volume_feature_sample_kwargs = {
                "draws": args.volume_feature_draws,
                "tune": args.volume_feature_draws,
                "chains": 2,
            }
        pipeline.volume_model.postseason_role_features = args.postseason
        pipeline.volume_model.mean_preserving_innovation = (
            tuple(args.mean_preserving_layers)
            if args.mean_preserving_layers
            else args.mean_preserving_innovation
        )
        pipeline.volume_model.calibrated_innovation = args.calibrated_innovation
        pipeline.volume_model.cold_role_innovation = args.cold_role_innovation
        pipeline.volume_model.innovation_cap = args.innovation_cap
        pipeline.volume_model.team_model.models_play_transition = args.play_transition
        if coupling is not None:
            pipeline.volume_model.workload_model.couple_gate_to_availability = coupling
        pipeline.fit(
            train,
            volume_sample_kwargs=sample_kwargs,
            efficiency_sample_kwargs=sample_kwargs,
        )
        prediction = pipeline.predict_samples(test, seed=42)

        fold: dict[str, object] = {
            scoring: score_fantasy_points_posterior(prediction, scoring=scoring)
            for scoring in SCORING_FORMATS
        }
        fold["seconds"] = round(time.perf_counter() - started, 1)
        report[str(holdout)] = fold
        print(
            f"[{args.label}] holdout {holdout} done in {fold['seconds']}s "
            f"ppr cov95={fold['ppr']['coverage_95']:.3f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"scoring_{args.label}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[{args.label}] wrote {path}")


if __name__ == "__main__":
    main()
