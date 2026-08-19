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
from _walkforward_data import (
    add_common_arguments,
    frames_fingerprint,
    gate_override,
    load_frames,
)

from ffmodel.evaluation.efficiency_posterior import score_fantasy_points_posterior
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline

SCORING_FORMATS = ("standard", "half_ppr", "ppr")


def _diagnostics(pipeline) -> dict[str, dict[str, object]]:
    """Sampler health for the volume components and every efficiency response."""
    out: dict[str, dict[str, object]] = {}
    for name, result in pipeline.volume_model.diagnostics().items():
        out[f"volume/{name}"] = {
            "max_rhat": float(result["max_rhat"]),
            "min_ess": float(result["min_bulk_ess"]),
            "divergences": int(result["divergences"]),
        }
    for name, result in pipeline.efficiency_model.diagnostics().items():
        out[f"efficiency/{name}"] = {
            "max_rhat": float(result["max_rhat"]),
            "min_ess": float(result["min_bulk_ess"]),
            "divergences": int(result["divergences"]),
        }
    return out


def _fitted_features(pipeline) -> dict[str, list[str]]:
    """What each volume submodel actually put in its design matrix.

    A feature flag sets a name on a list; the design matrix drops names it
    cannot find in the frame. Between those two steps an arm can be enabled,
    report no error, and fit exactly the baseline -- which has happened twice on
    this branch. The fitted names belong in the record so a null result can be
    read as "the feature did nothing" rather than "the feature was not there".
    """
    volume = pipeline.volume_model
    names = (
        "availability_model",
        "workload_model",
        "snap_model",
        "carry_eligibility_model",
        "target_model",
        "carry_model",
    )
    out: dict[str, list[str]] = {}
    for name in names:
        model = getattr(volume, name, None)
        if model is not None and getattr(model, "idata", None) is not None:
            out[name] = sorted(getattr(model, "feature_names", []))
    return out


def main(argv=None) -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("scripts/validation_runs")
    )
    args = parser.parse_args(argv)

    player_rows, team_rows = load_frames(args.cache_dir)
    coupling = gate_override(args)
    sample_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}

    report: dict[str, object] = {
        "_frames": frames_fingerprint(player_rows, team_rows, args.cache_dir)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"scoring_{args.label}.json"

    def flush() -> None:
        """Persist after every holdout, not once at the end.

        A run is hours of sampling and the container it runs in is ephemeral.
        Writing only on completion means a restart in the third fold throws away
        the first two, which has now happened twice here. A partial file is
        readable -- the reader keys on holdout -- and says plainly which folds
        finished.
        """
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    flush()
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
            teammate_quality_features=args.teammate_quality,
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
        if args.postseason is not None:
            pipeline.volume_model.postseason_role_features = args.postseason
        if args.market_adp is not None:
            pipeline.volume_model.market_adp_features = args.market_adp
        pipeline.volume_model.mean_preserving_innovation = (
            tuple(args.mean_preserving_layers)
            if args.mean_preserving_layers
            else args.mean_preserving_innovation
        )
        pipeline.volume_model.calibrated_innovation = args.calibrated_innovation
        if args.cold_role_innovation is not None:
            pipeline.volume_model.cold_role_innovation = args.cold_role_innovation
        if args.cold_role_scale_mode is not None:
            pipeline.volume_model.cold_role_scale_mode = args.cold_role_scale_mode
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
        # Both layers, because a scoring run that samples badly reports a CRPS
        # like any other and nothing in the output says otherwise. The volume
        # walk-forward has recorded this since it was written; this one never
        # did, and a cap selection built on it hid 304 divergences in an
        # efficiency response behind a clean-looking table.
        fold["diagnostics"] = _diagnostics(pipeline)
        fold["volume_features"] = _fitted_features(pipeline)
        report[str(holdout)] = fold
        flush()
        print(
            f"[{args.label}] holdout {holdout} done in {fold['seconds']}s "
            f"ppr cov95={fold['ppr']['coverage_95']:.3f}",
            flush=True,
        )

    print(f"[{args.label}] wrote {path}")


if __name__ == "__main__":
    main()
