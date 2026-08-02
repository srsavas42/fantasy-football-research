"""Fit the promoted pipeline on every available season and save it for serving.

Holdout posteriors are validation evidence. They are fitted on a truncated
history on purpose, and reusing one to publish projections would serve a model
that has never seen the most recent season — the one that matters most for a
lagged-feature contract. This produces the artifact that is allowed to serve.

    python scripts/fit_production.py --output artifacts/season-average-2025

What it does beyond calling ``fit``:

* records sampler health for every component, not the ones someone remembered
  to check, and **fails** if any component diverges or misses the R-hat and ESS
  bounds the acceptance gate uses;
* round-trips the saved artifact and re-predicts, because several promoted
  settings live in metadata rather than in the posterior, and a flag that fails
  to serialize produces a model that differs from the one that was validated
  while raising nothing;
* writes a manifest naming the seasons, the configuration and the diagnostics,
  so a published projection can be traced to the fit that produced it.

A production fit is deliberately more expensive than a validation fold. The
default budget is 2000 draws, which is also the budget at which the team model's
R-hat stops depending on its seed.
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
from _walkforward_data import DEFAULT_CACHE, load_frames

from ffmodel.evaluation.acceptance import MIN_BULK_ESS, RHAT_ACCEPT
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.base import sampling_quality
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline


def _component_diagnostics(pipeline) -> dict[str, dict[str, float]]:
    """Sampler health for every fitted component in both layers."""
    out: dict[str, dict[str, float]] = {}
    volume = pipeline.volume_model
    components = {
        "team": volume.team_model,
        "availability": volume.availability_model,
        "snap": volume.snap_model,
        "workload": volume.workload_model,
        "qb_propensity": volume.qb_propensity_model,
        "target_role": volume.target_role_model,
        "carry_eligibility": volume.carry_eligibility_model,
        "target": volume.target_model,
        "carry": volume.carry_model,
    }
    for target, model in pipeline.efficiency_model.models.items():
        components[f"efficiency::{target}"] = model

    for name, model in components.items():
        idata = getattr(model, "idata", None)
        if idata is None:
            continue
        try:
            quality = sampling_quality(idata)
        except Exception as error:  # a component with no scalar parameters
            out[name] = {"error": str(error)}
            continue
        out[name] = {
            "max_rhat": float(quality["max_rhat"]),
            "min_bulk_ess": float(quality["min_bulk_ess"]),
            "divergences": int(quality["divergences"]),
        }
    return out


def _unhealthy(diagnostics: dict[str, dict[str, float]]) -> list[str]:
    problems = []
    for name, values in sorted(diagnostics.items()):
        if "error" in values:
            problems.append(f"{name}: diagnostics unavailable ({values['error']})")
            continue
        if values["divergences"] > 0:
            problems.append(f"{name}: {values['divergences']} divergences")
        if values["max_rhat"] >= RHAT_ACCEPT:
            problems.append(f"{name}: R-hat {values['max_rhat']:.4f}")
        if values["min_bulk_ess"] < MIN_BULK_ESS:
            problems.append(f"{name}: bulk ESS {values['min_bulk_ess']:.0f}")
    return problems


def _round_trip(directory: Path, data: SeasonAverageData, original) -> dict[str, object]:
    """Reload the artifact and confirm it predicts what the fitted object does.

    Several promoted settings — the availability-coupled gate, the postseason
    role features, the mean-preserving innovation, the efficiency exposure floor
    — live in metadata rather than in a posterior. A flag that fails to
    serialize yields a served model quietly different from the validated one, so
    this is checked rather than assumed.
    """
    reloaded = SeasonAverageScoringPipeline.load(directory)
    before = original.predict_samples(data, seed=11).fantasy_points["ppr"]
    after = reloaded.predict_samples(data, seed=11).fantasy_points["ppr"]
    difference = float(np.abs(np.asarray(before) - np.asarray(after)).max())
    return {
        "max_abs_ppr_difference": difference,
        "identical": bool(difference < 1e-9),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--volume-feature-estimator", choices=("ridge", "pipeline"), default="ridge"
    )
    parser.add_argument("--efficiency-exposure-floor", type=int, default=None)
    parser.add_argument(
        "--allow-unhealthy",
        action="store_true",
        help="save the artifact even if a component fails its sampler bounds; "
             "the manifest still records the failure",
    )
    args = parser.parse_args(argv)

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            f"{args.output} is not empty. A production artifact is never "
            "overwritten in place — publish to a new directory so a served "
            "projection can always be traced back to its fit."
        )

    player_rows, team_rows = load_frames(args.cache_dir)
    seasons = sorted(int(s) for s in pd.unique(player_rows["season"]))
    data = SeasonAverageData(team_rows.copy(), player_rows.copy())
    print(f"fitting on {len(seasons)} seasons: {seasons[0]}-{seasons[-1]}", flush=True)

    sample_kwargs = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "seed": args.seed,
    }
    pipeline = SeasonAverageScoringPipeline(
        volume_feature_estimator=args.volume_feature_estimator,
        efficiency_exposure_floor=args.efficiency_exposure_floor,
    )

    started = time.perf_counter()
    pipeline.fit(
        data,
        volume_sample_kwargs=sample_kwargs,
        efficiency_sample_kwargs=sample_kwargs,
    )
    elapsed = time.perf_counter() - started
    print(f"fit in {elapsed:.0f}s", flush=True)

    diagnostics = _component_diagnostics(pipeline)
    problems = _unhealthy(diagnostics)
    for problem in problems:
        print(f"  UNHEALTHY {problem}", flush=True)
    if problems and not args.allow_unhealthy:
        raise SystemExit(
            f"{len(problems)} component(s) failed their sampler bounds; "
            "nothing was saved. Re-run with a longer chain, or with "
            "--allow-unhealthy if the failure is understood and recorded."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    pipeline.save(args.output)
    round_trip = _round_trip(args.output, data, pipeline)
    if not round_trip["identical"]:
        print(
            f"  WARNING reloaded artifact differs by "
            f"{round_trip['max_abs_ppr_difference']:.6g} PPR points",
            flush=True,
        )

    manifest = {
        "seasons": seasons,
        "fit_seconds": elapsed,
        "sample_kwargs": sample_kwargs,
        "configuration": {
            "volume_feature_estimator": args.volume_feature_estimator,
            "efficiency_exposure_floor": args.efficiency_exposure_floor,
            "postseason_role_features": pipeline.volume_model.postseason_role_features,
            "mean_preserving_innovation": (
                pipeline.volume_model.mean_preserving_innovation
            ),
            "couple_gate_to_availability": (
                pipeline.volume_model.workload_model.couple_gate_to_availability
            ),
        },
        "diagnostics": diagnostics,
        "unhealthy": problems,
        "round_trip": round_trip,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.output}/manifest.json")
    return 0 if round_trip["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
