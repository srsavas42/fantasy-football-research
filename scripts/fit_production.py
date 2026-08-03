"""Fit the promoted pipeline on every available season and save it for serving.

Holdout posteriors are validation evidence. They are fitted on a truncated
history on purpose, and reusing one to publish projections would serve a model
that has never seen the most recent season — the one that matters most for a
lagged-feature contract. This produces the artifact that is allowed to serve.

    python scripts/fit_production.py --output artifacts/season-average-2025

What it does beyond calling ``fit``:

* records sampler health for every component, not the ones someone remembered
  to check, and refuses to publish if any component diverges or misses the
  R-hat and ESS bounds the acceptance gate uses — the posteriors are kept under
  a ``.rejected`` name so half an hour of sampling is not lost, but nothing
  loads them by accident;
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
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline


def _component_diagnostics(pipeline) -> dict[str, dict[str, float]]:
    """Sampler health for every fitted component in both layers.

    Both pipelines already expose a ``diagnostics()`` that names the global and
    variance terms worth gating on, which is what ``sampling_quality`` is built
    for — its docstring says so. Summarizing every variable instead is not just
    noisier: at a production budget of 2000 draws it fails outright, because
    ArviZ tries to build a summary row for each entry of the team model's
    per-team-season terms and overflows. Ask the pipelines rather than
    re-deriving the list here and getting it wrong.
    """
    out: dict[str, dict[str, float]] = {}
    layers = {
        "": pipeline.volume_model.diagnostics(),
        "efficiency::": pipeline.efficiency_model.diagnostics(),
    }
    for prefix, results in layers.items():
        for name, quality in results.items():
            if not isinstance(quality, dict) or "max_rhat" not in quality:
                continue
            out[f"{prefix}{name}"] = {
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

    Checked on the most recent season only. A season-scoring prediction holds
    roughly twenty arrays of (rows x draws); over ten seasons at a 2000-draw,
    four-chain budget that is about half a gigabyte each, and holding two
    pipelines' worth at once exhausted a 15 GB machine. One season is a
    twentieth of that, it exercises every serialized path the same way, and it
    is the season a projection is actually served for.
    """
    latest = max(int(s) for s in pd.unique(data.player_rows["season"]))
    slice_ = SeasonAverageData(
        data.team_rows[data.team_rows["season"] == latest].copy(),
        data.player_rows[data.player_rows["season"] == latest].copy(),
    )
    before = np.asarray(original.predict_samples(slice_, seed=11).fantasy_points["ppr"])
    reloaded = SeasonAverageScoringPipeline.load(directory)
    after = np.asarray(reloaded.predict_samples(slice_, seed=11).fantasy_points["ppr"])
    difference = float(np.abs(before - after).max())
    return {
        "season": latest,
        "rows": int(before.shape[0]),
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

    # An unhealthy fit must not land where something could serve it, but half an
    # hour of sampling should not be thrown away either — the first run of this
    # script lost exactly that to a bug in the diagnostics rather than in the
    # model. Quarantine instead: the posteriors are kept, under a name nothing
    # loads by accident, with a manifest saying why.
    destination = args.output
    rejected = bool(problems) and not args.allow_unhealthy
    if rejected:
        destination = args.output.with_name(args.output.name + ".rejected")
        print(
            f"  saving to {destination} instead; this artifact must not serve",
            flush=True,
        )

    destination.mkdir(parents=True, exist_ok=True)
    pipeline.save(destination)
    args.output = destination
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
            # Read off the efficiency pipeline, not off the command line. The
            # scoring pipeline's own field is an *override* and stays None when
            # the promoted default is being used, so reporting it would have
            # this manifest say "null" about a fit that used 5.
            "efficiency_exposure_floor": pipeline.efficiency_model.exposure_floor,
            "efficiency_exposure_floor_override": args.efficiency_exposure_floor,
            "postseason_role_features": pipeline.volume_model.postseason_role_features,
            "mean_preserving_innovation": (
                pipeline.volume_model.mean_preserving_innovation
            ),
            "calibrated_innovation": pipeline.volume_model.calibrated_innovation,
            # The scale the calibration actually solved for. It is the whole
            # point of the flag, it varies with the room sizes in the training
            # data, and a manifest that records only "calibration was on" cannot
            # tell two fits apart.
            "role_innovation_scale": {
                "workload": pipeline.volume_model.workload_model.role_innovation_scale,
                "target": pipeline.volume_model.target_model.role_innovation_scale,
                "carry": pipeline.volume_model.carry_model.role_innovation_scale,
            },
            "couple_gate_to_availability": (
                pipeline.volume_model.workload_model.couple_gate_to_availability
            ),
        },
        "diagnostics": diagnostics,
        "unhealthy": problems,
        "rejected": rejected,
        "servable": bool(not rejected and round_trip["identical"]),
        "round_trip": round_trip,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.output}/manifest.json")
    if rejected:
        print(
            f"{len(problems)} component(s) failed their sampler bounds. The fit is "
            f"kept at {args.output} for inspection and must not be served.",
            flush=True,
        )
    return 0 if manifest["servable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
