"""Is the target allocator's cold-start lift a defect or a correct adjustment?

The 2026 projection puts a rookie's target share next to an established WR1's
on a third of the snaps. That is not the role prior: the softmax score is
``log(role_prior) + log(snap_share) + X.beta + innovation``, so exposure is
already priced, and the deterministic prior allocation ranks DeVonta Smith over
Makai Lemon 0.260 to 0.100. The model returns 0.177 to 0.162.

The suspect is ``cold_role_innovation``. It gives rows with no prior role their
own, much wider innovation scale -- promoted for interval coverage, which it
genuinely fixed. But the noise is added on the input side of a softmax, and a
softmax is not linear: a wider scale raises a row's draw-*average* share, not
only its spread. At the cold scale this pipeline fits, ``exp(sigma^2/2)`` is
worth roughly 3x, and the observed cold-vs-warm lift over the deterministic
prior allocation is 2.65x. The arithmetic matches the artifact.

Whether that is wrong is an empirical question this script answers, and the
answer is not obvious in either direction: rookies really do break out, and if
the lift is buying real signal then removing it costs accuracy. So measure
*bias* on held-out seasons, split by cold-start status -- mean projection
against mean observation, where a defect shows up as cold rows projected high.

``mean_preserving_innovation`` is the instrument. It solves for a per-player
offset that restores the draw-average to the noiseless allocation, so it
removes exactly the softmax mean shift and nothing else. It is read only in
``_role_share_prediction`` and never during fit, so one posterior serves both
arms and predicting twice with the same seed makes the comparison exactly
paired: identical innovation draws, one difference.

The flag has been walk-forwarded twice before and rejected both times
(docs/role-innovation-2026-08.md). Neither run broke results out by cold-start
status, which is the population this is about.

    python scripts/measure_cold_start_bias.py cold --holdouts 2023 2024 2025
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
from _walkforward_data import add_common_arguments, frames_fingerprint, load_frames

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline


def summarise(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    """Bias first: this is a question about location, not spread."""
    observed = np.asarray(observed, dtype=float)
    mean = samples.mean(axis=1)
    total = float(observed.sum())
    return {
        "n": int(len(observed)),
        "observed_mean": float(observed.mean()),
        "projected_mean": float(mean.mean()),
        # Relative bias on the pooled total rather than the mean of per-row
        # ratios: the denominator is tiny for most cold rows, and a mean of
        # ratios there reports the handful of near-zero observations rather
        # than the population.
        "bias_pct": float((mean.sum() - total) / total * 100.0) if total > 0 else float("nan"),
        "mae": float(np.abs(observed - mean).mean()),
        "crps": float(empirical_crps(observed, samples).mean()),
        "cov80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
        "cov95": float(interval_coverage(observed, samples, level=0.95)["coverage"]),
    }


def main(argv=None) -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("scripts/validation_runs")
    )
    args = parser.parse_args(argv)

    player_rows, team_rows = load_frames(args.cache_dir)
    sample_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    report: dict[str, object] = {
        "_frames": frames_fingerprint(player_rows, team_rows, args.cache_dir)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"coldbias_{args.label}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

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
        pipeline = SeasonAverageVolumePipeline()
        pipeline.fit(train, **sample_kwargs)

        fold: dict[str, object] = {}
        for arm in ("baseline", "mean_preserving"):
            # Set on the share models directly rather than through the pipeline
            # flag: ``_enable_mean_preserving_innovation`` runs inside ``fit``,
            # and refitting would break the pairing this measurement depends on.
            for model in (pipeline.target_model, pipeline.carry_model):
                model.mean_preserving_innovation = arm == "mean_preserving"
            prediction = pipeline.predict_samples(test, seed=42)
            rows = prediction.player_rows
            named = (
                pd.to_numeric(
                    rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
                    errors="coerce",
                ).fillna(0).ne(1).to_numpy()
            )
            cold = (
                pd.to_numeric(
                    rows.get("cold_start", pd.Series(0, index=rows.index)),
                    errors="coerce",
                ).fillna(0).eq(1).to_numpy()
            )
            games = pd.to_numeric(rows["team_games"], errors="coerce").to_numpy(float)
            skill = (~rows["position"].eq("QB")).to_numpy()

            arm_result: dict[str, object] = {}
            for stream, column, samples in (
                ("target", "targets", prediction.targets_per_team_game),
                ("carry", "rush_att", prediction.carries_per_team_game),
            ):
                observed = (
                    pd.to_numeric(rows[column], errors="coerce").to_numpy(float) / games
                )
                base = named & skill if stream == "target" else named
                arm_result[stream] = {
                    "all": summarise(observed[base], samples[base]),
                    "cold": summarise(observed[base & cold], samples[base & cold]),
                    "warm": summarise(observed[base & ~cold], samples[base & ~cold]),
                }
                # Rookies are the population the projection artifact showed, and
                # they are not all of ``cold_start``: a journeyman returning from
                # a year out of the league is cold too, and priced differently.
                rookie = base & cold & pd.to_numeric(
                    rows.get("experience", pd.Series(np.nan, index=rows.index)),
                    errors="coerce",
                ).fillna(99).le(0).to_numpy()
                if rookie.sum() >= 5:
                    arm_result[stream]["rookie"] = summarise(
                        observed[rookie], samples[rookie]
                    )
            fold[arm] = arm_result

        fold["fit_seconds"] = round(time.perf_counter() - started, 1)
        report[str(holdout)] = fold
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"holdout {holdout} done in {fold['fit_seconds']}s -> {path}")


if __name__ == "__main__":
    main()
