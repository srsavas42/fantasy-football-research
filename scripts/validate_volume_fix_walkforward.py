"""Volume walk-forward over cached nflverse rows: holdouts 2022/2023/2024.

Used to validate the S0/S1/S2 fixes and the availability-coupled QB gate. The
results are in docs/volume-fix-validation-2026-08.md and the JSON it refers to
is in scripts/validation_runs/.

    python scripts/validate_volume_fix_walkforward.py coupled
    python scripts/validate_volume_fix_walkforward.py uncoupled --no-couple-gate

Both configurations read the same cached frames, so the comparison is
controlled. The first run builds and caches them.
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
from _walkforward_data import (
    add_common_arguments,
    frames_fingerprint,
    gate_override,
    load_frames,
)

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline


def distribution(observed, samples) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    mean = samples.mean(axis=1)
    return {
        "n": int(len(observed)),
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
    coupling = gate_override(args)
    sample_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}

    report: dict[str, object] = {
        "_frames": frames_fingerprint(player_rows, team_rows, args.cache_dir)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"wf_{args.label}.json"

    def flush() -> None:
        """Persist after every holdout, as the scoring walk-forward does.

        This script has lost a run to a container restart three times, each
        time discarding folds that had already finished, because the file was
        only written on completion. A partial file is readable -- every reader
        here keys on holdout -- and says plainly which folds it has.
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
        pipeline = SeasonAverageVolumePipeline(
            mean_preserving_innovation=(
                tuple(args.mean_preserving_layers)
                if args.mean_preserving_layers
                else args.mean_preserving_innovation
            ),
            calibrated_innovation=args.calibrated_innovation,
            innovation_cap=args.innovation_cap,
        )
        if args.postseason is not None:
            pipeline.postseason_role_features = args.postseason
        if args.market_adp is not None:
            pipeline.market_adp_features = args.market_adp
        if args.market_adp_interactions is not None:
            pipeline.market_adp_interactions = args.market_adp_interactions
        if args.cold_role_innovation is not None:
            pipeline.cold_role_innovation = args.cold_role_innovation
        if args.cold_role_scale_mode is not None:
            pipeline.cold_role_scale_mode = args.cold_role_scale_mode
        if args.snap_feature_prior is not None:
            pipeline.snap_model.feature_prior_scale = args.snap_feature_prior
        pipeline.team_model.models_play_transition = args.play_transition
        if coupling is not None:
            pipeline.workload_model.couple_gate_to_availability = coupling
        pipeline.fit(train, **sample_kwargs)
        prediction = pipeline.predict_samples(test, seed=42)

        rows = prediction.player_rows
        named = (
            pd.to_numeric(
                rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
                errors="coerce",
            )
            .fillna(0)
            .ne(1)
            .to_numpy()
        )
        quarterback = rows["position"].eq("QB").to_numpy()
        games = pd.to_numeric(rows["team_games"], errors="coerce").to_numpy(float)
        snaps_seen = (
            pd.to_numeric(rows["snap_counts_observed"], errors="coerce")
            .fillna(0)
            .gt(0)
            .to_numpy()
        )

        def per_game(column: str, mask: np.ndarray) -> np.ndarray:
            values = pd.to_numeric(rows[column], errors="coerce").to_numpy(float)
            return values[mask] / games[mask]

        fold: dict[str, object] = {
            "target": distribution(
                per_game("targets", named), prediction.targets_per_team_game[named]
            ),
            "carry": distribution(
                per_game("rush_att", named), prediction.carries_per_team_game[named]
            ),
            "pass_qb": distribution(
                per_game("pass_att", quarterback & named),
                prediction.pass_attempts_per_team_game[quarterback & named],
            ),
            "snap": distribution(
                pd.to_numeric(
                    rows.loc[snaps_seen & named, "snap_share"], errors="coerce"
                ).to_numpy(float),
                prediction.snap_share[snaps_seen & named],
            ),
            "availability": distribution(
                pd.to_numeric(
                    rows.loc[named, "observed_availability"], errors="coerce"
                ).to_numpy(float),
                prediction.availability[named],
            ),
            "qb_workload": distribution(
                pd.to_numeric(rows["observed_qb_workload_share"], errors="coerce")
                .fillna(0.0)
                .to_numpy(float)[quarterback & named],
                prediction.qb_workload_share[quarterback & named],
            ),
        }
        any_carry = (
            pd.to_numeric(rows.loc[named, "rush_att"], errors="coerce")
            .fillna(0)
            .gt(0)
            .to_numpy(float)
        )
        # Snap broken out by position. The pooled stream mixes quarterbacks with
        # skill positions, and a change can move them opposite ways -- widening
        # the snap model's feature prior buys running backs and receivers about
        # 6% of held-out MAE while costing backup quarterbacks. A pooled average
        # reports the net and hides that entirely.
        fold["snap_by_position"] = {}
        for position in ("QB", "RB", "WR", "TE"):
            at = snaps_seen & named & rows["position"].eq(position).to_numpy()
            if at.sum() < 5:
                continue
            fold["snap_by_position"][position] = distribution(
                pd.to_numeric(rows.loc[at, "snap_share"], errors="coerce").to_numpy(float),
                prediction.snap_share[at],
            )
        fold["carry_eligibility_brier"] = float(
            np.mean(
                (
                    any_carry
                    - prediction.carry_eligibility_probability[named].mean(axis=1)
                )
                ** 2
            )
        )
        fold["seconds"] = round(time.perf_counter() - started, 1)
        fold["diagnostics"] = {
            name: {
                "max_rhat": float(result["max_rhat"]),
                "min_ess": float(result["min_bulk_ess"]),
                "divergences": int(result["divergences"]),
            }
            for name, result in pipeline.diagnostics().items()
        }
        report[str(holdout)] = fold
        flush()
        print(f"[{args.label}] holdout {holdout} done in {fold['seconds']}s", flush=True)

    print(f"[{args.label}] wrote {path}")


if __name__ == "__main__":
    main()
