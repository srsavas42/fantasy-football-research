"""Save per-row predictive samples for each holdout, once, so analysis is cheap.

Every question left open about the ADP gap needs the pipeline's own per-row
predictions, and the walk-forward keeps none: it scores a fold, writes summary
metrics, and throws the samples away. That has meant refitting for each new
question, which at roughly a quarter-hour a fold has been the binding constraint
on how many questions get asked.

This fits the shipping configuration once per holdout and saves the draws. What
comes out supports blending, error decomposition, per-position attribution and
anything else, at numpy speed.

Two things are deliberate.

**The shipping configuration, whatever it currently is.** No flags. If a default
moves, re-running this re-measures everything downstream against the model that
actually ships, rather than against a configuration someone pinned months ago.
The recorded metadata names the settings that were live, so a stale export
announces itself.

**Every rostered row, not just the drafted pool.** The drafted pool is the
population of interest, but restricting here would foreclose questions about the
rest, and the storage difference is a few megabytes.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.efficiency_posterior import observed_scoring_rows
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.simulation.scoring import fantasy_points

CARRIED = (
    "season",
    # The frame's own key. Joining these rows to weekly data on name and team
    # instead loses every player who changed teams mid-season and every name
    # nflverse spells two ways, so carry the identifier even though nothing in
    # the model reads it.
    "player_id",
    "team",
    "player_name",
    "position",
    "adp_rank",
    "adp_drafted",
    "is_replacement_player",
    "games",
    "team_games",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp2")
    )
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--market-adp", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path(".cache/holdout-predictions"))
    parser.add_argument("--label", default="shipping")
    args = parser.parse_args(argv)

    player_rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    team_rows = pd.read_pickle(args.cache_dir / "team_rows.pkl")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_kwargs = {"draws": args.draws, "tune": args.draws, "chains": args.chains}

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
        pipeline = SeasonAverageScoringPipeline()
        if args.market_adp:
            pipeline.market_adp_features = True
        pipeline.fit(
            train,
            volume_sample_kwargs=sample_kwargs,
            efficiency_sample_kwargs=sample_kwargs,
        )
        prediction = pipeline.predict_samples(test, seed=42)

        rows = prediction.player_rows.reset_index(drop=True)
        observed = fantasy_points(
            observed_scoring_rows(rows), args.scoring
        ).to_numpy(float)
        samples = np.asarray(prediction.fantasy_points[args.scoring], dtype=float)
        # Season points are the product of a per-game rate and an exposure, and
        # the two answer different questions. Without the exposure draws there
        # is no way to ask what the model thinks a player scores *per game*,
        # which is the quantity a drafter comparing two backs actually wants.
        # Paired draw-for-draw with the points, so a ratio taken per draw keeps
        # whatever correlation the simulation put between them.
        games = np.asarray(prediction.volume.games_active, dtype=float)

        frame = pd.DataFrame(
            {name: rows[name] for name in CARRIED if name in rows}
        )
        frame["observed"] = observed
        path = args.out_dir / f"{args.label}_{holdout}"
        frame.to_parquet(path.with_suffix(".rows.parquet"))
        # float32 halves the file and is far finer than the sampling noise the
        # draws themselves carry.
        np.savez_compressed(
            path.with_suffix(".samples.npz"),
            samples=samples.astype(np.float32),
            games_active=games.astype(np.float32),
        )
        volume = pipeline.volume_model
        meta = {
            "holdout": holdout,
            "scoring": args.scoring,
            "rows": int(len(frame)),
            "draws": int(samples.shape[1]),
            "seconds": round(time.perf_counter() - started, 1),
            "cache_dir": str(args.cache_dir),
            "config": {
                "snap_feature_prior": float(volume.snap_model.feature_prior_scale),
                "market_adp_features": bool(volume.market_adp_features),
                "market_adp_interactions": bool(volume.market_adp_interactions),
                "postseason_role_features": bool(volume.postseason_role_features),
                "cold_role_innovation": bool(volume.cold_role_innovation),
                "cold_role_scale_mode": str(volume.cold_role_scale_mode),
                "calibrated_innovation": bool(volume.calibrated_innovation),
            },
        }
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"[{args.label}] {holdout}: {len(frame)} rows x {samples.shape[1]} draws "
            f"in {meta['seconds']}s -> {path}.*",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
