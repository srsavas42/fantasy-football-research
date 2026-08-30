"""Fit on every observed season and project the one that has not happened.

The shipping configuration and nothing else: preseason ADP in the availability
regression (promoted), and the market blend on top (promoted). The arms that
came back null or were rejected -- Vegas win totals in the team layer, ADP in
the quarterback room, the snap-based exposure target, position-varying
availability slopes -- stay off. A feature that failed its gate does not get
switched on because its point estimate looked friendly.

The output is a *blend*, not the raw model. On the players people draft the
pipeline alone loses to a plain rank curve, and mixing the two beats the board
by about 3% MAE and 2.5% CRPS. So a projected number here reflects the draft
board as well as the play-by-play history, which is the honest description of
what is being sold.
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
from ffmodel.models.market_blend import MarketBlend, RankCurve, slope_weight
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.simulation.scoring import fantasy_points

# From the walk-forward's own slope rule, whose weights climbed 0.203, 0.239,
# 0.316 as folds accumulated against a pooled disagreement slope of 0.409. The
# most recent fold's weight is used rather than refitting on 2026, which has no
# outcomes to fit against.
BLEND_WEIGHT = 0.316


def observed_points(rows: pd.DataFrame, scoring: str) -> np.ndarray:
    return fantasy_points(observed_scoring_rows(rows), scoring).to_numpy(float)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--weight", type=float, default=BLEND_WEIGHT)
    parser.add_argument("--out-dir", type=Path, default=Path("projections"))
    args = parser.parse_args(argv)

    player_rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    team_rows = pd.read_pickle(args.cache_dir / "team_rows.pkl")
    train = SeasonAverageData(
        team_rows[team_rows.season < args.season].copy(),
        player_rows[player_rows.season < args.season].copy(),
    )
    test = SeasonAverageData(
        team_rows[team_rows.season == args.season].copy(),
        player_rows[player_rows.season == args.season].copy(),
    )
    if test.player_rows.empty:
        raise SystemExit(f"no rows for {args.season} in {args.cache_dir}")

    started = time.perf_counter()
    pipeline = SeasonAverageScoringPipeline()
    sample_kwargs = {"draws": args.draws, "tune": args.draws, "chains": args.chains}
    pipeline.fit(
        train, volume_sample_kwargs=sample_kwargs, efficiency_sample_kwargs=sample_kwargs
    )
    print(f"fitted in {time.perf_counter() - started:.0f}s", flush=True)
    prediction = pipeline.predict_samples(test, seed=args.season)

    rows = prediction.player_rows.reset_index(drop=True)
    model = np.asarray(prediction.fantasy_points[args.scoring], dtype=float)

    # The rank curve, fitted on every observed season's drafted players.
    history = player_rows[player_rows.season < args.season].reset_index(drop=True)
    history = history[
        pd.to_numeric(history.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    ].reset_index(drop=True)
    points = np.full(len(history), np.nan)
    for _, block in history.groupby("season"):
        points[block.index] = observed_points(block.reset_index(drop=True), args.scoring)
    usable = np.isfinite(points)
    curve = RankCurve().fit(history[usable].reset_index(drop=True), points[usable])
    blend = MarketBlend(weight=float(args.weight), curve=curve)
    blended = blend.predict_samples(rows, model, seed=args.season)

    named = (
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    ).to_numpy()

    def draw_mean(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float).mean(axis=1)

    volume = prediction.volume
    efficiency = prediction.efficiency
    out = pd.DataFrame(
        {
            "player_name": rows["player_name"],
            "team": rows["team"],
            "position": rows["position"],
            "adp_rank": pd.to_numeric(rows.get("adp_rank"), errors="coerce"),
            "adp_drafted": pd.to_numeric(rows.get("adp_drafted"), errors="coerce"),
            "projection": blended.mean(axis=1),
            "p10": np.quantile(blended, 0.10, axis=1),
            "p50": np.quantile(blended, 0.50, axis=1),
            "p90": np.quantile(blended, 0.90, axis=1),
            "model_only": model.mean(axis=1),
            "projected_games": draw_mean(volume.games_active),
            # Snap share and role.
            "snap_share": draw_mean(volume.snap_share),
            "pass_attempt_share": draw_mean(volume.pass_attempt_share),
            "target_share": draw_mean(volume.target_share),
            "carry_share": draw_mean(volume.carry_share),
            # Volume (season totals).
            "pass_attempts": draw_mean(volume.pass_attempts),
            "targets": draw_mean(volume.targets),
            "carries": draw_mean(volume.carries),
            # Efficiency (per-opportunity rates).
            "completion_rate": draw_mean(efficiency.rates["pass_completion_rate"]),
            "yards_per_attempt": draw_mean(efficiency.rates["pass_yards_per_attempt"]),
            "pass_td_rate": draw_mean(efficiency.rates["pass_td_rate"]),
            "pass_int_rate": draw_mean(efficiency.rates["pass_int_rate"]),
            "catch_rate": draw_mean(efficiency.rates["rec_catch_rate"]),
            "yards_per_target": draw_mean(efficiency.rates["rec_yards_per_target"]),
            "rec_td_rate": draw_mean(efficiency.rates["rec_td_rate"]),
            "yards_per_carry": draw_mean(efficiency.rates["rush_yards_per_carry"]),
            "rush_td_rate": draw_mean(efficiency.rates["rush_td_rate"]),
            "fumble_lost_rate": draw_mean(efficiency.rates["fumble_lost_rate"]),
            # Projected stat lines (season totals, from the coherent draws).
            "pass_cmp": draw_mean(prediction.pass_cmp),
            "pass_yds": draw_mean(prediction.pass_yds),
            "pass_td": draw_mean(prediction.pass_td),
            "pass_int": draw_mean(prediction.pass_int),
            "receptions": draw_mean(prediction.receptions),
            "rec_yds": draw_mean(prediction.rec_yds),
            "rec_td": draw_mean(prediction.rec_td),
            "rush_yds": draw_mean(prediction.rush_yds),
            "rush_td": draw_mean(prediction.rush_td),
            "fumbles_lost": draw_mean(prediction.fumbles_lost),
        }
    )[named].reset_index(drop=True)
    out = out.sort_values("projection", ascending=False).reset_index(drop=True)
    out.insert(0, "overall", np.arange(1, len(out) + 1))
    out["position_rank"] = out.groupby("position")["projection"].rank(
        ascending=False, method="first"
    ).astype(int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"{args.season}_{args.scoring}"
    out.to_csv(stem.with_suffix(".csv"), index=False)
    np.savez_compressed(
        stem.with_suffix(".samples.npz"), samples=blended[named].astype(np.float32)
    )
    stem.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "season": args.season,
                "scoring": args.scoring,
                "rows": int(len(out)),
                "blend_weight": float(args.weight),
                "seconds": round(time.perf_counter() - started, 1),
                "config": {
                    "market_adp_availability": bool(
                        pipeline.volume_model.market_adp_availability_features
                    ),
                    "market_adp_features": bool(pipeline.volume_model.market_adp_features),
                    "market_adp_qb": bool(pipeline.volume_model.market_adp_qb_features),
                    "market_win_total": bool(
                        pipeline.volume_model.market_win_total_features
                    ),
                    "availability_target": str(pipeline.volume_model.availability_target),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {stem}.csv  ({len(out)} players)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
