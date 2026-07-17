"""Walk-forward validation for season-wide average volume projections.

The latest requested season is held out in full. Predictors are lagged from the
prior season. nflverse runs can use week-1 point-in-time rosters and depth
charts; legacy runs explicitly report their inferred support. Reported player
volume is per team game, while a separate Beta-Binomial layer projects active
games and per-active-game volume.

Example:

    python scripts/validate_season_average.py --seasons 2014 2015 2016 2017 \
        2018 2019 2020 --draws 500 --tune 500 --chains 4 --nuts-sampler nutpie
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.evaluation.season_average import (
    RidgeRosterBaseline,
    XGBoostRosterBaseline,
    persistence_shares,
    xgboost_available,
)
from ffmodel.features.season_average import SeasonAverageData, build_season_average_data
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline


def _point_metrics(label, observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    print(
        f"{label}: MAE={np.abs(observed - predicted).mean():.3f} "
        f"RMSE={np.sqrt(np.mean((observed - predicted) ** 2)):.3f}"
    )


def _distribution_metrics(label, observed, samples):
    observed = np.asarray(observed, dtype=float)
    mean = samples.mean(axis=1)
    coverage = interval_coverage(observed, samples, level=0.8)["coverage"]
    print(
        f"{label}: MAE={np.abs(observed - mean).mean():.3f} "
        f"CRPS={empirical_crps(observed, samples).mean():.3f} "
        f"80% coverage={coverage:.3f}"
    )


def _workload_metrics(rows, workload_share):
    quarterbacks = rows[rows["position"].eq("QB")].copy()
    quarterbacks["predicted"] = workload_share[quarterbacks.index].mean(axis=1)
    quarterbacks["observed"] = pd.to_numeric(
        quarterbacks["observed_qb_workload_share"], errors="coerce"
    ).fillna(0.0)
    _point_metrics(
        "QB offensive-snap workload share",
        quarterbacks["observed"],
        quarterbacks["predicted"],
    )
    realized = quarterbacks.groupby(["season", "team"])["observed"].idxmax()
    scores = {
        "prior workload": pd.to_numeric(
            quarterbacks["prior_qb_snap_share"], errors="coerce"
        ).fillna(
            pd.to_numeric(quarterbacks["prior_pass_role"], errors="coerce").fillna(0)
        ),
        "depth chart": (
            pd.to_numeric(quarterbacks["qb_listed_starter"], errors="coerce").fillna(0)
            * 1_000_000
            - pd.to_numeric(quarterbacks["qb_depth_rank"], errors="coerce").fillna(99)
            * 1_000
            + pd.to_numeric(quarterbacks["prior_pass_role"], errors="coerce").fillna(0)
        ),
        "workload model": quarterbacks["predicted"],
    }
    realized_keys = set(realized)
    for label, score in scores.items():
        selected = score.groupby(
            [quarterbacks["season"], quarterbacks["team"]]
        ).idxmax()
        accuracy = np.mean([index in realized_keys for index in selected])
        print(f"QB1 by snap workload ({label}): top-1 accuracy={accuracy:.3f}")


def _team_rate(player_rows, team_rows, column):
    lookup = team_rows.set_index(["season", "team"])[column]
    keys = pd.MultiIndex.from_frame(player_rows[["season", "team"]])
    return lookup.reindex(keys).to_numpy(dtype=float)


def _point_player_prediction(train, test, team_test, method, stream):
    if method == "persistence":
        shares = persistence_shares(test, stream)
    elif method == "ridge":
        shares = RidgeRosterBaseline(stream).fit(train).predict_shares(test)
    elif method == "xgboost":
        shares = XGBoostRosterBaseline(stream).fit(train).predict_shares(test)
    else:
        raise ValueError(method)
    rate_column = {
        "pass": "prior_pass_attempts_per_game",
        "target": "prior_targets_per_game",
        "carry": "prior_rush_attempts_per_game",
    }[stream]
    return shares * _team_rate(test, team_test, rate_column)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=[2014, 2015, 2016, 2017, 2018, 2019, 2020]
    )
    parser.add_argument("--source", choices=("auto", "legacy", "nflverse"), default="legacy")
    parser.add_argument(
        "--roster-mode",
        choices=("auto", "point_in_time", "inferred"),
        default="auto",
    )
    parser.add_argument("--roster-cutoff-week", type=int, default=1)
    parser.add_argument("--holdout-season", type=int)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--nuts-sampler", choices=("pymc", "nutpie"), default="pymc")
    parser.add_argument("--skip-bayesian", action="store_true")
    parser.add_argument("--xgboost", action="store_true")
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args(argv)

    data = build_season_average_data(
        args.seasons,
        source=args.source,
        roster_mode=args.roster_mode,
        roster_cutoff_week=args.roster_cutoff_week,
    )
    holdout = args.holdout_season or max(args.seasons)
    train = SeasonAverageData(
        team_rows=data.team_rows[data.team_rows["season"] < holdout].copy(),
        player_rows=data.player_rows[data.player_rows["season"] < holdout].copy(),
    )
    test = SeasonAverageData(
        team_rows=data.team_rows[data.team_rows["season"] == holdout].copy(),
        player_rows=data.player_rows[data.player_rows["season"] == holdout].copy(),
    )
    if train.team_rows.empty or test.team_rows.empty:
        raise ValueError("holdout needs at least one earlier transition season")
    print(
        f"train team-seasons={len(train.team_rows):,}, player-seasons={len(train.player_rows):,}; "
        f"holdout={holdout}: teams={len(test.team_rows)}, players={len(test.player_rows):,}"
    )
    print("positions=" + ",".join(sorted(test.player_rows["position"].unique())))
    print(
        "roster snapshots="
        + ",".join(sorted(test.player_rows["roster_snapshot_source"].unique()))
    )

    print("\nTeam persistence baselines")
    for label, observed, prior in (
        (
            "opportunity plays/game",
            "opportunity_plays_per_game",
            "prior_opportunity_plays_per_game",
        ),
        ("pass attempts/game", "pass_attempts_per_game", "prior_pass_attempts_per_game"),
        ("targets/game", "targets_per_game", "prior_targets_per_game"),
        ("rush attempts/game", "rush_attempts_per_game", "prior_rush_attempts_per_game"),
    ):
        _point_metrics(label, test.team_rows[observed], test.team_rows[prior])

    observed = {
        "pass": test.player_rows["pass_att"].to_numpy(dtype=float)
        / test.player_rows["team_games"].to_numpy(dtype=float),
        "target": test.player_rows["targets"].to_numpy(dtype=float)
        / test.player_rows["team_games"].to_numpy(dtype=float),
        "carry": test.player_rows["rush_att"].to_numpy(dtype=float)
        / test.player_rows["team_games"].to_numpy(dtype=float),
    }
    print("\nPlayer per-team-game point baselines")
    for stream in ("pass", "target", "carry"):
        for method in ("persistence", "ridge"):
            prediction = _point_player_prediction(
                train.player_rows, test.player_rows, test.team_rows, method, stream
            )
            _point_metrics(f"{stream} {method}", observed[stream], prediction)
            if stream == "pass":
                quarterback = test.player_rows["position"].eq("QB").to_numpy()
                _point_metrics(
                    f"{stream} {method} (QB only)",
                    observed[stream][quarterback],
                    prediction[quarterback],
                )

    _point_metrics(
        "availability prior",
        test.player_rows["observed_availability"],
        test.player_rows["prior_availability"].fillna(
            train.player_rows["observed_availability"].median()
        ),
    )

    if args.xgboost:
        if not xgboost_available():
            print("xgboost challenger skipped: install ffmodel[ml]")
        else:
            for stream in ("pass", "target", "carry"):
                prediction = _point_player_prediction(
                    train.player_rows, test.player_rows, test.team_rows, "xgboost", stream
                )
                _point_metrics(f"{stream} xgboost", observed[stream], prediction)

    if args.skip_bayesian:
        return

    fit_kwargs = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "nuts_sampler": args.nuts_sampler,
    }
    pipeline = SeasonAverageVolumePipeline().fit(train, **fit_kwargs)
    if args.save_dir:
        print(f"saved posterior pipeline to {pipeline.save(args.save_dir)}")
    prediction = pipeline.predict_samples(test, seed=42)
    print("\nBayesian fit seconds: " + ", ".join(
        f"{name}={seconds:.1f}" for name, seconds in pipeline.fit_seconds.items()
    ))
    _distribution_metrics(
        "team opportunity plays/game",
        test.team_rows["opportunity_plays_per_game"],
        prediction.team["opportunity_plays_per_game"],
    )
    sacks_observed = test.team_rows["sacks_observed"].to_numpy(dtype=bool)
    if sacks_observed.any():
        _distribution_metrics(
            "team official plays/game",
            test.team_rows.loc[sacks_observed, "plays_per_game"],
            prediction.team["plays_per_game"][sacks_observed],
        )
        _distribution_metrics(
            "team sacks/game",
            test.team_rows.loc[sacks_observed, "sacks_per_game"],
            prediction.team["sacks_per_game"][sacks_observed],
        )
    else:
        print("team official plays/sacks: not scored (source has no sack observations)")
    _distribution_metrics(
        "team pass attempts/game",
        test.team_rows["pass_attempts_per_game"],
        prediction.team["pass_attempts_per_game"],
    )
    _distribution_metrics(
        "team targets/game",
        test.team_rows["targets_per_game"],
        prediction.team["targets_per_game"],
    )
    _distribution_metrics(
        "team rush attempts/game",
        test.team_rows["rush_attempts_per_game"],
        prediction.team["rush_attempts_per_game"],
    )

    player_observed_target = (
        prediction.player_rows["targets"].to_numpy(dtype=float)
        / prediction.player_rows["team_games"].to_numpy(dtype=float)
    )
    player_observed_carry = (
        prediction.player_rows["rush_att"].to_numpy(dtype=float)
        / prediction.player_rows["team_games"].to_numpy(dtype=float)
    )
    player_observed_pass = (
        prediction.player_rows["pass_att"].to_numpy(dtype=float)
        / prediction.player_rows["team_games"].to_numpy(dtype=float)
    )
    _distribution_metrics(
        "player availability",
        prediction.player_rows["observed_availability"],
        prediction.availability,
    )
    _workload_metrics(prediction.player_rows, prediction.qb_workload_share)
    _distribution_metrics(
        "player pass attempts/team-game",
        player_observed_pass,
        prediction.pass_attempts_per_team_game,
    )
    quarterback = prediction.player_rows["position"].eq("QB").to_numpy()
    _distribution_metrics(
        "player pass attempts/team-game (QB only)",
        player_observed_pass[quarterback],
        prediction.pass_attempts_per_team_game[quarterback],
    )
    _distribution_metrics(
        "player targets/team-game",
        player_observed_target,
        prediction.targets_per_team_game,
    )
    _distribution_metrics(
        "player carries/team-game",
        player_observed_carry,
        prediction.carries_per_team_game,
    )
    for name, result in pipeline.diagnostics().items():
        print(
            f"{name} quality: passed={result['passed']} "
            f"max_rhat={result['max_rhat']:.3f} "
            f"min_bulk_ess={result['min_bulk_ess']:.0f} "
            f"divergences={result['divergences']}"
        )


if __name__ == "__main__":
    main()
