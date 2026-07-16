"""Walk-forward smoke validation for the Phase 3B volume models.

Example (small diagnostic run):

    python scripts/validate_volume_models.py --seasons 2018 2019 2020 \
        --draws 250 --tune 250 --chains 2

The holdout is the final four weeks of the last requested season.  Feature
construction remains leak-free, and no Vegas columns are used.  Increase to at
least 1,000 draws and four chains before treating convergence or coverage as a
model-selection result.
"""

from __future__ import annotations

import argparse

import numpy as np

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.features import build_features
from ffmodel.models.base import convergence_summary
from ffmodel.models.volume_share import OpportunityShareModel
from ffmodel.models.volume_team import TeamVolumeModel, prepare_team_weeks


def _split(frame, holdout_season: int, holdout_weeks: int):
    last_week = int(frame.loc[frame["season"] == holdout_season, "week"].max())
    first_holdout = last_week - holdout_weeks + 1
    test_mask = (frame["season"] == holdout_season) & (frame["week"] >= first_holdout)
    return frame.loc[~test_mask].copy(), frame.loc[test_mask].copy(), first_holdout


def _print_distribution_metrics(label, observed, samples):
    observed = np.asarray(observed, dtype=float)
    mean = samples.mean(axis=1)
    coverage = interval_coverage(observed, samples, level=0.8)["coverage"]
    print(
        f"{label}: MAE={np.abs(observed - mean).mean():.3f} "
        f"CRPS={empirical_crps(observed, samples).mean():.3f} "
        f"80% coverage={coverage:.3f}"
    )


def _diagnostics(label, idata, variables):
    summary, converged = convergence_summary(idata, var_names=variables)
    worst_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
    print(f"{label}: converged={converged} max_rhat={worst_rhat:.3f} min_bulk_ess={min_ess:.0f}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2018, 2019, 2020])
    parser.add_argument("--source", choices=("auto", "legacy", "nflverse"), default="legacy")
    parser.add_argument("--holdout-weeks", type=int, default=4)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args(argv)

    features = build_features(
        args.seasons, source=args.source, with_context=False
    )
    holdout_season = max(args.seasons)
    train, test, first_holdout = _split(
        features, holdout_season, args.holdout_weeks
    )
    fit_kw = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    print(
        f"train rows={len(train):,}; holdout={holdout_season} weeks "
        f"{first_holdout}+ ({len(test):,} rows); Vegas disabled"
    )

    team_model = TeamVolumeModel().fit(prepare_team_weeks(train), **fit_kw)
    test_teams = prepare_team_weeks(test)
    team_pred = team_model.predict_samples(test_teams)
    _print_distribution_metrics("team plays", test_teams["team_plays"], team_pred["plays"])
    _print_distribution_metrics(
        "team pass attempts", test_teams["team_pass_att"], team_pred["pass_attempts"]
    )
    _diagnostics(
        "team model",
        team_model.idata,
        ["play_intercept", "pass_intercept", "play_beta", "pass_beta", "play_alpha"],
    )

    for stream in ("target", "carry"):
        model = OpportunityShareModel(stream).fit(train, **fit_kw)
        prediction = model.predict_samples(test)
        observed = prediction.rows[model.outcome_col].to_numpy(dtype=float)
        _print_distribution_metrics(
            f"player {model.outcome_col}", observed, prediction.counts
        )
        conserved = np.all(prediction.counts.sum(axis=0) == prediction.group_totals.sum(axis=0))
        print(f"{stream} allocation conserves totals={conserved}")
        _diagnostics(
            f"{stream} share model",
            model.idata,
            ["mu_position", "position_sd", "beta", "player_sd"],
        )


if __name__ == "__main__":
    main()
