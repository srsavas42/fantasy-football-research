"""Walk-forward validation for the coherent team-to-player volume pipeline.

Example:

    python scripts/validate_volume_models.py --seasons 2018 2019 2020 \
        --draws 500 --tune 500 --chains 2 --nuts-sampler nutpie

The final four weeks of the latest requested season are held out. The report
first validates team plays, passes, and targets, then evaluates both conditional
allocation and the end-to-end player target/carry distributions. Team-weeks
that violate ``targets <= pass_attempts`` are reported and excluded because the
legacy source is missing a QB stat line for a small number of historical games.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.features import build_features
from ffmodel.features.volume import opportunity_accounting_summary
from ffmodel.models.volume_pipeline import VolumePipeline
from ffmodel.models.volume_team import prepare_team_weeks


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


def _history_baseline(train_teams, test_teams, draws, *, window=8, seed=0):
    """Bootstrap paired recent team games as a coherent volume baseline."""
    rng = np.random.default_rng(seed)
    values = {
        "plays": np.zeros((len(test_teams), draws), dtype=int),
        "pass_attempts": np.zeros((len(test_teams), draws), dtype=int),
        "targets": np.zeros((len(test_teams), draws), dtype=int),
    }
    global_history = train_teams.tail(max(window, 1))
    for row_i, (_, row) in enumerate(test_teams.iterrows()):
        history = train_teams.loc[train_teams["team"] == row["team"]].tail(window)
        if history.empty:
            history = global_history
        picked = rng.integers(0, len(history), size=draws)
        values["plays"][row_i] = history["team_plays"].to_numpy(dtype=int)[picked]
        values["pass_attempts"][row_i] = history["team_pass_att"].to_numpy(dtype=int)[picked]
        target_column = (
            "team_target_support"
            if "team_target_support" in history
            else "team_targets"
        )
        values["targets"][row_i] = history[target_column].to_numpy(dtype=int)[picked]
    return values


def _align_team_draws(group_keys, team_rows, values):
    keys = ["season", "week", "team"]
    lookup = pd.MultiIndex.from_frame(team_rows[keys])
    requested = pd.MultiIndex.from_frame(group_keys[keys])
    positions = lookup.get_indexer(requested)
    if (positions < 0).any():
        raise ValueError("share-model team-week is missing from team-volume predictions")
    return values[positions]


def _share_persistence_baseline(prediction, feature, totals, *, seed=0):
    """Allocate totals using leak-free trailing share as a persistence baseline."""
    rng = np.random.default_rng(seed)
    rows = prediction.rows
    draws = totals.shape[1]
    counts = np.zeros((len(rows), draws), dtype=int)
    for group_i in range(len(prediction.group_keys)):
        row_idx = np.flatnonzero(rows["_group_idx"].to_numpy(dtype=int) == group_i)
        raw = pd.to_numeric(rows.iloc[row_idx][feature], errors="coerce").fillna(0.0)
        probability = np.clip(raw.to_numpy(dtype=float), 1e-4, None)
        probability /= probability.sum()
        for draw in range(draws):
            counts[row_idx, draw] = rng.multinomial(
                int(totals[group_i, draw]), probability
            )
    return counts


def _print_quality(label: str, result: dict[str, object]) -> None:
    print(
        f"{label} quality: passed={result['passed']} "
        f"max_rhat={result['max_rhat']:.3f} "
        f"min_bulk_ess={result['min_bulk_ess']:.0f} "
        f"divergences={result['divergences']}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2018, 2019, 2020])
    parser.add_argument("--source", choices=("auto", "legacy", "nflverse"), default="legacy")
    parser.add_argument("--holdout-weeks", type=int, default=4)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--nuts-sampler", choices=("pymc", "nutpie"), default="pymc")
    parser.add_argument("--baseline-window", type=int, default=8)
    parser.add_argument("--min-bulk-ess", type=float, default=100.0)
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args(argv)

    raw_features = build_features(args.seasons, source=args.source, with_context=False)
    accounting = opportunity_accounting_summary(raw_features)
    valid = raw_features.get(
        "team_opportunity_valid", pd.Series(True, index=raw_features.index)
    ).fillna(True)
    features = raw_features.loc[valid].copy()
    print(
        f"accounting: team_weeks={accounting['team_weeks']:,} "
        f"invalid={accounting['invalid_team_weeks']:,} "
        f"mean_abs_target_gap={accounting['mean_absolute_target_gap']:.3f}; "
        f"excluded_player_rows={(~valid).sum():,}"
    )

    holdout_season = max(args.seasons)
    train, test, first_holdout = _split(
        features, holdout_season, args.holdout_weeks
    )
    fit_kw = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "nuts_sampler": args.nuts_sampler,
    }
    print(
        f"train rows={len(train):,}; holdout={holdout_season} weeks "
        f"{first_holdout}+ ({len(test):,} rows); Vegas disabled"
    )

    pipeline = VolumePipeline().fit(train, **fit_kw)
    for label, seconds in pipeline.fit_seconds.items():
        print(f"{label} model fit seconds={seconds:.1f}")
    if args.save_dir:
        saved = pipeline.save(args.save_dir)
        print(f"saved posterior pipeline to {saved}")

    prediction = pipeline.predict_samples(test)
    test_teams = prepare_team_weeks(test)
    team_pred = prediction.team
    draws = team_pred["plays"].shape[1]
    train_teams = prepare_team_weeks(train)
    team_baseline = _history_baseline(
        train_teams, test_teams, draws, window=args.baseline_window
    )
    for label, column, sample_key in (
        ("team plays", "team_plays", "plays"),
        ("team pass attempts", "team_pass_att", "pass_attempts"),
        ("team target support", "team_target_support", "targets"),
    ):
        _print_distribution_metrics(label, test_teams[column], team_pred[sample_key])
        _print_distribution_metrics(
            f"{label} baseline (last {args.baseline_window})",
            test_teams[column],
            team_baseline[sample_key],
        )

    streams = (
        ("target", pipeline.target_model, prediction.targets, "targets"),
        ("carry", pipeline.carry_model, prediction.carries, "rush_att"),
    )
    for stream, model, joint, outcome in streams:
        conditional = model.predict_samples(test)
        observed = conditional.rows[outcome].to_numpy(dtype=float)
        share_feature = f"ewma_{stream}_share"
        conditional_baseline = _share_persistence_baseline(
            conditional, share_feature, conditional.group_totals, seed=11
        )
        _print_distribution_metrics(
            f"player {outcome} (observed team total)", observed, conditional.counts
        )
        _print_distribution_metrics(
            f"player {outcome} persistence baseline", observed, conditional_baseline
        )

        if stream == "target":
            baseline_totals = team_baseline["targets"]
        else:
            baseline_totals = (
                team_baseline["plays"] - team_baseline["pass_attempts"]
            )
        baseline_totals = _align_team_draws(
            joint.group_keys, test_teams, baseline_totals
        )
        joint_baseline = _share_persistence_baseline(
            joint, share_feature, baseline_totals, seed=29
        )
        joint_observed = joint.rows[outcome].to_numpy(dtype=float)
        _print_distribution_metrics(
            f"player {outcome} joint volume", joint_observed, joint.counts
        )
        _print_distribution_metrics(
            f"player {outcome} joint baseline", joint_observed, joint_baseline
        )
        conserved = np.all(
            joint.counts.sum(axis=0) == joint.group_totals.sum(axis=0)
        )
        print(f"{stream} allocation conserves totals={conserved}")

    for label, result in pipeline.diagnostics(min_bulk_ess=args.min_bulk_ess).items():
        _print_quality(label, result)


if __name__ == "__main__":
    main()
