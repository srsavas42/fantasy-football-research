"""Paired walk-forward of the availability layer, with and without its own history.

The layer has always read one lagged season. ``prior_availability_3yr`` -- a
career exponentially-weighted mean at alpha 0.50, grouped by ``player_key`` so
it follows a trade -- has been built for every row since the pathway features
landed and read by nothing.

Both arms are the real ``SeasonAvailabilityModel``, fitted on the same frames
with the same seeds, differing only in ``extra_features``. That matters here
more than usual, because a linear proxy answers a materially different question:
against ``prior_availability`` alone the career mean is worth -1.51% to -2.19%
depending on the frame, and against the full ten-feature design it is worth
about -0.07%. Neither is the layer. The layer is a hurdle -- Bernoulli for
playing at all, Beta-Binomial for games conditional on playing -- with
position-specific intercepts and an SVD-projected design, and the only honest
way to measure a feature in it is to fit it.

Reported per fold and pooled: MAE and CRPS on games active, 80% interval
coverage, maximum R-hat and divergences. Coverage is reported because a feature
that buys accuracy by narrowing honest intervals has not bought anything.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.models.season_availability import (
    AVAILABILITY_HISTORY_FEATURES,
    SeasonAvailabilityModel,
)
from ffmodel.features.volume import MODEL_POSITIONS

HOLDOUTS = (2022, 2023, 2024)


def crps(samples: np.ndarray, actual: np.ndarray) -> float:
    """Mean CRPS of an empirical predictive distribution, by the energy form."""
    ordered = np.sort(samples, axis=1)
    draws = ordered.shape[1]
    absolute = np.abs(ordered - actual[:, None]).mean(axis=1)
    weights = 2 * np.arange(1, draws + 1) - draws - 1
    spread = (ordered * weights).sum(axis=1) / (draws * draws)
    return float((absolute - spread).mean())


def fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    extra: tuple[str, ...],
    *,
    draws: int,
    chains: int,
    seed: int,
) -> dict:
    import arviz as az

    model = SeasonAvailabilityModel(extra_features=extra)
    model.fit(train, draws=draws, tune=draws, chains=chains, seed=seed)
    prediction = model.predict_samples(test, seed=seed - 4)
    games = np.asarray(prediction.games_active, dtype=float)
    actual = test["games"].to_numpy(float)
    low, high = np.quantile(games, [0.10, 0.90], axis=1)
    summary = az.summary(model.idata)
    return {
        "mae": float(np.abs(games.mean(axis=1) - actual).mean()),
        "crps": crps(games, actual),
        "cov80": float(((actual >= low) & (actual <= high)).mean()),
        "max_rhat": float(summary["r_hat"].max()),
        "divergences": int(model.idata.sample_stats.diverging.values.sum()),
        "fitted_features": list(model.feature_names),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player-rows",
        type=Path,
        default=Path(".cache/player_rows_2014_2025.pkl"),
        help="a built player_rows frame. Default is the shipping window, which "
        "is also what validate_persistence_mean.py reads, so the two "
        "validations in this branch are on the same data",
    )
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--holdouts", nargs="+", type=int, default=list(HOLDOUTS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.player_rows.exists():
        raise SystemExit(f"no frame at {args.player_rows}")
    rows = pd.read_pickle(args.player_rows)
    rows = rows[rows["position"].isin(MODEL_POSITIONS)].copy()
    rows["games"] = pd.to_numeric(rows["games"], errors="coerce")
    rows = rows[rows["team_games"].gt(0) & rows["games"].notna()]

    arms = {"base": (), "history": AVAILABILITY_HISTORY_FEATURES}
    records = []
    for holdout in args.holdouts:
        train = rows[rows["season"] < holdout]
        test = rows[rows["season"] == holdout]
        if train.empty or test.empty:
            continue
        record = {"holdout": int(holdout), "n": int(len(test))}
        for label, extra in arms.items():
            result = fold(
                train,
                test,
                extra,
                draws=args.draws,
                chains=args.chains,
                seed=args.seed,
            )
            for key, value in result.items():
                record[f"{label}_{key}"] = value
            print(
                f"  {holdout} {label:<8s} MAE {result['mae']:.4f}  "
                f"CRPS {result['crps']:.4f}  cov80 {result['cov80']:.3f}  "
                f"rhat {result['max_rhat']:.4f}  div {result['divergences']}",
                flush=True,
            )
        # The check that makes the comparison mean something: the arms must
        # actually differ in what they fitted, not merely in what was asked for.
        gained = set(record["history_fitted_features"]) - set(
            record["base_fitted_features"]
        )
        if gained != set(AVAILABILITY_HISTORY_FEATURES):
            raise SystemExit(
                f"the history arm fitted {sorted(gained)}, expected "
                f"{sorted(AVAILABILITY_HISTORY_FEATURES)}; the design filter "
                "dropped the column and this comparison is not what it says"
            )
        records.append(record)

    frame = pd.DataFrame(records)
    total = int(frame["n"].sum())
    print("\n=== pooled, weighted by holdout size ===")
    summary = {"holdouts": records}
    for metric in ("mae", "crps", "cov80"):
        base = float((frame[f"base_{metric}"] * frame["n"]).sum() / total)
        history = float((frame[f"history_{metric}"] * frame["n"]).sum() / total)
        wins = int((frame[f"history_{metric}"] < frame[f"base_{metric}"]).sum())
        summary[metric] = {"base": base, "history": history, "folds_improved": wins}
        if metric == "cov80":
            print(
                f"  {metric:6s} {base:.5f} -> {history:.5f}   "
                f"{history - base:+.4f} absolute"
            )
        else:
            print(
                f"  {metric:6s} {base:.5f} -> {history:.5f}   "
                f"{100 * (history - base) / base:+.2f}%   {wins}/{len(frame)} folds"
            )
    worst_rhat = float(
        max(frame["base_max_rhat"].max(), frame["history_max_rhat"].max())
    )
    divergences = int(
        frame["base_divergences"].sum() + frame["history_divergences"].sum()
    )
    print(f"  sampling: max R-hat {worst_rhat:.4f}, {divergences} divergences in total")
    summary["max_rhat"] = worst_rhat
    summary["divergences"] = divergences

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
