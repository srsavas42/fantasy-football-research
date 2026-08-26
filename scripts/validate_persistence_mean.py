"""Paired walk-forward of the ``prior``-mode efficiency responses.

``prior`` mode is an identity map: ``_prior_signal`` links the lagged shrunk
feature and subtracts a centre, ``_prior_mean`` re-adds that centre and inverts
the link, so the conditional mean handed to the simulator *is*
``prior_<response>``. The layer asserts a persistence coefficient of 1.000 and
its whole regression to the mean is the ``K`` pseudo-count in the feature.

``persistence`` mode replaces the assertion with an intercept, sum-to-zero
position offsets and a fitted slope, and admits no covariates. Both arms are
the real ``PosteriorSeasonEfficiencyModel`` on the same frames with the same
seeds; only ``mean_mode`` differs.

Two things this script is deliberate about.

**It reads a built frame, not a hand-assembled one.** An earlier version of
this comparison lagged `player_season_efficiency` itself and reached back to
2006, where the nflverse weekly feed reports one target for players with fifty
receptions. Those rows give catch rates above 1.0, ``_prior_signal`` sends them
to +8.5 on the logit scale, and a fitted slope through them collapses to 0.08 --
a result about broken targets, presented as a result about persistence. The
contamination stops after 2008 and the shipping window starts at 2014, so
reading the frame the pipeline actually builds avoids it, and ``check_bounds``
asserts it on every run rather than trusting it.

**It reports the fitted slope with its interval.** The point of the change is
that the shipped value is 1.000 and the data disagrees; a posterior that
straddles 1.0 would mean the change is not worth making, whatever the MAE says.

**It scores the posterior predictive, not the latent rate.** The response is an
observed season rate and carries binomial sampling noise at the player's
exposure; the latent rate draws do not. Scoring one against the other penalises
whichever arm has the tighter latent distribution, and fitting the mean is
exactly what tightens it -- so the better arm takes the larger penalty. An
earlier version of this script scored the latent draws and reported CRPS
*regressing* 2.00% on ``rec_td_rate``; on the predictive the same fits give
-0.31%, and ``rush_td_rate`` moves from -0.26% on one fold of three to -2.31% on
three of three. ``crps_latent`` is retained per fold so the difference stays
visible rather than being a claim in a comment.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    PERSISTENCE_MEAN_MODE,
    PosteriorSeasonEfficiencyModel,
)

HOLDOUTS = (2022, 2023, 2024)


def crps(samples: np.ndarray, actual: np.ndarray) -> float:
    ordered = np.sort(samples, axis=1)
    draws = ordered.shape[1]
    absolute = np.abs(ordered - actual[:, None]).mean(axis=1)
    weights = 2 * np.arange(1, draws + 1) - draws - 1
    return float((absolute - (ordered * weights).sum(axis=1) / (draws * draws)).mean())


def check_bounds(rows: pd.DataFrame, target: str) -> None:
    """A rate outside its own bounds is a data fault, not a modelling result."""
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    for column in (target, spec.prior_feature):
        values = pd.to_numeric(rows.get(column), errors="coerce").dropna()
        if values.empty:
            continue
        outside = values[(values < spec.lower) | (values > spec.upper)]
        if not outside.empty:
            raise SystemExit(
                f"{column} has {len(outside)} values outside [{spec.lower}, "
                f"{spec.upper}] (max {outside.max():.3f}). The nflverse weekly "
                "feed under-reports targets before 2009; use a frame built over "
                "the shipping window instead of widening the season range."
            )


def fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    mode: str,
    *,
    draws: int,
    chains: int,
    seed: int,
) -> dict | None:
    import arviz as az

    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    model = PosteriorSeasonEfficiencyModel(
        spec, mean_mode=mode, use_volume=False, use_advanced=False
    )
    model.fit(train, draws=draws, tune=draws, chains=chains, seed=seed)
    held = model._eligible(test)
    if held.empty:
        return None
    prediction = model.predict_samples(held, draws=1000, seed=seed - 2)
    # Score the posterior predictive at the player's realized exposure, not the
    # latent rate draws. The response is an observed season rate and carries
    # binomial sampling noise at n; the latent draws do not. Scoring one against
    # the other penalises whichever arm has the tighter latent distribution --
    # and fitting the mean is exactly what tightens it, so the better arm takes
    # the larger penalty. Measured, that reversed the sign on rec_td_rate: CRPS
    # +2.00% against the latent draws, -0.31% against the predictive.
    observed = model.predict_observed_samples(held, draws=1000, seed=seed - 2)
    actual = pd.to_numeric(held[target], errors="coerce").to_numpy(float)
    exposure = pd.to_numeric(held[spec.exposure], errors="coerce").to_numpy(float)
    weights = exposure / exposure.mean()
    low, high = np.quantile(observed, [0.10, 0.90], axis=1)
    summary = az.summary(model.idata)
    out = {
        "n": int(len(held)),
        "mae": float(
            np.average(np.abs(np.nanmean(prediction.mean, axis=1) - actual), weights=weights)
        ),
        "crps": crps(observed, actual),
        "crps_latent": crps(prediction.rate, actual),
        "cov80": float(((actual >= low) & (actual <= high)).mean()),
        "max_rhat": float(summary["r_hat"].max()),
        "divergences": int(model.idata.sample_stats.diverging.values.sum()),
    }
    if mode == "persistence":
        slope = model.idata.posterior["prior_persistence"].values
        out["slope"] = float(slope.mean())
        out["slope_low"] = float(np.percentile(slope, 2.5))
        out["slope_high"] = float(np.percentile(slope, 97.5))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player-rows", type=Path, default=Path(".cache/player_rows_2014_2025.pkl")
    )
    parser.add_argument("--draws", type=int, default=600)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--holdouts", nargs="+", type=int, default=list(HOLDOUTS))
    parser.add_argument("--targets", nargs="+", default=sorted(PERSISTENCE_MEAN_MODE))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.player_rows.exists():
        raise SystemExit(f"no frame at {args.player_rows}")
    rows = pd.read_pickle(args.player_rows)

    results = {}
    for target in args.targets:
        check_bounds(rows, target)
        print(f"\n########## {target} ##########", flush=True)
        records = []
        for holdout in args.holdouts:
            train = rows[rows["season"] < holdout]
            test = rows[rows["season"] == holdout]
            record = {"holdout": int(holdout)}
            for label, mode in (("prior", "prior"), ("persistence", "persistence")):
                result = fold(
                    train,
                    test,
                    target,
                    mode,
                    draws=args.draws,
                    chains=args.chains,
                    seed=args.seed,
                )
                if result is None:
                    break
                for key, value in result.items():
                    record[f"{label}_{key}"] = value
                print(
                    f"  {holdout} {label:<11s} MAE {result['mae']:.5f}  "
                    f"CRPS {result['crps']:.5f}  cov80 {result['cov80']:.3f}  "
                    f"rhat {result['max_rhat']:.4f}  div {result['divergences']}",
                    flush=True,
                )
            if "persistence_slope" in record:
                print(
                    f"     fitted persistence (logit scale): "
                    f"{record['persistence_slope']:.3f} "
                    f"[{record['persistence_slope_low']:.3f}, "
                    f"{record['persistence_slope_high']:.3f}]  "
                    f"-- shipped policy asserts 1.000",
                    flush=True,
                )
            if f"persistence_mae" in record:
                records.append(record)

        if not records:
            continue
        frame = pd.DataFrame(records)
        total = int(frame["prior_n"].sum())
        summary = {"folds": len(frame), "n": total}
        print(f"  ---- {target} pooled over {len(frame)} folds, n = {total} ----")
        for metric in ("mae", "crps", "cov80"):
            base = float((frame[f"prior_{metric}"] * frame["prior_n"]).sum() / total)
            new = float(
                (frame[f"persistence_{metric}"] * frame["prior_n"]).sum() / total
            )
            wins = int((frame[f"persistence_{metric}"] < frame[f"prior_{metric}"]).sum())
            summary[metric] = {"prior": base, "persistence": new, "folds_improved": wins}
            if metric == "cov80":
                print(f"    {metric:6s} {base:.5f} -> {new:.5f}   {new - base:+.4f} absolute")
            else:
                print(
                    f"    {metric:6s} {base:.5f} -> {new:.5f}   "
                    f"{100 * (new - base) / base:+.2f}%   {wins}/{len(frame)}"
                )
        slopes = frame["persistence_slope"]
        summary["slope_mean"] = float(slopes.mean())
        print(
            f"    fitted persistence across folds: "
            f"{slopes.min():.3f} to {slopes.max():.3f} (shipped asserts 1.000)"
        )
        results[target] = summary

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
