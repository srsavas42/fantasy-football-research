"""Does a player's own history beat the one season the layer predicts from?

Every efficiency response is predicted from a single prior season, and that
season is a noisy measurement of the thing being predicted. Split-half
reliability puts yards per target at 42.7% signal and receiving touchdown rate
at 26.7%, while the underlying talent barely moves -- about 0.93 correlation
year to year once the noise is divided out. So most of the prior's error is
measurement error in the prior, which no covariate beside it can repair.

``prior_<response>_career`` is a decayed, exposure-weighted accumulation of the
player's earlier seasons: numerators and denominators summed with 0.7 per season
of decay, then shrunk toward the same season-and-position mean the one-year
prior uses. Descriptively, against the one-season prior on the population where
the response is observed at 30+ opportunities:

    rush_yards_per_carry     8.8% -> 15.5%
    rec_catch_rate          20.9% -> 26.8%
    rec_yards_per_target     9.3% -> 14.7%
    rush_td_rate             4.8% ->  5.7%
    rec_td_rate              2.6% ->  3.7%

It also beats the rate-EWMA that ``season_pathways`` already builds, on every
response, because that one averages per-season *rates* and so counts a
twenty-target season and a hundred-and-fifty-target season equally.

Arms, per response:

    baseline    the shipping spec
    career      plus that response's own career prior

Added rather than substituted. The one-season prior drives the shrinkage
backbone through ``_prior_signal``; replacing it would change the model's spine
and the covariate question at the same time, and the two would not be separable
in the result.

``rec_td_rate`` cannot be tested this way -- it runs in persistence mode, whose
covariate design is empty by construction -- so it is reported as skipped rather
than silently returning the baseline twice.

    python scripts/validate_career_priors.py --holdouts 2023
    python scripts/validate_career_priors.py --merge a.json b.json c.json
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    PERSISTENCE_MEAN_MODE,
    POSTERIOR_MEAN_MODE,
    PosteriorSeasonEfficiencyModel,
)

MATERIAL = 0.0025
TARGETS = (
    "rec_yards_per_target", "rec_catch_rate", "rush_yards_per_carry",
    "rush_td_rate", "pass_yards_per_attempt", "pass_completion_rate",
    "pass_td_rate",
)
ARMS = ("baseline", "career")


def mean_mode(target: str) -> str:
    return PERSISTENCE_MEAN_MODE.get(target) or POSTERIOR_MEAN_MODE[target]


def _metrics(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    mean = samples.mean(axis=1)
    return {
        "mae": float(np.abs(observed - mean).mean()),
        "rmse": float(np.sqrt(np.mean((observed - mean) ** 2))),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
        "n": int(len(observed)),
    }


def _evaluate(train, test, target, arm, *, fit_kwargs, seed):
    base = EFFICIENCY_MODEL_BY_TARGET[target]
    feature = f"prior_{target}_career"
    spec = base
    if arm == "career":
        spec = dataclasses.replace(
            base, advanced_features=tuple(base.advanced_features) + (feature,)
        )
    mode = mean_mode(target)
    model = PosteriorSeasonEfficiencyModel(spec=spec, mean_mode=mode)
    model.fit(train, **fit_kwargs)
    held = model._eligible(test)
    prediction = model.predict_samples(held, seed=seed)
    # Beta-Binomial responses are scored on the posterior predictive at realized
    # exposure; the latent draws carry no sampling noise and scoring against
    # them charges whichever arm fits the location more tightly.
    predictive = np.asarray(
        model.predict_observed_samples(held, seed=seed), dtype=float
    )
    rows = prediction.rows
    observed = pd.to_numeric(rows[target], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(observed) & np.isfinite(predictive).all(axis=1)
    observed, samples = observed[keep], predictive[keep]
    out = {
        "overall": _metrics(observed, samples),
        "features": len(model.feature_names),
        "ridge_features": (
            len(model.ridge_model.feature_names) if model.ridge_model else 0
        ),
    }
    history = pd.to_numeric(rows.get(feature), errors="coerce").to_numpy()[keep]
    covered = np.isfinite(history)
    if covered.sum() >= 40:
        out["has_history"] = _metrics(observed[covered], samples[covered])
    if (~covered).sum() >= 20:
        out["no_history"] = _metrics(observed[~covered], samples[~covered])
    del model, prediction, predictive
    gc.collect()
    return out


def _report(report: dict, args) -> int:
    folds = report["folds"]
    holdouts = sorted(int(k) for k in folds)
    for target in report["targets"]:
        print(f"\n{'=' * 90}\n{target}   mode={mean_mode(target)}\n{'=' * 90}")
        for population in ("overall", "has_history", "no_history"):
            rows = []
            for arm in ARMS:
                values = [
                    folds[str(h)][target][arm][population]
                    for h in holdouts
                    if target in folds[str(h)]
                    and population in folds[str(h)][target][arm]
                ]
                if not values:
                    continue
                rows.append({
                    "arm": arm,
                    "n": int(np.mean([v["n"] for v in values])),
                    **{m: float(np.mean([v[m] for v in values]))
                       for m in ("mae", "crps", "coverage_80")},
                })
            if len(rows) < 2:
                continue
            table = pd.DataFrame(rows).set_index("arm")
            base = table.loc["baseline"]
            for metric in ("mae", "crps"):
                table[f"{metric}_delta"] = (table[metric] - base[metric]) / base[metric]
            for arm in ARMS:
                scored = [
                    h for h in holdouts
                    if target in folds[str(h)]
                    and population in folds[str(h)][target][arm]
                ]
                wins = sum(
                    folds[str(h)][target][arm][population]["crps"]
                    < folds[str(h)][target]["baseline"][population]["crps"]
                    for h in scored
                )
                table.loc[arm, "crps_folds_won"] = f"{wins}/{len(scored)}"
            print(f"\n-- {population} (n~{int(table['n'].iloc[0])}) --")
            print(table[
                ["mae", "crps", "coverage_80", "mae_delta", "crps_delta", "crps_folds_won"]
            ].to_string(float_format=lambda v: f"{v:.5f}" if abs(v) > 1e-3 else f"{v:+.2%}"))
    print(f"\nmateriality floor {MATERIAL:.2%}; a smaller move is not a result")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, default=str), "utf-8")
    print(f"wrote {args.report_json}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--draws", type=int, default=600)
    parser.add_argument("--tune", type=int, default=600)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-json", type=Path, default=Path("reports/career_priors.json")
    )
    parser.add_argument("--merge", type=Path, nargs="+", default=None)
    args = parser.parse_args(argv)

    if args.merge:
        folds: dict = {}
        targets: list[str] = []
        for path in args.merge:
            blob = json.loads(path.read_text("utf-8"))
            folds.update(blob["folds"])
            targets = blob.get("targets", targets)
        return _report({"targets": targets, "folds": folds}, args)

    player_rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    selected = []
    for target in args.targets:
        column = f"prior_{target}_career"
        if column not in player_rows:
            raise SystemExit(
                f"{column} is not in the cache; rebuild it before validating"
            )
        if mean_mode(target) == "persistence":
            print(f"skipping {target}: persistence mode fits an empty design")
            continue
        selected.append(target)
        got = player_rows[column].notna().sum()
        print(f"{column}: {int(got)} of {len(player_rows)} rows", flush=True)

    fit_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    report: dict[str, object] = {"targets": selected, "folds": {}}
    for holdout in args.holdouts:
        train = player_rows[player_rows.season.lt(holdout)].copy()
        test = player_rows[player_rows.season.eq(holdout)].copy()
        fold: dict = {}
        for target in selected:
            fold[target] = {}
            for arm in ARMS:
                fold[target][arm] = _evaluate(
                    train, test, target, arm, fit_kwargs=fit_kwargs, seed=args.seed
                )
                block = fold[target][arm]["overall"]
                print(
                    f"{holdout} {target:24s} {arm:9s} CRPS {block['crps']:.6f}  "
                    f"MAE {block['mae']:.6f}",
                    flush=True,
                )
        report["folds"][str(holdout)] = fold
    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
