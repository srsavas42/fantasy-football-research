"""One prior slope for the players a draft is about, another for everyone else.

The layer fits a single coefficient on the lagged response across every player,
and measured separately the coefficients differ: yards per target persists at
+0.669 among drafted players against +0.454 among the rest, catch rate at +0.854
against +0.776. A shared slope is a compromise between two populations rather
than a description of either.

The alternative -- training on the drafted players alone -- was assessed and
rejected in ``screen_adp_truncated_training.py``. It breaks the volume softmax,
whose shares only sum to one over a whole roster, and it selects the training set
on a forecast of the response, discarding the 131 player-seasons that finished
top-100 undrafted while keeping the 402 drafted ones that finished outside the
top 300. The interaction buys the same tier-specific flexibility on every row.

``prior_<response>_x_drafted`` is the lagged response centred on its
season-and-position mean and multiplied by the drafted indicator, so it carries
the *difference* in slope and leaves the level to the indicator. Uncentred, the
two terms are collinear and the design's rank reduction absorbs the interaction
instead of fitting it.

Scored on the drafted population as well as overall, because the whole point is
the players a draft is about; a gain that shows up pooled but not on them would
be beside the point.

    python scripts/validate_adp_tier_slope.py --holdouts 2023
    python scripts/validate_adp_tier_slope.py --merge a.json b.json c.json
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
ARMS = ("baseline", "tier_slope")


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
    spec = base
    if arm == "tier_slope":
        spec = dataclasses.replace(
            base,
            advanced_features=tuple(base.advanced_features)
            + (f"{base.prior_feature}_x_drafted", "adp_drafted"),
        )
    model = PosteriorSeasonEfficiencyModel(spec=spec, mean_mode=mean_mode(target))
    model.fit(train, **fit_kwargs)
    held = model._eligible(test)
    prediction = model.predict_samples(held, seed=seed)
    predictive = np.asarray(
        model.predict_observed_samples(held, seed=seed), dtype=float
    )
    rows = prediction.rows
    observed = pd.to_numeric(rows[target], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(observed) & np.isfinite(predictive).all(axis=1)
    observed, samples = observed[keep], predictive[keep]
    out = {"overall": _metrics(observed, samples), "features": len(model.feature_names)}
    drafted = (
        pd.to_numeric(rows.get("adp_drafted"), errors="coerce")
        .fillna(0).eq(1).to_numpy()[keep]
    )
    if drafted.sum() >= 25:
        out["drafted"] = _metrics(observed[drafted], samples[drafted])
    rank = pd.to_numeric(rows.get("adp_rank"), errors="coerce").to_numpy()[keep]
    early = np.isfinite(rank) & (rank <= 150)
    if early.sum() >= 25:
        out["adp_top150"] = _metrics(observed[early], samples[early])
    del model, prediction, predictive
    gc.collect()
    return out


def _report(report: dict, args) -> int:
    folds = report["folds"]
    holdouts = sorted(int(k) for k in folds)
    for target in report["targets"]:
        print(f"\n{'=' * 90}\n{target}   mode={mean_mode(target)}\n{'=' * 90}")
        for population in ("overall", "drafted", "adp_top150"):
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
        "--report-json", type=Path, default=Path("reports/adp_tier_slope.json")
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
        spec = EFFICIENCY_MODEL_BY_TARGET[target]
        column = f"{spec.prior_feature}_x_drafted"
        if column not in player_rows:
            raise SystemExit(f"{column} is not in the cache; rebuild it first")
        if mean_mode(target) == "persistence":
            print(f"skipping {target}: persistence mode fits an empty design")
            continue
        selected.append(target)

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
                    f"{holdout} {target:24s} {arm:11s} CRPS {block['crps']:.6f}  "
                    f"MAE {block['mae']:.6f}",
                    flush=True,
                )
        report["folds"][str(holdout)] = fold
    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
