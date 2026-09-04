"""Walk-forward for the one situational feature the layer cannot already see.

``screen_situational_usage.py`` split the short-yardage hypothesis in half.

The yards-per-carry half holds. A back's share of carries with two or fewer
yards to go, taken from last season, correlates at -0.108 with this season's
yards per carry *after* the response's whole ridge design is regressed out --
1.2% of a residual that design has already cut to 21.3% explained, at p = 0.0004
and with the sign the hypothesis predicted. Short-yardage share persists at
16.3% year over year, which is enough to carry a prior-season feature.

The touchdown half does not. Short-yardage share against next season's rushing
touchdown rate is r = +0.022, p = 0.47, and goal-line share is if anything
negative. The reason is visible in the persistence table: carries inside the
five persist at 2.2% and goal-to-go carries at 1.8%. Goal-line work is assigned
by a season's circumstances, not held as a trait, so last season's cannot
forecast this season's touchdowns however real the within-season effect is.

So only ``rush_yards_per_carry`` is walked forward here, and only on the
features whose descriptive sign was right.

``rush_yards_per_carry`` runs in ridge mean mode, whose design is built from
``spec.advanced_features``; ``extra_features`` is read only in posterior mode
and would silently do nothing. The arms therefore swap the spec, not the model.

    python scripts/validate_short_yardage.py --holdouts 2023
    python scripts/validate_short_yardage.py --merge a.json b.json c.json
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

from ffmodel.evaluation.metrics import point_and_distribution
from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    PosteriorSeasonEfficiencyModel,
)

TARGET = "rush_yards_per_carry"
MATERIAL = 0.0025

ARMS = {
    "baseline": (),
    "short_yardage": ("prior_rush_short_yardage_share",),
    "situational": ("prior_rush_short_yardage_share", "prior_rush_ydstogo_mean"),
}


def rushing_context(cache: Path, seasons: list[int]) -> pd.DataFrame:
    """Per player-season carry context from pbp, lagged a season. Cached.

    Twelve seasons of play-by-play is minutes of download and parsing, and this
    is run once per fold, so the aggregate is written next to the row cache.
    """
    path = cache / "rush_context.pkl"
    if path.exists():
        return pd.read_pickle(path)
    import nflreadpy as nfl

    frames = []
    for season in seasons:
        try:
            pbp = nfl.load_pbp(seasons=[season]).to_pandas()
        except Exception as exc:
            print(f"  {season}: {type(exc).__name__} {str(exc)[:60]}", flush=True)
            continue
        pbp = pbp[pbp.season_type.eq("REG")]
        run = pbp.rush_attempt.eq(1) & pbp.rusher_player_id.notna()
        ydstogo = pd.to_numeric(pbp.ydstogo, errors="coerce")[run]
        frame = pd.DataFrame({
            "season": season,
            "player_id": pbp.rusher_player_id[run],
            "ydstogo": ydstogo,
            "short": ydstogo.le(2).astype(float),
        })
        frames.append(
            frame.groupby(["season", "player_id"]).agg(
                prior_rush_context_plays=("ydstogo", "size"),
                prior_rush_ydstogo_mean=("ydstogo", "mean"),
                prior_rush_short_yardage_share=("short", "mean"),
            ).reset_index()
        )
        del pbp
        gc.collect()
    out = pd.concat(frames, ignore_index=True)
    out["season"] = out.season + 1  # last season's context, this season's row
    out.to_pickle(path)
    return out


def _evaluate(train, test, features, *, fit_kwargs, seed):
    base = EFFICIENCY_MODEL_BY_TARGET[TARGET]
    spec = dataclasses.replace(
        base, advanced_features=tuple(base.advanced_features) + tuple(features)
    )
    model = PosteriorSeasonEfficiencyModel(spec=spec)
    model.fit(train, **fit_kwargs)
    held = model._eligible(test)
    prediction = model.predict_samples(held, seed=seed)
    # An identity-link response: predict_observed_samples returns the rate draws
    # unchanged, so latent and predictive coincide here and there is no
    # Beta-Binomial exposure step to get wrong.
    rows = prediction.rows
    samples = np.asarray(prediction.rate, dtype=float)
    observed = pd.to_numeric(rows[TARGET], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(observed) & np.isfinite(samples).all(axis=1)
    observed, samples = observed[keep], samples[keep]
    out = {
        "overall": point_and_distribution(observed, samples),
        "ridge_features": len(model.ridge_model.feature_names) if model.ridge_model else 0,
    }
    share = pd.to_numeric(
        rows.get("prior_rush_short_yardage_share"), errors="coerce"
    ).to_numpy()[keep]
    if np.isfinite(share).sum() >= 60:
        cut = np.nanquantile(share, 0.75)
        heavy = np.isfinite(share) & (share >= cut)
        if heavy.sum() >= 20:
            out["short_yardage_backs"] = point_and_distribution(
                observed[heavy], samples[heavy]
            )
        light = np.isfinite(share) & (share <= np.nanquantile(share, 0.25))
        if light.sum() >= 20:
            out["open_field_backs"] = point_and_distribution(
                observed[light], samples[light]
            )
    del model, prediction, samples
    gc.collect()
    return out


def _report(report: dict, args) -> int:
    folds = report["folds"]
    holdouts = sorted(int(k) for k in folds)
    print(f"\n{'=' * 92}\n{TARGET}   short-yardage context, holdouts {holdouts}\n{'=' * 92}")
    for population in ("overall", "short_yardage_backs", "open_field_backs"):
        rows = []
        for arm in ARMS:
            values = [
                folds[str(h)][arm][population]
                for h in holdouts if population in folds[str(h)][arm]
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
            if arm not in table.index:
                continue
            scored = [h for h in holdouts if population in folds[str(h)][arm]]
            wins = sum(
                folds[str(h)][arm][population]["crps"]
                < folds[str(h)]["baseline"][population]["crps"]
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
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--draws", type=int, default=600)
    parser.add_argument("--tune", type=int, default=600)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-json", type=Path, default=Path("reports/short_yardage.json")
    )
    parser.add_argument("--merge", type=Path, nargs="+", default=None)
    args = parser.parse_args(argv)

    if args.merge:
        folds: dict = {}
        for path in args.merge:
            folds.update(json.loads(path.read_text("utf-8"))["folds"])
        return _report({"folds": folds}, args)

    player_rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    seasons = [int(s) for s in sorted(player_rows.season.dropna().unique()) if s <= 2025]
    context = rushing_context(args.cache_dir, seasons)
    player_rows = player_rows.merge(context, on=["season", "player_id"], how="left")
    got = player_rows.prior_rush_short_yardage_share.notna().sum()
    print(f"carry context on {int(got)} of {len(player_rows)} rows", flush=True)

    fit_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    report: dict[str, object] = {"folds": {}}
    for holdout in args.holdouts:
        train = player_rows[player_rows.season.lt(holdout)].copy()
        test = player_rows[player_rows.season.eq(holdout)].copy()
        fold: dict = {}
        for arm, features in ARMS.items():
            fold[arm] = _evaluate(
                train, test, features, fit_kwargs=fit_kwargs, seed=args.seed
            )
            block = fold[arm]["overall"]
            print(
                f"{holdout} {arm:14s} CRPS {block['crps']:.6f}  MAE {block['mae']:.6f}  "
                f"ridge feat {fold[arm]['ridge_features']}",
                flush=True,
            )
        report["folds"][str(holdout)] = fold
    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
