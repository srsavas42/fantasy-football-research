"""Does a mobile quarterback lift his running back's yards per carry?

The one survivor of the red-zone / coach / teammate screen. A back gains yards
per carry when his quarterback runs more -- r = +0.135, p < 1e-4, 1.8% of a
residual whose ridge design already explains 22.2% -- which is the light-box
mechanism: a defense that must account for the quarterback cannot load up on the
back. Larger, descriptively, than the short-yardage share that already cleared
this gate.

The reason to be suspicious anyway is that it is a team-level covariate, and
team-level covariates have failed every forecast test in this line of work:
opponent defence, three O-line proxies, team RYOE, team pressure rate, and now
head coach. What killed them was some combination of not persisting and not
surviving removal of the player himself. Neither obviously applies here -- a
quarterback's rushing is his own trait, not the back's, and it is measured on a
different player -- but the base rate for this family is bad, which is what the
walk-forward is for.

Arms are cumulative so the question is what each adds to what already ships:

    baseline    the shipping spec, short-yardage share included
    qb_share    plus the quarterback's prior-season share of team carries
    both        plus team passing efficiency, the other teammate proxy, which
                was descriptively null and is here to confirm it stays null

rush_yards_per_carry runs in ridge mean mode, whose design comes from
spec.advanced_features, so the arms swap the spec rather than passing
extra_features -- which posterior mode reads and ridge mode ignores.

    python scripts/validate_qb_carry_share.py --holdouts 2023
    python scripts/validate_qb_carry_share.py --merge a.json b.json c.json
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
    PosteriorSeasonEfficiencyModel,
)

TARGET = "rush_yards_per_carry"
MATERIAL = 0.0025

ARMS = {
    "baseline": (),
    "qb_share": ("prior_qb_carry_share",),
    "both": ("prior_qb_carry_share", "prior_team_pass_ypa"),
}


def _metrics(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    mean = samples.mean(axis=1)
    return {
        "mae": float(np.abs(observed - mean).mean()),
        "rmse": float(np.sqrt(np.mean((observed - mean) ** 2))),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
        "n": int(len(observed)),
    }


def teammate_context(cache: Path, seasons: list[int]) -> pd.DataFrame:
    """Prior-season team passing efficiency and quarterback share of carries."""
    path = cache / "teammate_context.pkl"
    if path.exists():
        return pd.read_pickle(path)
    import nflreadpy as nfl

    stats = nfl.load_player_stats(seasons=seasons).to_pandas()
    if "season_type" in stats:
        stats = stats[stats.season_type.eq("REG")]
    for column in ("passing_yards", "attempts", "carries"):
        stats[column] = pd.to_numeric(stats.get(column), errors="coerce").fillna(0.0)
    is_qb = stats["position"].astype(str).str.upper().eq("QB")
    team = stats.groupby(["season", "team"], as_index=False).agg(
        pass_yds=("passing_yards", "sum"),
        pass_att=("attempts", "sum"),
        carries=("carries", "sum"),
    )
    qb = (
        stats[is_qb].groupby(["season", "team"], as_index=False)
        .agg(qb_carries=("carries", "sum"))
    )
    team = team.merge(qb, on=["season", "team"], how="left")
    team["prior_team_pass_ypa"] = team.pass_yds / team.pass_att.replace(0, np.nan)
    team["prior_qb_carry_share"] = (
        team.qb_carries.fillna(0) / team.carries.replace(0, np.nan)
    )
    team["season"] = team.season + 1  # last season's offense, this season's row
    out = team[["season", "team", "prior_team_pass_ypa", "prior_qb_carry_share"]]
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
    # Identity link: the predictive and the latent draws coincide, so there is
    # no Beta-Binomial exposure step to get wrong here.
    rows = prediction.rows
    samples = np.asarray(prediction.rate, dtype=float)
    observed = pd.to_numeric(rows[TARGET], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(observed) & np.isfinite(samples).all(axis=1)
    observed, samples = observed[keep], samples[keep]
    out = {
        "overall": _metrics(observed, samples),
        "ridge_features": len(model.ridge_model.feature_names) if model.ridge_model else 0,
    }
    share = pd.to_numeric(
        rows.get("prior_qb_carry_share"), errors="coerce"
    ).to_numpy()[keep]
    if np.isfinite(share).sum() >= 60:
        top = np.isfinite(share) & (share >= np.nanquantile(share, 0.75))
        low = np.isfinite(share) & (share <= np.nanquantile(share, 0.25))
        if top.sum() >= 20:
            out["mobile_qb"] = _metrics(observed[top], samples[top])
        if low.sum() >= 20:
            out["pocket_qb"] = _metrics(observed[low], samples[low])
    del model, prediction, samples
    gc.collect()
    return out


def _report(report: dict, args) -> int:
    folds = report["folds"]
    holdouts = sorted(int(k) for k in folds)
    print(f"\n{'=' * 92}\n{TARGET}   quarterback carry share, holdouts {holdouts}\n{'=' * 92}")
    for population in ("overall", "mobile_qb", "pocket_qb"):
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
        "--report-json", type=Path, default=Path("reports/qb_carry_share.json")
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
    player_rows = player_rows.merge(
        teammate_context(args.cache_dir, seasons), on=["season", "team"], how="left"
    )
    got = player_rows.prior_qb_carry_share.notna().sum()
    print(f"teammate context on {int(got)} of {len(player_rows)} rows", flush=True)

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
                f"{holdout} {arm:10s} CRPS {block['crps']:.6f}  MAE {block['mae']:.6f}  "
                f"ridge feat {fold[arm]['ridge_features']}",
                flush=True,
            )
        report["folds"][str(holdout)] = fold
    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
