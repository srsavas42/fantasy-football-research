"""What the NGS screen actually found: catch rate is fitting an empty design.

The tracking screen flagged one large signal -- a receiver's average intended
air yards last season against his catch rate this season, 2.8% of the residual
the shrunk prior leaves. Chasing it down, it is not a tracking signal at all.
NGS aDOT and the play-by-play aDOT already sitting in the cache
(``prior_rec_air_yards_per_target``) correlate at 0.861, the pbp version carries
the same 2.2-2.9%, and NGS adds 0.06% once pbp aDOT is controlled for
(p = 0.38). The pbp version also covers 2022 rows against NGS's 1207, because
NGS charts a qualifying subset and play-by-play charts everyone.

So the finding is about the model, not the feed. ``rec_catch_rate`` runs in
``persistence`` mean mode, whose covariate design is empty by construction --
the shrunk prior with a fitted intercept, position offsets and a slope, nothing
else. Its ``advanced_features`` block *names* ``prior_rec_air_yards_per_target``
and the mode discards it. A deep threat's catch rate is structurally low, the
model has the number that says so, and it is not being read.

The arms:

    baseline    the shipping configuration, mean_mode="persistence"
    posterior   mean_mode="posterior", which admits the spec's own covariate
                block: prior + exposure + volume + base + advanced + teammate
    ngs         that, plus NGS aDOT, to confirm the tracking version is the
                passenger the descriptive work says it is

``rush_td_rate`` gets the same treatment as the only other persistence response
with a flagged metric (NGS ``efficiency``, r = -0.117), and as a check that a
mode change is not just generically good. ``rec_td_rate`` is left out: nothing
correlated with it at all, so there is no hypothesis to test.

    python scripts/validate_catch_rate_covariates.py --holdouts 2023
    python scripts/validate_catch_rate_covariates.py --merge a.json b.json c.json
"""

from __future__ import annotations

import argparse
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
    PosteriorSeasonEfficiencyModel,
)

MATERIAL = 0.0025

# response -> (NGS stat type, the NGS field the screen flagged)
NGS_FIELD = {
    "rec_catch_rate": ("receiving", "targets", "avg_intended_air_yards"),
    "rush_td_rate": ("rushing", "rush_attempts", "efficiency"),
}

ARMS = ("baseline", "posterior", "ngs")


def _metrics(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    mean = samples.mean(axis=1)
    return {
        "mae": float(np.abs(observed - mean).mean()),
        "rmse": float(np.sqrt(np.mean((observed - mean) ** 2))),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
        "n": int(len(observed)),
    }


def _ngs_column(stat_type: str, exposure: str, field: str, seasons: list[int]) -> pd.DataFrame:
    """Exposure-weighted season aggregate of the weekly charting, lagged a year.

    The published season rows are a shorter leaderboard than the weekly ones, so
    the weeks are summed here; a per-play average is re-weighted by the exposure
    of the weeks it was charted in rather than averaged flat.
    """
    import nflreadpy as nfl

    d = nfl.load_nextgen_stats(seasons=seasons, stat_type=stat_type).to_pandas()
    d = d[d.season_type.eq("REG") & d.week.gt(0)].copy()
    weight = pd.to_numeric(d[exposure], errors="coerce").fillna(0.0)
    value = pd.to_numeric(d[field], errors="coerce")
    ok = value.notna() & weight.gt(0)
    agg = (
        pd.DataFrame({
            "season": d.season, "player_id": d.player_gsis_id,
            "num": (value * weight).where(ok, 0.0), "den": weight.where(ok, 0.0),
        })
        .groupby(["season", "player_id"], as_index=False)[["num", "den"]].sum()
    )
    name = f"ngs_{field}"
    agg[name] = np.where(agg.den.gt(0), agg.num / agg.den, np.nan)
    agg["season"] = agg.season + 1  # last season's charting, this season's row
    return agg[["season", "player_id", name]]


def _evaluate(train, test, target, arm, *, fit_kwargs, seed):
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    field = NGS_FIELD[target][2]
    if arm == "baseline":
        mode, extra = PERSISTENCE_MEAN_MODE[target], ()
    elif arm == "posterior":
        mode, extra = "posterior", ()
    else:
        mode, extra = "posterior", (f"ngs_{field}",)
    model = PosteriorSeasonEfficiencyModel(spec=spec, mean_mode=mode, extra_features=extra)
    model.fit(train, **fit_kwargs)
    prediction = model.predict_samples(test, seed=seed)

    rows = prediction.rows
    samples = np.asarray(prediction.rate, dtype=float)
    observed = pd.to_numeric(rows[target], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(observed) & np.isfinite(samples).all(axis=1)
    observed, samples = observed[keep], samples[keep]
    out = {"overall": _metrics(observed, samples), "features": len(model.feature_names)}

    # The population the covariate is supposed to move: the tail of the flagged
    # metric, where a structurally low rate is being predicted from the mean.
    charted = pd.to_numeric(rows.get(f"ngs_{field}"), errors="coerce").to_numpy()[keep]
    if np.isfinite(charted).sum() >= 40:
        cut = np.nanquantile(charted, 0.75)
        deep = np.isfinite(charted) & (charted >= cut)
        if deep.sum() >= 20:
            out["high_metric"] = _metrics(observed[deep], samples[deep])
    covered = np.isfinite(charted)
    if covered.sum() >= 40:
        out["ngs_covered"] = _metrics(observed[covered], samples[covered])
    del model, prediction, samples
    gc.collect()
    return out


def _report(report: dict, args) -> int:
    folds = report["folds"]
    holdouts = [h for h in sorted(int(k) for k in folds)]
    for target in report["targets"]:
        print(f"\n{'=' * 88}\n{target}   (baseline = shipping persistence mode)\n{'=' * 88}")
        for population in ("overall", "ngs_covered", "high_metric"):
            rows = []
            for arm in ARMS:
                values = [
                    folds[str(h)][target][arm][population]
                    for h in holdouts
                    if population in folds[str(h)][target][arm]
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
                scored = [h for h in holdouts if population in folds[str(h)][target][arm]]
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
    parser.add_argument("--targets", nargs="+", default=list(NGS_FIELD))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-json", type=Path, default=Path("reports/catch_rate_covariates.json")
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
    # NGS charting starts in 2016 and the feed refuses a season it has not
    # played; the cache spans 2014 to the projection season.
    seasons = [
        int(s) for s in sorted(player_rows.season.dropna().unique())
        if 2016 <= int(s) <= 2025
    ]
    for target in args.targets:
        stat_type, exposure, field = NGS_FIELD[target]
        column = _ngs_column(stat_type, exposure, field, seasons)
        player_rows = player_rows.merge(column, on=["season", "player_id"], how="left")
        got = player_rows[f"ngs_{field}"].notna().sum()
        print(f"ngs_{field}: present on {int(got)} of {len(player_rows)} rows", flush=True)

    fit_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    report: dict[str, object] = {"targets": args.targets, "folds": {}}
    for holdout in args.holdouts:
        train = player_rows[player_rows.season.lt(holdout)].copy()
        test = player_rows[player_rows.season.eq(holdout)].copy()
        fold: dict = {}
        for target in args.targets:
            fold[target] = {}
            for arm in ARMS:
                fold[target][arm] = _evaluate(
                    train, test, target, arm, fit_kwargs=fit_kwargs, seed=args.seed
                )
                block = fold[target][arm]["overall"]
                print(
                    f"{holdout} {target:16s} {arm:10s} CRPS {block['crps']:.6f}  "
                    f"MAE {block['mae']:.6f}  feat {fold[target][arm]['features']}",
                    flush=True,
                )
        report["folds"][str(holdout)] = fold
    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
