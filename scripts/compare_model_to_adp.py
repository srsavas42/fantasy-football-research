"""Model against the draft board, and what the disagreement between them is worth.

Two questions a drafter actually has. Is the model better than ADP? And when it
disagrees with ADP, is the disagreement information or noise?

The second is the one that decides whether the model is usable. A model can beat
ADP on average and still be useless at the table if it beats it by being a
slightly smoothed copy: what you need is for the *direction* of a disagreement
to predict which way the board is wrong.

The ADP baseline is ``RankCurve`` from the shipped blend module -- a per-position
fit of season points on log draft rank, with a local residual pool for spread --
fitted on training seasons only, so it is a genuine forecast rather than a
retrospective fit. It is the same baseline the blend weights were derived
against.

Scored on the drafted population, where both forecasts exist, and broken out by
draft tier because a pooled number over the whole board is dominated by picks
nobody agonises over.

The directional test buckets players by ``model mean - ADP mean`` and asks, in
each bucket, whether the actual outcome moved the way the model said. The
summary statistic is the correlation between the disagreement and ADP's signed
error: if the model's disagreement predicts where ADP is wrong, that correlation
is positive and the model carries information the board does not.

    python scripts/compare_model_to_adp.py --holdouts 2023
    python scripts/compare_model_to_adp.py --merge a.json b.json c.json
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

from ffmodel.evaluation.efficiency_posterior import observed_scoring_rows
from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.market_blend import RankCurve
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.simulation.scoring import fantasy_points

TIERS = (("top50", 1, 50), ("51_150", 51, 150), ("151_300", 151, 300), ("drafted", 1, 400))


def _metrics(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.abs(observed - samples.mean(axis=1)).mean()),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
        "n": int(len(observed)),
    }


def _run_fold(holdout: int, cache_dir: Path, *, draws, tune, chains, seed):
    player_rows = pd.read_pickle(cache_dir / "player_rows.pkl")
    team_rows = pd.read_pickle(cache_dir / "team_rows.pkl")
    player_rows = player_rows[player_rows.season.lt(2026)]
    team_rows = team_rows[team_rows.season.lt(2026)]
    train = SeasonAverageData(
        team_rows[team_rows.season.lt(holdout)].copy(),
        player_rows[player_rows.season.lt(holdout)].copy(),
    )
    test = SeasonAverageData(
        team_rows[team_rows.season.eq(holdout)].copy(),
        player_rows[player_rows.season.eq(holdout)].copy(),
    )
    pipeline = SeasonAverageScoringPipeline()
    kwargs = {"draws": draws, "tune": tune, "chains": chains}
    pipeline.fit(train, volume_sample_kwargs=kwargs, efficiency_sample_kwargs=kwargs)
    prediction = pipeline.predict_samples(test, seed=seed)

    rows = prediction.player_rows.reset_index(drop=True)
    observed = fantasy_points(observed_scoring_rows(rows), "ppr").to_numpy(dtype=float)
    model = np.asarray(prediction.fantasy_points["ppr"], dtype=float)

    # The ADP baseline, fitted on the training seasons only.
    train_rows = train.player_rows.reset_index(drop=True)
    train_points = fantasy_points(
        observed_scoring_rows(train_rows), "ppr"
    ).to_numpy(dtype=float)
    curve = RankCurve().fit(train_rows, train_points)
    adp = curve.predict_samples(rows, draws=model.shape[1], seed=seed + 7)

    rank = pd.to_numeric(rows.get("adp_rank"), errors="coerce").to_numpy(float)
    keep = (
        np.isfinite(observed)
        & np.isfinite(model).all(axis=1)
        & np.isfinite(adp).all(axis=1)
    )
    fold = {"tiers": {}}
    for name, low, high in TIERS:
        mask = keep & np.isfinite(rank) & (rank >= low) & (rank <= high)
        if mask.sum() < 25:
            continue
        fold["tiers"][name] = {
            "model": _metrics(observed[mask], model[mask]),
            "adp": _metrics(observed[mask], adp[mask]),
        }
    drafted = keep & np.isfinite(rank) & (rank <= 400)
    fold["rows"] = {
        "observed": observed[drafted].tolist(),
        "model_mean": model[drafted].mean(axis=1).tolist(),
        "adp_mean": adp[drafted].mean(axis=1).tolist(),
        "rank": rank[drafted].tolist(),
    }
    del pipeline, prediction, model, adp
    gc.collect()
    return fold


def _report(report: dict, args) -> int:
    folds = report["folds"]
    holdouts = sorted(int(k) for k in folds)
    print(f"\n{'=' * 78}\nmodel against the draft board, holdouts {holdouts}\n{'=' * 78}")
    print(f"\n{'tier':10} {'n':>5} {'model MAE':>10} {'ADP MAE':>9} {'delta':>8}"
          f" {'model CRPS':>11} {'ADP CRPS':>10} {'delta':>8}")
    print("-" * 78)
    for name, *_ in TIERS:
        blocks = [folds[str(h)]["tiers"][name] for h in holdouts
                  if name in folds[str(h)]["tiers"]]
        if not blocks:
            continue
        mm = np.mean([b["model"]["mae"] for b in blocks])
        am = np.mean([b["adp"]["mae"] for b in blocks])
        mc = np.mean([b["model"]["crps"] for b in blocks])
        ac = np.mean([b["adp"]["crps"] for b in blocks])
        n = int(np.mean([b["model"]["n"] for b in blocks]))
        print(f"{name:10} {n:5d} {mm:10.2f} {am:9.2f} {(mm-am)/am:+8.1%}"
              f" {mc:11.2f} {ac:10.2f} {(mc-ac)/ac:+8.1%}")

    observed = np.concatenate([np.array(folds[str(h)]["rows"]["observed"]) for h in holdouts])
    model = np.concatenate([np.array(folds[str(h)]["rows"]["model_mean"]) for h in holdouts])
    adp = np.concatenate([np.array(folds[str(h)]["rows"]["adp_mean"]) for h in holdouts])
    gap = model - adp
    adp_error = observed - adp

    print(f"\n{'=' * 78}\nis the disagreement information?\n{'=' * 78}")
    from scipy import stats as st
    r, p = st.pearsonr(gap, adp_error)
    print(f"  corr(model - ADP, actual - ADP) = {r:+.3f}  p={p:.3g}  n={len(gap)}")
    print("  Positive means the direction of a disagreement predicts which way")
    print("  the board is wrong -- the model carries information ADP does not.")

    edges = np.quantile(gap, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"\n  {'bucket':22} {'n':>5} {'mean gap':>9} {'ADP proj':>9} "
          f"{'model proj':>11} {'actual':>8} {'model wins':>11}")
    print("  " + "-" * 82)
    labels = ["model much lower", "model lower", "agree", "model higher",
              "model much higher"]
    for i, label in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        m = (gap >= lo) & (gap <= hi) if i == 4 else (gap >= lo) & (gap < hi)
        if m.sum() < 10:
            continue
        wins = (np.abs(observed[m] - model[m]) < np.abs(observed[m] - adp[m])).mean()
        print(f"  {label:22} {int(m.sum()):5d} {gap[m].mean():+9.1f} "
              f"{adp[m].mean():9.1f} {model[m].mean():11.1f} {observed[m].mean():8.1f} "
              f"{wins:11.1%}")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, default=str), "utf-8")
    print(f"\nwrote {args.report_json}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-json", type=Path, default=Path("reports/model_vs_adp.json")
    )
    parser.add_argument("--merge", type=Path, nargs="+", default=None)
    args = parser.parse_args(argv)

    if args.merge:
        folds: dict = {}
        for path in args.merge:
            folds.update(json.loads(path.read_text("utf-8"))["folds"])
        return _report({"folds": folds}, args)

    report: dict[str, object] = {"folds": {}}
    for holdout in args.holdouts:
        report["folds"][str(holdout)] = _run_fold(
            holdout, args.cache_dir,
            draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed,
        )
        print(f"holdout {holdout} done", flush=True)
    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
