"""Is the shipped blend weight too timid?

Three independent measurements say the pipeline's disagreement with the board
should be trusted more than 0.316:

    the disagreement slope on the raw pipeline          +0.392
    a sweep of the weight against pooled MAE            optimum 0.40
    the stack's own fitted coefficient on the pipeline  0.31 to 0.44

and a fourth says the same thing from the other side: the *blend's* disagreement
slope is +1.221, meaning the blended forecast is undersized -- what it does say
about a player, it should say about a quarter more strongly.

None of those is a walk-forward. They are pooled fits over the same three
seasons the weight would be scored on, and the sweep in particular chooses its
own optimum on the data it reports. This scores candidate weights per holdout
with full predictive draws, so CRPS is real rather than inferred from means, and
reports fold wins rather than a pooled number a single season could carry.

``accumulating`` is the arm that matches how the shipped weight was actually
derived: for each holdout, the weight is the disagreement slope measured on the
*earlier* holdouts only, so 2023 has no history and falls back to the shipped
constant. It is the honest version of "refit the weight as folds accumulate".

    python scripts/validate_blend_weight.py --holdouts 2023
    python scripts/validate_blend_weight.py --merge a.json b.json c.json
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
from ffmodel.models.market_blend import RankCurve, blend_samples, slope_weight
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.simulation.scoring import fantasy_points

SHIPPED = 0.316
WEIGHTS = (0.0, 0.200, 0.316, 0.400, 0.500, 0.600, 1.0)
TIERS = (("top50", 1, 50), ("51_150", 51, 150), ("151_300", 151, 300), ("drafted", 1, 400))


def _metrics(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.abs(observed - samples.mean(axis=1)).mean()),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
        "n": int(len(observed)),
    }


def _run_fold(holdout: int, cache_dir: Path, *, draws, tune, chains, seed, earlier):
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

    train_rows = train.player_rows.reset_index(drop=True)
    train_points = fantasy_points(
        observed_scoring_rows(train_rows), "ppr"
    ).to_numpy(dtype=float)
    usable = np.isfinite(train_points)
    curve = RankCurve().fit(train_rows[usable], train_points[usable])
    adp = curve.predict_samples(rows, draws=model.shape[1], seed=seed + 7)

    # The weight a deployment would have had at this point in time: the
    # disagreement slope over the holdouts already scored, and the shipped
    # constant when there are none.
    if earlier:
        past_gap = np.concatenate([e["model"] - e["adp"] for e in earlier])
        past_error = np.concatenate([e["observed"] - e["adp"] for e in earlier])
        # slope_weight computes the regression of (observed - curve) on
        # (model - curve); passing the two differences with a zero curve gives
        # exactly that, and keeps the clipping to [0, 1] it already applies.
        accumulated = slope_weight(
            past_error, past_gap, np.zeros_like(past_gap)
        )
    else:
        accumulated = SHIPPED

    rank = pd.to_numeric(rows.get("adp_rank"), errors="coerce").to_numpy(float)
    keep = (
        np.isfinite(observed)
        & np.isfinite(model).all(axis=1)
        & np.isfinite(adp).all(axis=1)
    )
    arms = {f"w{w:.3f}": blend_samples(model, adp, w, seed=seed + 11) for w in WEIGHTS}
    arms["accumulating"] = blend_samples(model, adp, accumulated, seed=seed + 11)

    fold = {"accumulated_weight": float(accumulated), "tiers": {}}
    for name, low, high in TIERS:
        mask = keep & np.isfinite(rank) & (rank >= low) & (rank <= high)
        if mask.sum() < 25:
            continue
        fold["tiers"][name] = {
            arm: _metrics(observed[mask], samples[mask]) for arm, samples in arms.items()
        }
    drafted = keep & np.isfinite(rank) & (rank <= 400)
    fold["rows"] = {
        "observed": observed[drafted].tolist(),
        "model": model[drafted].mean(axis=1).tolist(),
        "adp": adp[drafted].mean(axis=1).tolist(),
    }
    del pipeline, prediction, model, adp, arms
    gc.collect()
    return fold


def _report(report: dict, args) -> int:
    folds = report["folds"]
    holdouts = sorted(int(k) for k in folds)
    arms = [f"w{w:.3f}" for w in WEIGHTS] + ["accumulating"]
    print(f"\n{'=' * 92}\nblend weight, holdouts {holdouts}\n{'=' * 92}")
    print("  weights the accumulating arm chose: "
          + ", ".join(f"{h}={folds[str(h)]['accumulated_weight']:.3f}" for h in holdouts))
    for metric in ("mae", "crps"):
        print(f"\n  {metric.upper()}  (macro: each season weighted equally)")
        print(f"  {'tier':12} " + "".join(f"{a:>13}" for a in arms))
        print("  " + "-" * (12 + 13 * len(arms)))
        for name, *_ in TIERS:
            blocks = [folds[str(h)]["tiers"][name] for h in holdouts
                      if name in folds[str(h)]["tiers"]]
            if not blocks:
                continue
            cells = []
            for arm in arms:
                cells.append(f"{np.mean([b[arm][metric] for b in blocks]):13.2f}")
            print(f"  {name:12} " + "".join(cells))
    print("\n  fold wins against the shipped w=0.316, on drafted CRPS")
    base = [folds[str(h)]["tiers"]["drafted"][f"w{SHIPPED:.3f}"]["crps"] for h in holdouts]
    for arm in arms:
        got = [folds[str(h)]["tiers"]["drafted"][arm]["crps"] for h in holdouts]
        wins = sum(g < b for g, b in zip(got, base))
        delta = (np.mean(got) - np.mean(base)) / np.mean(base)
        print(f"    {arm:14} {wins}/{len(base)}   {delta:+7.2%}")
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
        "--report-json", type=Path, default=Path("reports/blend_weight.json")
    )
    parser.add_argument("--merge", type=Path, nargs="+", default=None)
    args = parser.parse_args(argv)

    if args.merge:
        folds: dict = {}
        for path in args.merge:
            folds.update(json.loads(path.read_text("utf-8"))["folds"])
        return _report({"folds": folds}, args)

    report: dict[str, object] = {"folds": {}}
    earlier: list[dict] = []
    for holdout in sorted(args.holdouts):
        fold = _run_fold(
            holdout, args.cache_dir, draws=args.draws, tune=args.tune,
            chains=args.chains, seed=args.seed, earlier=earlier,
        )
        report["folds"][str(holdout)] = fold
        earlier.append({k: np.array(v, float) for k, v in fold["rows"].items()})
        print(f"holdout {holdout} done (weight {fold['accumulated_weight']:.3f})", flush=True)
    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
