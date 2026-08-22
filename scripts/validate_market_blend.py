"""Re-score the shipped blend module against the numbers that justified it.

``scripts/blend_with_market.py`` is the analysis that measured the blend; this
is the promoted implementation in ``ffmodel.models.market_blend``. They are
separate code, so "the blend works" is not evidence that *this* code works. The
port is only faithful if it reproduces the measurement.

Weights come from earlier holdouts only, exactly as in the analysis, so 2022 is
unscorable and is reported without a blend rather than being given an in-sample
weight.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.efficiency_posterior import observed_scoring_rows
from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.models.market_blend import MODEL_POSITIONS, MarketBlend, RankCurve, slope_weight
from ffmodel.simulation.scoring import fantasy_points


def curve_training_rows(cache_dir: Path, before: int, scoring: str):
    """Every rostered season-row before ``before``, with its observed points.

    The curve is fitted from the season-average cache rather than from the
    exported holdouts, because the exports cover four seasons and the cache
    covers eleven. Fitting on the exports would throw away most of the history
    a shipped curve should use, and would not reproduce the measurement this
    script exists to check.
    """
    rows = pd.read_pickle(cache_dir / "player_rows.pkl")
    rows = rows[rows.season.lt(before) & rows.position.isin(MODEL_POSITIONS)]
    rows = rows[
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    ].reset_index(drop=True)
    points = np.full(len(rows), np.nan)
    for _, block in rows.groupby("season"):
        points[block.index] = fantasy_points(
            observed_scoring_rows(block.reset_index(drop=True)), scoring
        ).to_numpy()
    keep = np.isfinite(points)
    return rows[keep].reset_index(drop=True), points[keep]


def score(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    mean = samples.mean(axis=1)
    return {
        "n": int(len(observed)),
        "mae": float(np.abs(mean - observed).mean()),
        "rmse": float(np.sqrt(np.mean((mean - observed) ** 2))),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, 0.80)["coverage"]),
        "coverage_95": float(interval_coverage(observed, samples, 0.95)["coverage"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument(
        "--out-dir", type=Path, default=Path(".cache/holdout-predictions")
    )
    parser.add_argument("--label", default="shipping")
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp2")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("scripts/validation_runs/market_blend_shipped.json")
    )
    args = parser.parse_args(argv)

    folds: dict[int, dict] = {}
    for holdout in args.holdouts:
        base = args.out_dir / f"{args.label}_{holdout}"
        rows = pd.read_parquet(base.with_suffix(".rows.parquet"))
        samples = np.load(base.with_suffix(".samples.npz"))["samples"].astype(float)
        named = (
            pd.to_numeric(rows.get("is_replacement_player"), errors="coerce")
            .fillna(0)
            .ne(1)
            .to_numpy()
        )
        drafted = pd.to_numeric(rows["adp_drafted"], errors="coerce").eq(1).to_numpy()
        # A handful of drafted rows each season have no observed stat line --
        # a player who never took a snap. They cannot be scored, and one NaN in
        # a mean or a CRPS turns the whole fold into NaN rather than erroring.
        scorable = rows["observed"].notna().to_numpy()
        keep = named & drafted & scorable
        folds[holdout] = {
            "rows": rows[keep].reset_index(drop=True),
            "model": samples[keep],
            "observed": rows.loc[keep, "observed"].to_numpy(float),
            # The curve for a holdout is fitted on every earlier season's rows,
            # which is what the analysis did and what a shipped model would do.
            "all_rows": rows[named & scorable].reset_index(drop=True),
            "all_observed": rows.loc[named & scorable, "observed"].to_numpy(float),
        }

    report: list[dict] = []
    for index, holdout in enumerate(args.holdouts):
        earlier = args.holdouts[:index]
        fold = folds[holdout]
        entry: dict[str, object] = {"holdout": holdout, "honest": bool(earlier)}
        if not earlier:
            entry["model"] = score(fold["observed"], fold["model"])
            report.append(entry)
            continue

        train_rows, train_points = curve_training_rows(
            args.cache_dir, holdout, args.scoring
        )
        curve = RankCurve().fit(train_rows, train_points)

        # The weight is estimated where both forecasts exist, on earlier folds.
        model_means, curve_means, observed_earlier = [], [], []
        for h in earlier:
            earlier_fold = folds[h]
            earlier_curve = curve.predict_samples(
                earlier_fold["rows"], draws=earlier_fold["model"].shape[1], seed=h
            )
            usable = np.isfinite(earlier_curve).all(axis=1)
            model_means.append(earlier_fold["model"][usable].mean(axis=1))
            curve_means.append(earlier_curve[usable].mean(axis=1))
            observed_earlier.append(earlier_fold["observed"][usable])
        weight = slope_weight(
            np.concatenate(observed_earlier),
            np.concatenate(model_means),
            np.concatenate(curve_means),
        )

        blend = MarketBlend(weight=weight, curve=curve)
        blended = blend.predict_samples(fold["rows"], fold["model"], seed=holdout)
        curve_samples = curve.predict_samples(
            fold["rows"], draws=fold["model"].shape[1], seed=holdout
        )
        usable = np.isfinite(curve_samples).all(axis=1)

        entry["weight"] = weight
        entry["model"] = score(fold["observed"][usable], fold["model"][usable])
        entry["curve"] = score(fold["observed"][usable], curve_samples[usable])
        entry["blend"] = score(fold["observed"][usable], blended[usable])
        report.append(entry)

    print("\nSHIPPED BLEND, drafted pool\n")
    print(
        f"  {'holdout':>7s} {'w':>6s} {'model':>18s} {'curve':>18s} {'blend':>18s}"
    )
    print(f"  {'':>7s} {'':>6s} {'mae/crps':>18s} {'mae/crps':>18s} {'mae/crps':>18s}")
    for entry in report:
        if "blend" not in entry:
            m = entry["model"]
            print(f"  {entry['holdout']:>7d} {'--':>6s} {m['mae']:>8.2f}/{m['crps']:<9.2f}")
            continue
        m, c, b = entry["model"], entry["curve"], entry["blend"]
        print(
            f"  {entry['holdout']:>7d} {entry['weight']:>6.3f} "
            f"{m['mae']:>8.2f}/{m['crps']:<9.2f} "
            f"{c['mae']:>8.2f}/{c['crps']:<9.2f} "
            f"{b['mae']:>8.2f}/{b['crps']:<9.2f}"
        )
    scored = [e for e in report if "blend" in e]
    if scored:
        for metric in ("mae", "crps"):
            board = np.mean([e["curve"][metric] for e in scored])
            blended_mean = np.mean([e["blend"][metric] for e in scored])
            print(f"\n  pooled {metric.upper()} vs board: {blended_mean / board - 1:+.2%}")
        print(
            f"  pooled coverage: 80% {np.mean([e['blend']['coverage_80'] for e in scored]):.3f}"
            f"  95% {np.mean([e['blend']['coverage_95'] for e in scored]):.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
