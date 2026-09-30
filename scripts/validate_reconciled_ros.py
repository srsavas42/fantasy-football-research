"""Does the reconciled rest-of-season total beat the one it replaces, out of sample?

    python scripts/validate_reconciled_ros.py

For each holdout season the reconciled model (expected games x points per game
played, ``ffmodel.weekly.ros_reconciled``) is fitted on earlier seasons only, its
intervals measured on the two seasons before that, and its projection scored on the
holdout. Two references are scored on the same rows:

``shipped``
    The direct regression blended toward the draft-board curve -- what the
    rest-of-season file carried before the reconciliation.
``direct``
    The same regression without the blend.

Accuracy is MAE and bias (actual minus projected, so positive means the projection
was too low) on the total, by the populations the decision is about. Calibration is
the share of outcomes below each quantile: p10, p50 and p90 should leave 10%, 50%
and 90% below. Totals have an atom at zero -- a season that ends -- so the lower tail
is reported strictly below the quantile; "at or below" would count those zeros as
under it and flatter nothing.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.models.market_blend import blend_samples
from ffmodel.weekly.features import relevant_population
from ffmodel.weekly.injury_type import attach_injury_type
from ffmodel.weekly.market import WeeklyRankCurve, bucket_labels, fit_blend_weights
from ffmodel.weekly.restofseason import OFFSET, TARGET, DirectTotal, add_rest_of_season_target
from ffmodel.weekly.ros_reconciled import QUANTILE_LEVELS, ReconciledROS, add_absence_state

DRAWS = 400


def _grid(samples: np.ndarray) -> np.ndarray:
    return np.quantile(samples, QUANTILE_LEVELS, axis=1).T


def _pinball_crps(grid: np.ndarray, actual: np.ndarray) -> float:
    err = actual[:, None] - grid
    q = QUANTILE_LEVELS[None, :]
    return float(2.0 * np.maximum(q * err, (q - 1.0) * err).mean())


def _coverage(grid: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    lo, mid, hi = grid[:, 9], grid[:, 49], grid[:, 89]
    return {
        "below_p10": float(np.mean(actual < lo)),
        "below_p50": float(np.mean(actual < mid)),
        "at_or_below_p90": float(np.mean(actual <= hi)),
        "inside_80": float(np.mean((actual >= lo) & (actual <= hi))),
        "crps": _pinball_crps(grid, actual),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path(".cache/weekly_features_2016_2025.pkl"))
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--output", type=Path, default=Path("reports/reconciled_ros.json"))
    args = parser.parse_args(argv)

    frame = add_absence_state(attach_injury_type(add_rest_of_season_target(pd.read_pickle(args.features))))
    frame = frame[np.isfinite(pd.to_numeric(frame[TARGET], errors="coerce"))].reset_index(drop=True)

    def direct_total():
        return DirectTotal(use_team=True, use_phase=True, use_adp=True, use_role=True)

    parts = []
    for holdout in args.holdouts:
        train = frame[frame["season"] < holdout]
        test = frame[frame["season"] == holdout]
        test = test[relevant_population(test).to_numpy(bool)].copy()

        model = ReconciledROS().fit(train).fit_intervals(train)
        predicted = model.predict(test)
        test["reconciled"] = predicted["total"].to_numpy()
        test["expected_games"] = predicted["expected_games"].to_numpy()
        grid = model.quantile_grid(test, test["reconciled"].to_numpy())

        direct = direct_total().fit(train, train[TARGET].to_numpy(float))
        d = direct.predict_samples(test, draws=DRAWS, seed=holdout)
        weights = fit_blend_weights(train, direct_total, TARGET, seed=holdout)
        curve = WeeklyRankCurve(per_game=False, offset=OFFSET).fit(train, train["points"].to_numpy(float))
        c = curve.predict_samples(test, draws=DRAWS, seed=holdout)
        shipped = d.copy()
        labels = bucket_labels(test["week"].to_numpy(float))
        drafted = test["adp_drafted"].eq(1).to_numpy()
        for name, weight in weights.items():
            hit = (labels == name) & drafted
            if hit.any():
                shipped[hit] = blend_samples(d[hit], c[hit], weight, seed=holdout + 1)
        test["shipped"], test["direct"] = shipped.mean(axis=1), d.mean(axis=1)
        parts.append((test, grid, _grid(shipped)))
        print(f"  {holdout} done", flush=True)

    t = pd.concat([p[0] for p in parts], ignore_index=True)
    reconciled_grid = np.vstack([p[1] for p in parts])
    shipped_grid = np.vstack([p[2] for p in parts])
    y = pd.to_numeric(t[TARGET], errors="coerce").to_numpy(float)
    adp = pd.to_numeric(t["adp_rank"], errors="coerce")
    top50 = t["adp_drafted"].eq(1) & (adp <= 50)
    populations = {
        "everyone": pd.Series(True, index=t.index),
        "drafted": t["adp_drafted"].eq(1),
        "missed last game": t["missed_last"].eq(1),
        "drafted, missed last": t["adp_drafted"].eq(1) & t["missed_last"].eq(1),
        "top-50, missed last": top50 & t["missed_last"].eq(1),
        "reserve list last game": t["res_prev"].eq(1),
        "designated Out last game": t["out_prev"].eq(1),
        "top-50, Out/Q/D last game": top50 & ((t["out_prev"] == 1) | (t["qd_prev"] == 1)),
        # The draft board knows things a model with no in-season history does not,
        # which is why the shipped total blends toward it early. The pooled numbers
        # above are dominated by later weeks, so the early ones are read on their own.
        "weeks 1-4": t["week"].le(4),
        "weeks 1-4, drafted": t["week"].le(4) & t["adp_drafted"].eq(1),
        "weeks 1-4, top-50": t["week"].le(4) & top50,
        **{f"week {w}": t["week"].eq(w) for w in range(1, 7)},
        "weeks 5-9": t["week"].between(5, 9),
        "weeks 10+": t["week"].ge(10),
    }
    report: dict = {"holdouts": args.holdouts, "accuracy": {}, "coverage": {}}
    print("\n=== total: MAE (bias = actual - projected) ===")
    print(f"  {'':28s}{'n':>6s} | {'shipped':>16s} | {'direct':>16s} | {'reconciled':>16s}")
    for name, mask in populations.items():
        m = mask.to_numpy()
        if m.sum() < 25:
            continue
        cells = {}
        for col in ("shipped", "direct", "reconciled"):
            e = y[m] - t.loc[m, col].to_numpy()
            cells[col] = {"mae": float(np.abs(e).mean()), "bias": float(e.mean())}
        report["accuracy"][name] = {"n": int(m.sum()), **cells}
        fmt = lambda c: f"{cells[c]['mae']:6.1f} ({cells[c]['bias']:+6.1f})"
        print(f"  {name:28s}{int(m.sum()):6d} | {fmt('shipped')} | {fmt('direct')} | {fmt('reconciled')}")

    print("\n=== percentile calibration of the total (nominal: <p10 10%, <p50 50%, <=p90 90%, inside 80%) ===")
    print(f"  {'':28s}{'<p10':>7s}{'<p50':>7s}{'<=p90':>7s}{'in80':>7s}{'CRPS':>8s}")
    for label, grid in (("shipped", shipped_grid), ("reconciled", reconciled_grid)):
        print(f"  {label}")
        for name, mask in populations.items():
            m = mask.to_numpy()
            if m.sum() < 25:
                continue
            c = _coverage(grid[m], y[m])
            report["coverage"].setdefault(name, {})[label] = c
            print(f"    {name:26s}{100*c['below_p10']:6.1f}%{100*c['below_p50']:6.1f}%"
                  f"{100*c['at_or_below_p90']:6.1f}%{100*c['inside_80']:6.1f}%{c['crps']:8.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), "utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
