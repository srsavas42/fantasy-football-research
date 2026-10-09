"""Does knowing a game was cut short improve the rest-of-season projection?

Walk-forward over the holdout seasons: each is predicted by a model fitted strictly
before it, with the games model given (a) nothing about partial games, (b) the lagged
flag, (c) the flag plus the count of the last three, (d) the flag as a group of its
own. The rate model is the same in every arm, so only expected games differ.

    python scripts/validate_partial_game.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.weekly import ros_reconciled as R
from ffmodel.weekly.features import relevant_population
from ffmodel.weekly.injury_type import attach_injury_type
from ffmodel.weekly.restofseason import TARGET, add_rest_of_season_target

ARMS = {
    "none": dict(features=(), group=False),
    "flag": dict(features=("partial_prev",), group=False),
    "flag+count": dict(features=("partial_prev", "partial_recent"), group=False),
    "group": dict(features=("partial_prev", "partial_recent"), group=True),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path(".cache/weekly_features_2016_2025.pkl"))
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--output", type=Path, default=Path("reports/partial_game.json"))
    args = parser.parse_args(argv)

    frame = pd.read_pickle(args.features)
    if TARGET not in frame.columns:
        frame = add_rest_of_season_target(frame)
    if "mechc_muscle" not in frame.columns:
        frame = attach_injury_type(frame)
    frame = R.add_absence_state(frame)
    frame = frame[np.isfinite(pd.to_numeric(frame[TARGET], errors="coerce"))].reset_index(drop=True)

    parts = []
    for holdout in args.holdouts:
        train = frame[frame["season"] < holdout]
        test = frame[frame["season"] == holdout]
        test = test[relevant_population(test).to_numpy(bool)].copy()
        for name, arm in ARMS.items():
            R.PARTIAL_FEATURES, R.PARTIAL_GROUP = arm["features"], arm["group"]
            predicted = R.ReconciledROS().fit(train).predict(test)
            test[f"T_{name}"] = predicted["total"].to_numpy()
            test[f"E_{name}"] = predicted["expected_games"].to_numpy()
        print(f"  {holdout} done", flush=True)
        parts.append(test)
    R.PARTIAL_FEATURES, R.PARTIAL_GROUP = ARMS["flag+count"]["features"], False

    t = pd.concat(parts, ignore_index=True)
    y = pd.to_numeric(t[TARGET], errors="coerce").to_numpy(float)
    left = pd.to_numeric(t["games_remaining"], errors="coerce").to_numpy(float)
    drafted = t["adp_drafted"].eq(1)
    flagged = t["partial_prev"].eq(1)
    populations = {
        "everyone": pd.Series(True, index=t.index),
        "drafted": drafted,
        "cut short last game": flagged,
        "  drafted": flagged & drafted,
        "  with >=3 games left": flagged & pd.Series(left >= 3, index=t.index),
        "played full last game": ~flagged & t["missed_last"].eq(0),
        "missed last game": t["missed_last"].eq(1),
        "weeks 1-4": t["week"].le(4),
        "weeks 5-9": t["week"].between(5, 9),
        "weeks 10+": t["week"].ge(10),
    }
    report: dict = {"holdouts": args.holdouts, "accuracy": {}, "games": {}}
    names = list(ARMS)
    print(f"\n=== total MAE (bias = actual - projected) ===")
    print(f"  {'':26s}{'n':>6s}" + "".join(f"{n:>17s}" for n in names))
    for label, mask in populations.items():
        m = mask.to_numpy()
        if m.sum() < 25:
            continue
        cells = {n: (float(np.abs(y[m] - t[f"T_{n}"].to_numpy()[m]).mean()),
                     float((y[m] - t[f"T_{n}"].to_numpy()[m]).mean())) for n in names}
        report["accuracy"][label.strip()] = {"n": int(m.sum()), **{n: {"mae": a, "bias": b} for n, (a, b) in cells.items()}}
        print(f"  {label:26s}{int(m.sum()):6d}" + "".join(f"{cells[n][0]:9.2f} ({cells[n][1]:+5.1f})" for n in names))

    print("\n=== share of remaining games played: actual and projected ===")
    gp = pd.to_numeric(t.get("games_played_rest"), errors="coerce") if "games_played_rest" in t else None
    if gp is None:
        from ffmodel.weekly.availability_rate import add_played_rate_target
        base = add_played_rate_target(frame)
        gp = t[["player_key", "season", "week"]].merge(
            base[["player_key", "season", "week", "played_rate_rest"]],
            on=["player_key", "season", "week"], how="left")["played_rate_rest"]
        gp = gp * left
    for label in ("cut short last game", "  with >=3 games left", "played full last game"):
        m = populations[label].to_numpy() & np.isfinite(gp.to_numpy(float))
        if m.sum() < 25:
            continue
        actual = float(np.nansum(gp.to_numpy(float)[m]) / left[m].sum())
        row = {n: float(t[f"E_{n}"].to_numpy()[m].sum() / left[m].sum()) for n in names}
        report["games"][label.strip()] = {"actual": actual, **row}
        print(f"  {label:26s} actual {actual:.3f} | " + "  ".join(f"{n} {row[n]:.3f}" for n in names))

    rng = np.random.default_rng(0)
    print("\n=== paired bootstrap over player-seasons (gain in MAE vs 'none', points) ===")
    for label in ("cut short last game", "  with >=3 games left", "drafted", "everyone"):
        m = populations[label].to_numpy()
        for arm in names[1:]:
            d = pd.DataFrame({"k": t["player_key"].astype(str) + t["season"].astype(str),
                              "g": np.abs(y - t["T_none"].to_numpy()) - np.abs(y - t[f"T_{arm}"].to_numpy())})[m]
            v = d.groupby("k")["g"].mean().to_numpy()
            bs = [rng.choice(v, len(v)).mean() for _ in range(1000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            report.setdefault("paired", {}).setdefault(label.strip(), {})[arm] = {"gain": float(v.mean()), "lo": float(lo), "hi": float(hi)}
            print(f"  {label:26s}{arm:12s} {v.mean():+6.2f}  [{lo:+.2f}, {hi:+.2f}]")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), "utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
