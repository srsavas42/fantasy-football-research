"""Does the previous game being cut short improve next week's start/sit projection?

The weekly hurdle with and without ``use_partial``, walk-forward over the holdout
seasons, on the configuration ``scripts/project_live.py`` runs. Reports the mean's MAE
and the play probability's Brier score, overall and on the games that follow a game cut
short.

    python scripts/validate_partial_weekly.py
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
from ffmodel.weekly.nextweek import Hurdle
from ffmodel.weekly.restofseason import TARGET, add_rest_of_season_target


def build(use_partial: bool) -> Hurdle:
    return Hurdle(
        use_team=True, use_matchup=True, use_phase=True, use_script=True,
        use_adp=True, use_news=True, use_snaps=True, use_recent=True,
        use_pedigree=True, use_charting=True, by_position=True, use_partial=use_partial,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path(".cache/weekly_features_2016_2025.pkl"))
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("reports/partial_weekly.json"))
    args = parser.parse_args(argv)

    frame = pd.read_pickle(args.features)
    if TARGET not in frame.columns:
        frame = add_rest_of_season_target(frame)
    if "mechc_muscle" not in frame.columns:
        frame = attach_injury_type(frame)
    frame = R.add_absence_state(frame)

    parts = []
    for holdout in args.holdouts:
        train = frame[frame["season"] < holdout]
        test = frame[frame["season"] == holdout]
        test = test[relevant_population(test).to_numpy(bool)].copy()
        target = train["points"].to_numpy(float)
        for use in (False, True):
            tag = "partial" if use else "base"
            model = build(use).fit(train, target)
            samples = model.predict_samples(test, draws=args.draws, seed=holdout)
            test[f"mean_{tag}"] = samples.mean(axis=1)
            test[f"p_{tag}"] = model.play_probability(test)
        print(f"  {holdout} done", flush=True)
        parts.append(test)

    t = pd.concat(parts, ignore_index=True)
    y = pd.to_numeric(t["points"], errors="coerce").to_numpy(float)
    played = pd.to_numeric(t["played"], errors="coerce").fillna(0).to_numpy(float)
    flagged = t["partial_prev"].eq(1).to_numpy()
    populations = {
        "everyone": np.ones(len(t), bool),
        "drafted": t["adp_drafted"].eq(1).to_numpy(),
        "after a game cut short": flagged,
        "  drafted": flagged & t["adp_drafted"].eq(1).to_numpy(),
        "played full last game": ~flagged & t["missed_last"].eq(0).to_numpy(),
    }
    report: dict = {"holdouts": args.holdouts, "weekly": {}}
    print("\n=== weekly points: MAE of the mean (bias = actual - projected) | play probability: Brier ===")
    print(f"  {'':26s}{'n':>7s}{'base MAE':>18s}{'partial MAE':>18s}{'base Brier':>12s}{'part Brier':>12s}  actual plays / predicted base / partial")
    for label, m in populations.items():
        if m.sum() < 25:
            continue
        row = {}
        for tag in ("base", "partial"):
            e = y[m] - t[f"mean_{tag}"].to_numpy()[m]
            row[tag] = {"mae": float(np.abs(e).mean()), "bias": float(e.mean()),
                        "brier": float(((played[m] - t[f"p_{tag}"].to_numpy()[m]) ** 2).mean()),
                        "p_mean": float(t[f"p_{tag}"].to_numpy()[m].mean())}
        row["actual_play_rate"] = float(played[m].mean())
        report["weekly"][label.strip()] = {"n": int(m.sum()), **row}
        b, p = row["base"], row["partial"]
        print(f"  {label:26s}{int(m.sum()):7d}  {b['mae']:7.3f} ({b['bias']:+6.2f})  {p['mae']:7.3f} ({p['bias']:+6.2f})"
              f"{b['brier']:12.4f}{p['brier']:12.4f}   {row['actual_play_rate']:.3f} / {b['p_mean']:.3f} / {p['p_mean']:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), "utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
