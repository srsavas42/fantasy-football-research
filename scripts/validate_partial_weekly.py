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

from ffmodel.weekly import partial_game as PG
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
    base = frame
    frames = {}
    for early in (False, True):
        PG.EARLY_BASELINE = early
        frames[early] = R.add_absence_state(base)
    PG.EARLY_BASELINE = True

    # arm -> (use the flag, weeks 1-2 depth-chart rule on)
    arms = {"base": (False, False), "partial": (True, False), "partial+early": (True, True)}
    parts = []
    for holdout in args.holdouts:
        canon = frames[True]
        test = canon[canon["season"] == holdout]
        test = test[relevant_population(test).to_numpy(bool)].copy()
        for tag, (use, early) in arms.items():
            data = frames[early]
            train = data[data["season"] < holdout]
            this = data[data["season"] == holdout]
            this = this[relevant_population(this).to_numpy(bool)]
            model = build(use).fit(train, train["points"].to_numpy(float))
            samples = model.predict_samples(this, draws=args.draws, seed=holdout)
            test[f"mean_{tag}"] = samples.mean(axis=1)
            test[f"p_{tag}"] = model.play_probability(this)
        print(f"  {holdout} done", flush=True)
        parts.append(test)

    t = pd.concat(parts, ignore_index=True)
    y = pd.to_numeric(t["points"], errors="coerce").to_numpy(float)
    played = pd.to_numeric(t["played"], errors="coerce").fillna(0).to_numpy(float)
    in_season = t["partial_prev"].eq(1).to_numpy()
    early_only = (t["partial_prev_early"].eq(1) & t["partial_prev"].eq(0)).to_numpy()
    flagged = in_season | early_only
    populations = {
        "everyone": np.ones(len(t), bool),
        "drafted": t["adp_drafted"].eq(1).to_numpy(),
        "after a game cut short": flagged,
        "  drafted": flagged & t["adp_drafted"].eq(1).to_numpy(),
        "  weeks 1-2 rule only": early_only,
        "  weeks 1-2 rule only, week 2-3 games": early_only & t["week"].between(2, 3).to_numpy(),
        "played full last game": ~flagged & t["missed_last"].eq(0).to_numpy(),
    }
    report: dict = {"holdouts": args.holdouts, "weekly": {}}
    tags = list(arms)
    print("\n=== weekly points: MAE of the mean (bias) | play probability: Brier | mean predicted vs actual play rate ===")
    print(f"  {'':38s}{'n':>6s}" + "".join(f"{tg:>26s}" for tg in tags))
    for label, m in populations.items():
        if m.sum() < 25:
            continue
        row = {"n": int(m.sum()), "actual_play_rate": float(played[m].mean())}
        cells = []
        for tg in tags:
            e = y[m] - t[f"mean_{tg}"].to_numpy()[m]
            br = float(((played[m] - t[f"p_{tg}"].to_numpy()[m]) ** 2).mean())
            row[tg] = {"mae": float(np.abs(e).mean()), "bias": float(e.mean()), "brier": br,
                       "p_mean": float(t[f"p_{tg}"].to_numpy()[m].mean())}
            cells.append(f"{np.abs(e).mean():6.3f} ({e.mean():+5.2f}) {br:.4f} {t[f'p_{tg}'].to_numpy()[m].mean():.3f}")
        report["weekly"][label.strip()] = row
        print(f"  {label:38s}{int(m.sum()):6d}" + "".join(f"{c:>26s}" for c in cells) + f"   actual {row['actual_play_rate']:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), "utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
