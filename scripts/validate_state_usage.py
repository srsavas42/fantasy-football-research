"""Does usage with the scoreboard taken out improve the weekly projection?

Walk-forward over the holdout seasons, on the configuration ``scripts/project_live.py``
runs, adding one group of game-state features at a time and then all of them:

    neutral       shares of the team's plays when the game was neutral
    clean         shares with garbage time removed
    script        how much of his work came trailing (targets) or leading (carries)
    team_neutral  the team's pass rate when the game was neutral
    last_game     how much of his team's last game was garbage time
    all           every group

    python scripts/validate_state_usage.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.weekly import nextweek
from ffmodel.weekly import ros_reconciled as R
from ffmodel.weekly.features import relevant_population
from ffmodel.weekly.injury_type import attach_injury_type
from ffmodel.weekly.nextweek import Hurdle
from ffmodel.weekly.restofseason import TARGET, add_rest_of_season_target
from ffmodel.weekly.state_usage import (
    CLEAN_FEATURES,
    LAST_GAME_FEATURES,
    NEUTRAL_FEATURES,
    SCRIPT_FEATURES,
    STATE_FEATURES,
    TEAM_STATE_FEATURES,
    attach_state_usage,
)

GROUPS = {
    "base": (),
    "neutral": NEUTRAL_FEATURES,
    "clean": CLEAN_FEATURES,
    "script": SCRIPT_FEATURES,
    "team_neutral": TEAM_STATE_FEATURES,
    "last_game": LAST_GAME_FEATURES,
    "all": STATE_FEATURES,
}


def build(use_state: bool) -> Hurdle:
    return Hurdle(
        use_team=True, use_matchup=True, use_phase=True, use_script=True,
        use_adp=True, use_news=True, use_snaps=True, use_recent=True,
        use_pedigree=True, use_charting=True, use_partial=True, use_state=use_state,
        by_position=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path(".cache/weekly_features_2016_2025.pkl"))
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("reports/state_usage.json"))
    args = parser.parse_args(argv)

    frame = pd.read_pickle(args.features)
    if TARGET not in frame.columns:
        frame = add_rest_of_season_target(frame)
    if "mechc_muscle" not in frame.columns:
        frame = attach_injury_type(frame)
    frame = R.add_absence_state(frame)
    frame = attach_state_usage(frame)

    parts = []
    for holdout in args.holdouts:
        train = frame[frame["season"] < holdout]
        test = frame[frame["season"] == holdout]
        test = test[relevant_population(test).to_numpy(bool)].copy()
        target = train["points"].to_numpy(float)
        for name, columns in GROUPS.items():
            nextweek.STATE_FEATURES = columns
            model = build(bool(columns)).fit(train, target)
            samples = model.predict_samples(test, draws=args.draws, seed=holdout)
            test[f"mean_{name}"] = samples.mean(axis=1)
            test[f"p_{name}"] = model.play_probability(test)
        print(f"  {holdout} done", flush=True)
        parts.append(test)
    nextweek.STATE_FEATURES = STATE_FEATURES

    t = pd.concat(parts, ignore_index=True)
    y = pd.to_numeric(t["points"], errors="coerce").to_numpy(float)
    played = pd.to_numeric(t["played"], errors="coerce").fillna(0).to_numpy(float)
    adp = pd.to_numeric(t["adp_rank"], errors="coerce")
    populations = {
        "everyone": np.ones(len(t), bool),
        "drafted": t["adp_drafted"].eq(1).to_numpy(),
        "played": played == 1,
        "QB": t["position"].eq("QB").to_numpy(),
        "RB": t["position"].eq("RB").to_numpy(),
        "WR": t["position"].eq("WR").to_numpy(),
        "TE": t["position"].eq("TE").to_numpy(),
        "top-50 ADP": (t["adp_drafted"].eq(1) & (adp <= 50)).to_numpy(),
        "weeks 1-4": t["week"].le(4).to_numpy(),
        "weeks 5+": t["week"].ge(5).to_numpy(),
        "trailed last game (team garbage > 25%)": (t["prior_last_game_garbage_frac"] > 0.25).to_numpy(),
    }
    names = list(GROUPS)
    report: dict = {"holdouts": args.holdouts, "mae": {}, "rmse": {}, "paired": {}}
    print("\n=== weekly points, MAE of the mean (bias = actual - projected) ===")
    print(f"  {'':40s}{'n':>7s}" + "".join(f"{n:>14s}" for n in names))
    for label, m in populations.items():
        if m.sum() < 50:
            continue
        cells = {}
        for n in names:
            e = y[m] - t[f"mean_{n}"].to_numpy()[m]
            cells[n] = (float(np.abs(e).mean()), float(e.mean()), float(np.sqrt((e ** 2).mean())))
        report["mae"][label] = {"n": int(m.sum()), **{n: cells[n][0] for n in names}}
        report["rmse"][label] = {"n": int(m.sum()), **{n: cells[n][2] for n in names}}
        print(f"  {label:40s}{int(m.sum()):7d}" + "".join(f"{cells[n][0]:8.4f}({cells[n][1]:+5.2f})"[:14].rjust(14) for n in names))

    rng = np.random.default_rng(0)
    print("\n=== paired bootstrap over player-seasons: gain in MAE vs base, points per game ===")
    key = t["player_key"].astype(str) + t["season"].astype(str)
    for label in ("everyone", "drafted", "RB", "WR", "TE", "QB", "top-50 ADP", "weeks 1-4"):
        m = populations[label]
        for n in names[1:]:
            d = pd.DataFrame({"k": key[m], "g": (np.abs(y - t["mean_base"].to_numpy()) - np.abs(y - t[f"mean_{n}"].to_numpy()))[m]})
            v = d.groupby("k")["g"].mean().to_numpy()
            bs = [rng.choice(v, len(v)).mean() for _ in range(500)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            report["paired"].setdefault(label, {})[n] = {"gain": float(v.mean()), "lo": float(lo), "hi": float(hi)}
            flag = "  *" if lo > 0 else ("  -" if hi < 0 else "")
            print(f"  {label:14s}{n:14s}{v.mean():+8.4f}  [{lo:+.4f}, {hi:+.4f}]{flag}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), "utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
