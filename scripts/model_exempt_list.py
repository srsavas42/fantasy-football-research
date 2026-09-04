"""Fit the exempt-list duration model and report what it does and does not know.

Prints the episode population it was fitted on, the predictive distribution for
a placement with a given number of games left, a posterior predictive check,
and the sensitivity of the whole thing to the one subjective choice in it --
which placements count as conduct cases at all. That sensitivity is larger than
the posterior width, so a run that reports only the point estimate is
misreporting the model.

    python scripts/model_exempt_list.py --weeks-remaining 18
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.data import ingest
from ffmodel.features.exempt_list import ExemptListModel, exempt_episodes


def posterior_predictive(model: ExemptListModel, episodes: pd.DataFrame, *, draws=2000):
    """Bayesian p-values for summary statistics of the uncensored episodes.

    A statistic the model reproduces lands near 0.5; one near 0 or 1 is a
    feature of the data the model cannot generate.
    """
    observed = episodes.loc[~episodes["censored"].astype(bool), "games_missed"]
    observed = observed.to_numpy(dtype=float)
    if len(observed) < 2:
        return {}
    rng = np.random.default_rng(0)
    cap = int(episodes["weeks_remaining"].max())
    statistics = {
        "mean": np.mean,
        "median": np.median,
        "p25": lambda x: np.quantile(x, 0.25),
        "p75": lambda x: np.quantile(x, 0.75),
        "max": np.max,
    }
    simulated = {name: [] for name in statistics}
    for _ in range(draws):
        hazard = rng.beta(model.posterior_events, model.posterior_survived)
        sample = np.minimum(rng.geometric(hazard, size=len(observed)), cap)
        for name, function in statistics.items():
            simulated[name].append(function(sample))
    return {
        name: {
            "observed": round(float(function(observed)), 2),
            "p": round(float((np.asarray(simulated[name]) >= function(observed)).mean()), 2),
        }
        for name, function in statistics.items()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs=2, default=(2016, 2025))
    parser.add_argument("--weeks-remaining", type=int, default=18)
    parser.add_argument("--min-weeks", type=int, default=2)
    parser.add_argument("--preseason-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/exempt_list.json"))
    args = parser.parse_args(argv)

    seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    rosters = ingest.load_weekly_rosters(seasons)
    episodes = exempt_episodes(rosters, min_weeks=args.min_weeks)
    if args.preseason_only:
        episodes = episodes[episodes["first_week"].eq(1)].reset_index(drop=True)

    print(f"=== episodes, {seasons[0]}-{seasons[-1]}, min_weeks={args.min_weeks} ===")
    print(
        episodes[
            [
                "season", "player_name", "position", "team", "exempt_weeks",
                "first_week", "games_missed", "converted_to_suspension", "censored",
            ]
        ].to_string(index=False)
    )

    model = ExemptListModel().fit(episodes)
    summary = model.summary(weeks_remaining=args.weeks_remaining)
    print(f"\n=== predictive distribution, {args.weeks_remaining} games left ===")
    for key, value in summary.items():
        print(f"  {key:24s} {value}")

    print("\n=== posterior predictive check (uncensored episodes) ===")
    print("  p near 0.5 is good; near 0 or 1 the model cannot generate the data")
    checks = posterior_predictive(model, episodes)
    for name, block in checks.items():
        flag = "   <-- tail" if block["p"] < 0.05 or block["p"] > 0.95 else ""
        print(f"  {name:8s} observed {block['observed']:>6}   p={block['p']:.2f}{flag}")

    print("\n=== sensitivity to the identification filter ===")
    print("  This is the dominant uncertainty. Quote the range, not the point.")
    print(f"  {'min_weeks':>9}  {'n':>3}  {'mean':>5}  {'median':>6}  {'p90':>3}")
    sensitivity = {}
    for minimum in (1, 2, 3):
        alternative = exempt_episodes(rosters, min_weeks=minimum)
        if args.preseason_only:
            alternative = alternative[alternative["first_week"].eq(1)]
        if alternative.empty:
            continue
        block = ExemptListModel().fit(alternative).summary(
            weeks_remaining=args.weeks_remaining
        )
        sensitivity[minimum] = block
        print(
            f"  {minimum:>9}  {block['episodes']:>3}  {block['mean_games_missed']:>5}"
            f"  {block['median_games_missed']:>6}  {block['p90']:>3}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "seasons": seasons,
                "min_weeks": args.min_weeks,
                "preseason_only": bool(args.preseason_only),
                "episodes": episodes.to_dict("records"),
                "summary": summary,
                "posterior_predictive": checks,
                "filter_sensitivity": sensitivity,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
