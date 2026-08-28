"""Project a week, or the rest of a season, for every player in the panel.

    # start/sit for week 10 of 2025
    python scripts/project_week.py --season 2025 --week 10

    # what everyone is worth from week 10 to the end
    python scripts/project_week.py --season 2025 --week 10 --horizon rest-of-season

Both models are fitted on seasons strictly before ``--season``, so running this
for a past week reproduces what the model would have said at the time rather
than what it knows now.

The output carries the whole predictive distribution as quantiles, not just a
mean. For a start/sit call the mean is usually the wrong summary: two players
with the same projection and different floors are not the same decision, and the
p10 is what separates them.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.models.market_blend import blend_samples
from ffmodel.weekly import FEATURES_CACHE, PANEL_CACHE
from ffmodel.weekly.features import add_features, relevant_population
from ffmodel.weekly.frame import load_panel
from ffmodel.weekly.news import add_news_features
from ffmodel.weekly.pedigree import add_pedigree_features
from ffmodel.weekly.market import (
    WeeklyRankCurve,
    attach_adp,
    bucket_labels,
    fit_blend_weights,
)
from ffmodel.weekly.nextweek import Hurdle
from ffmodel.weekly.restofseason import (
    OFFSET,
    TARGET,
    DirectTotal,
    add_rest_of_season_target,
)

DEFAULT_FEATURES = FEATURES_CACHE


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--horizon", choices=["next-week", "rest-of-season"], default="next-week"
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-blend",
        action="store_true",
        help="rest-of-season only: skip the draft-board blend and use the model alone",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="keep every rostered player, not just the fantasy-relevant ones",
    )
    args = parser.parse_args(argv)

    if args.features.exists():
        frame = pd.read_pickle(args.features)
    else:
        frame = add_pedigree_features(
            add_news_features(add_features(attach_adp(load_panel(range(2016, args.season + 1)))))
        )
    if "inj_status" not in frame.columns:
        frame = add_news_features(frame)
    if "draft_round" not in frame.columns:
        frame = add_pedigree_features(frame)
    frame = add_rest_of_season_target(frame)

    train = frame[frame["season"] < args.season]
    if train.empty:
        raise SystemExit(f"no seasons before {args.season} to fit on")
    rows = frame[(frame["season"] == args.season) & (frame["week"] == args.week)]
    if rows.empty:
        raise SystemExit(f"no panel rows for {args.season} week {args.week}")
    if not args.all_rows:
        rows = rows[relevant_population(rows).to_numpy(bool)]

    weekly_target = train["points"].to_numpy(float)
    seed = args.season * 100 + args.week
    weights = None
    if args.horizon == "next-week":
        model = Hurdle(
            use_team=True,
            use_matchup=True,
            use_phase=True,
            use_script=True,
            use_adp=True,
            use_news=True,
            use_snaps=True,
            use_recent=True,
            use_pedigree=True,
            by_position=True,
        ).fit(train, weekly_target)
        label = "points"
        samples = model.predict_samples(rows, draws=args.draws, seed=seed)
    else:
        # The direct regression, not the forward simulation. Simulating the
        # remaining games from a hierarchical weekly model is the better story
        # and the worse forecast: it lost on CRPS on all three holdouts and its
        # 80% interval covered 0.59 against a nominal 0.80, while this one
        # covered 0.80. See docs/weekly-modeling-2026-08.md.
        def build():
            return DirectTotal(
                use_team=True, use_phase=True, use_adp=True, use_role=True
            )

        model = build().fit(train, train[TARGET].to_numpy(float))
        label = "rest_of_season_points"
        samples = model.predict_samples(rows, draws=args.draws, seed=seed)

        if not args.no_blend:
            # At the draft the model has no in-season information the board
            # lacks, and the board wins. Blending closes that and costs nothing
            # later, because the fitted weight goes to 1.0 once the model has
            # usage the board never saw.
            weights = fit_blend_weights(train, build, TARGET, seed=seed)
            curve = WeeklyRankCurve(per_game=False, offset=OFFSET).fit(
                train, weekly_target
            )
            curve_samples = curve.predict_samples(rows, draws=args.draws, seed=seed)
            labels = bucket_labels(rows["week"].to_numpy(float))
            drafted = pd.to_numeric(rows["adp_drafted"], errors="coerce").eq(1).to_numpy()
            for name, weight in weights.items():
                want = (labels == name) & drafted
                if not want.any():
                    continue
                samples[want] = blend_samples(
                    samples[want], curve_samples[want], weight, seed=seed + 1
                )

    out = pd.DataFrame(
        {
            "player": rows["player_name"].to_numpy(),
            "position": rows["position"].to_numpy(),
            "team": rows["team"].to_numpy(),
            "opponent": rows["opponent"].to_numpy(),
            f"{label}_mean": samples.mean(axis=1),
            "p10": np.quantile(samples, 0.10, axis=1),
            "p50": np.quantile(samples, 0.50, axis=1),
            "p90": np.quantile(samples, 0.90, axis=1),
        }
    )
    if args.horizon == "rest-of-season":
        out["games_left"] = rows[OFFSET].to_numpy()
        if weights:
            print(f"blend weight on the model, by horizon: {weights}")
    else:
        # The two halves of the hurdle are worth seeing apart: a p10 of zero
        # means "he might not play", which is a different call from a low
        # projection for someone certain to suit up.
        out["p_plays"] = model.play_probability(rows)
    out = out.sort_values(f"{label}_mean", ascending=False).reset_index(drop=True)

    print(
        f"{args.horizon}, {args.season} week {args.week}: "
        f"{len(out)} players, fitted on {train.season.min()}-{train.season.max()}"
    )
    print(out.head(args.top).round(2).to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
