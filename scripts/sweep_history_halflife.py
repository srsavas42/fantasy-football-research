"""Pick the decay on player history, without spending the holdouts on it.

``HISTORY_HALFLIFE`` was set at four games a priori and never tuned. That was the
right call at the time — a holdout spent on a nuisance parameter is a holdout
gone — but the role-change finding raises the question directly: the model takes
about two weeks to price a promotion, and the decay is what sets that pace. A
shorter half-life would react faster to a real role change and also chase more
noise, and there is no way to know which dominates without measuring.

The selection is nested so the reported holdouts stay clean. Candidates are
scored on an **inner** validation window — fitted on seasons before it, scored on
seasons that are themselves earlier than any reported holdout — and only the
winner is then run on 2023-2025. Nothing in the outer holdouts influences the
choice, so the confirmation is still out of sample.

A caveat this cannot escape: the inner window is two seasons, so the curve it
traces is noisy. The honest use of the result is to keep four games unless
something clearly beats it, not to take the argmin of a wobbly line.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.weekly.evaluate import score
from ffmodel.weekly.features import add_features, relevant_population
from ffmodel.weekly.market import attach_adp
from ffmodel.weekly.news import add_news_features
from ffmodel.weekly.nextweek import Hurdle

# The first pass ran (2, 3, 4, 6, 8) and came back monotone with the minimum at
# the edge, which is a grid that is too narrow rather than an answer. Extended
# downward until the optimum is interior.
CANDIDATES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def _shipped(use_snaps: bool) -> Hurdle:
    return Hurdle(
        use_team=True, use_matchup=True, use_phase=True, use_script=True,
        use_adp=True, use_news=True, use_snaps=use_snaps, by_position=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inner", type=int, nargs="+", default=[2021, 2022],
        help="validation seasons, all strictly before any reported holdout",
    )
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument(
        "--candidates", type=float, nargs="+", default=list(CANDIDATES)
    )
    parser.add_argument("--no-snaps", action="store_true")
    parser.add_argument(
        "--panel", type=Path, default=PANEL_CACHE
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    panel = attach_adp(pd.read_pickle(args.panel))
    inner = sorted(args.inner)
    if max(inner) >= 2023:
        raise SystemExit(
            "the inner window must sit strictly before the reported holdouts"
        )

    # The evaluation population must not move with the parameter being swept.
    # ``relevant_population`` reads ``prior_points_recent_given_played``, which is
    # itself an exponentially weighted average at the half-life under test, so
    # scoring each candidate on "its own" relevant rows compares different
    # populations: the eight-game arm admitted 10% more rows than the two-game
    # arm, and those extra rows are marginal players who are easier to project.
    # That alone produced a spurious 0.93% win for the longest decay. The
    # population is therefore fixed once, at the a-priori default, and every
    # candidate is scored on exactly those rows.
    reference = add_news_features(add_features(panel, halflife=4.0))
    population = {}
    for holdout in inner:
        block = reference[reference["season"] == holdout]
        population[holdout] = set(
            block.loc[relevant_population(block).to_numpy(bool)]
            .set_index(["player_key", "week"])
            .index
        )

    rows = []
    for halflife in args.candidates:
        frame = add_news_features(add_features(panel, halflife=halflife))
        for holdout in inner:
            train = frame[frame["season"] < holdout]
            test = frame[frame["season"] == holdout]
            if train["season"].nunique() < 2 or test.empty:
                continue
            keys = list(zip(test["player_key"], test["week"]))
            keep = np.array([k in population[holdout] for k in keys])
            test = test[keep]
            model = _shipped(not args.no_snaps).fit(
                train, train["points"].to_numpy(float)
            )
            samples = model.predict_samples(test, draws=args.draws, seed=holdout)
            got = score(
                test["points"].to_numpy(float),
                samples,
                groups=test["position"].astype(str).to_numpy(),
            )
            rows.append({"halflife": halflife, "holdout": holdout, **got})
        print(f"  half-life {halflife:.0f} done", flush=True)

    table = pd.DataFrame(rows)
    pooled = (
        table.assign(
            **{
                c: table[c] * table["n"]
                for c in ("mae", "rmse", "crps", "within_group_spearman")
                if c in table.columns
            }
        )
        .groupby("halflife", as_index=False)
        .sum(numeric_only=True)
    )
    for column in ("mae", "rmse", "crps", "within_group_spearman"):
        if column in pooled.columns:
            pooled[column] = pooled[column] / pooled["n"]

    print(f"\n=== inner validation ({inner}), relevant population ===")
    print(
        pooled[["halflife", "n", "mae", "rmse", "crps", "within_group_spearman"]]
        .round(4)
        .to_string(index=False)
    )
    best = pooled.loc[pooled["crps"].idxmin(), "halflife"]
    default = pooled.loc[pooled["halflife"].eq(4.0)]
    margin = (
        float(default["crps"].iloc[0] - pooled["crps"].min())
        / float(default["crps"].iloc[0])
        if len(default)
        else np.nan
    )
    print(f"\nbest by CRPS: {best:.0f} games ({margin:+.2%} against the a-priori 4)")
    print(
        "Below a quarter of a percent this is a wobble on two seasons, not a "
        "finding;\nkeep four unless the margin is clearly worth the selection."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "inner": inner,
                    "per_fold": table.to_dict("records"),
                    "pooled": pooled.to_dict("records"),
                    "best": float(best),
                    "margin_vs_default": margin,
                },
                indent=2,
                default=str,
            ),
            "utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
