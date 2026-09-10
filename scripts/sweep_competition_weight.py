"""How hard to discount a week produced against a depleted room.

``attach_competition`` down-weights, by a factor ``kappa``, every past week in
which the player listed ahead of this one was ruled out. ``kappa`` of one is the
shipped behaviour and zero throws those weeks away entirely; the useful value is
an empirical question and this is where it is asked.

The selection is nested exactly as ``scripts/sweep_history_halflife.py`` does it,
for the same reason: candidates are scored on an inner window that sits strictly
before every reported holdout, so the confirmation run on 2023-2025 is still out
of sample. And the evaluation population is fixed once at the unweighted arm
rather than recomputed per candidate -- ``relevant_population`` reads
``prior_points_recent_given_played``, which is one of the averages under test, so
letting it move would score each candidate on a different set of rows.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.weekly import PANEL_CACHE
from ffmodel.weekly.competition import attach_competition
from ffmodel.weekly.evaluate import score
from ffmodel.weekly.features import add_features, relevant_population
from ffmodel.weekly.market import attach_adp
from ffmodel.weekly.news import add_news_features
from ffmodel.weekly.nextweek import Hurdle

CANDIDATES = (1.0, 0.75, 0.5, 0.25, 0.0)


def _shipped() -> Hurdle:
    return Hurdle(
        use_team=True, use_matchup=True, use_phase=True, use_script=True,
        use_adp=True, use_news=True, use_snaps=True, by_position=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner", type=int, nargs="+", default=[2021, 2022])
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--candidates", type=float, nargs="+", default=list(CANDIDATES))
    parser.add_argument("--panel", type=Path, default=PANEL_CACHE)
    parser.add_argument("--output", type=Path, default=Path("reports/competition_weight_sweep.json"))
    args = parser.parse_args(argv)

    inner = sorted(args.inner)
    if max(inner) >= 2023:
        raise SystemExit("the inner window must sit strictly before the reported holdouts")

    panel = attach_adp(pd.read_pickle(args.panel))

    reference = add_news_features(add_features(attach_competition(panel, kappa=1.0)))
    population = {}
    for holdout in inner:
        block = reference[reference["season"] == holdout]
        population[holdout] = set(
            block.loc[relevant_population(block).to_numpy(bool)]
            .set_index(["player_key", "week"]).index
        )
    del reference

    rows = []
    for kappa in args.candidates:
        frame = add_news_features(add_features(attach_competition(panel, kappa=kappa)))
        for holdout in inner:
            train = frame[frame["season"] < holdout]
            test = frame[frame["season"] == holdout]
            if train["season"].nunique() < 2 or test.empty:
                continue
            keys = list(zip(test["player_key"], test["week"]))
            test = test[np.array([k in population[holdout] for k in keys])]
            model = _shipped().fit(train, train["points"].to_numpy(float))
            samples = model.predict_samples(test, draws=args.draws, seed=holdout)
            got = score(
                test["points"].to_numpy(float), samples,
                groups=test["position"].astype(str).to_numpy(),
            )
            rows.append({"kappa": kappa, "holdout": holdout, **got})
        print(f"  kappa {kappa:.2f} done", flush=True)

    table = pd.DataFrame(rows)
    metrics = [c for c in ("mae", "rmse", "crps", "within_group_spearman") if c in table.columns]
    pooled = (
        table.assign(**{c: table[c] * table["n"] for c in metrics})
        .groupby("kappa", as_index=False).sum(numeric_only=True)
    )
    for column in metrics:
        pooled[column] = pooled[column] / pooled["n"]

    print(f"\n=== inner validation ({inner}), relevant population ===")
    print(pooled[["kappa", "n", *metrics]].round(4).to_string(index=False))

    base = pooled[pooled["kappa"] == 1.0]
    if not base.empty:
        print("\n=== against the unweighted arm (negative is better) ===")
        for _, row in pooled.iterrows():
            deltas = " ".join(
                f"{c} {100.0 * (row[c] - base[c].iloc[0]) / base[c].iloc[0]:+6.2f}%"
                for c in ("mae", "crps") if c in metrics
            )
            print(f"  kappa {row['kappa']:.2f}  {deltas}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"inner": inner, "per_fold": rows, "pooled": pooled.to_dict("records")},
                indent=2, default=str,
            ),
            "utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
