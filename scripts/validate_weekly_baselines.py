"""How much does the weekly model actually beat "average his last few weeks"?

The ladder in `scripts/validate_weekly.py` already carries two naive rungs -- a
career expanding mean and an exponentially weighted one -- but it has never
answered the question a sceptical reader asks first: *which* naive average, and
would a differently-tuned one have closed the gap on its own? A model that beats
a badly-chosen baseline has proved less than it looks.

So this scores the shipped model against a grid of the heuristics a manager
actually uses:

``last``
    Last week's points. The zero-memory baseline.

``ma{N}``
    A simple moving average over his last N **rostered** weeks, zeros included.
    "He's averaged eight points over the last month."

``ma{N}_played``
    The same window over the weeks he actually played. "He's averaged twelve
    when he suits up." A different and usually higher number, and the one people
    quote.

``ewma{H}``
    Exponentially weighted, half-life H games. The shipped feature layer uses
    H = 1; the rest are here to show whether that choice is load-bearing or
    whether the whole family lands in the same place.

``season``
    Mean of every week so far this season, reset at the season boundary. What a
    league site shows in its standings column.

Every baseline is wrapped in the same `HistoryMean` estimator the ladder already
uses, so each gets an honest predictive *distribution* (residuals resampled from
training rows with a similar fitted value) rather than a bare point estimate.
Scoring a point forecast against a distributional metric would hand the model a
win it did not earn, which is the whole reason this comparison is worth running
carefully.

    python scripts/validate_weekly_baselines.py --holdouts 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from ffmodel.weekly import FEATURES_CACHE
from ffmodel.weekly.evaluate import report, walk_forward
from ffmodel.weekly.features import _prior
from ffmodel.weekly.nextweek import HistoryMean, Hurdle

COLUMNS = ["mae", "rmse", "crps", "bias", "coverage_80", "within_group_spearman"]

# The shipped next-week model, as the ladder configures it.
SHIPPED = dict(
    use_team=True,
    use_matchup=True,
    use_phase=True,
    use_script=True,
    use_adp=True,
    use_news=True,
    use_snaps=True,
    use_recent=True,
    use_pedigree=True,
    use_charting=True,
    by_position=True,
)

WINDOWS = (3, 5, 8)
HALFLIVES = (1.0, 2.0, 4.0, 8.0)


def add_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach every naive average, each lagged so it cannot see its own week."""
    out = frame.sort_values(["player_key", "season", "week"], kind="mergesort").copy()
    points = pd.to_numeric(out["points"], errors="coerce").astype(float)
    played = pd.to_numeric(out["played"], errors="coerce").fillna(0).astype(float)

    # Simple moving averages over the last N rostered weeks. `_prior` lags by one
    # row inside each player, so the window never contains the week being
    # predicted.
    grouped = points.groupby(out["player_key"], sort=False)
    for window in WINDOWS:
        rolled = grouped.transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
        out[f"bl_ma{window}"] = (
            rolled.groupby(out["player_key"], sort=False).shift(1)
        )

    # The same windows over played weeks only -- the number people actually
    # quote, and a different quantity: it answers "what is he worth when he
    # suits up", not "what will he score next week", so it should over-project
    # anyone with real absence risk.
    played_points = points.where(played == 1.0)
    played_group = played_points.groupby(out["player_key"], sort=False)
    for window in WINDOWS:
        rolled = played_group.transform(
            lambda s, w=window: s.rolling(w, min_periods=1).mean()
        )
        out[f"bl_ma{window}_played"] = (
            rolled.groupby(out["player_key"], sort=False).shift(1)
        )

    # Exponentially weighted, several half-lives.
    for halflife in HALFLIVES:
        alpha = 1.0 - 0.5 ** (1.0 / halflife)
        out[f"bl_ewma{halflife:g}"] = _prior(
            out, ["player_key"], points, how="ewm", alpha=alpha
        )

    # Season-to-date, reset at the season boundary.
    out["bl_season"] = _prior(out, ["player_key", "season"], points, how="mean")

    # Last week's points, already in the panel under its own name.
    out["bl_last"] = out["prior_points_last"]
    return out


def ladder() -> list:
    rungs = [
        HistoryMean(column="bl_last", name="last-week"),
        HistoryMean(column="bl_season", name="season-to-date"),
        HistoryMean(column="prior_points_mean", name="career-mean"),
    ]
    rungs += [
        HistoryMean(column=f"bl_ma{w}", name=f"moving-avg-{w}") for w in WINDOWS
    ]
    rungs += [
        HistoryMean(column=f"bl_ma{w}_played", name=f"moving-avg-{w}-played")
        for w in WINDOWS
    ]
    rungs += [
        HistoryMean(column=f"bl_ewma{h:g}", name=f"ewma-hl{h:g}") for h in HALFLIVES
    ]
    rungs.append(Hurdle(name="shipped-model", **SHIPPED))
    return rungs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--draws", type=int, default=600)
    parser.add_argument("--features", type=Path, default=FEATURES_CACHE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.features.exists():
        raise SystemExit(f"no feature cache at {args.features}")
    frame = add_baselines(pd.read_pickle(args.features))
    print(f"panel {len(frame)} rows, seasons {frame.season.min()}-{frame.season.max()}")

    results = walk_forward(
        frame, ladder(), target="points", holdouts=args.holdouts, draws=args.draws
    )

    for population in ("relevant", "panel"):
        table = report(results, population)
        if table.empty:
            continue
        keep = ["estimator", "n"] + [c for c in COLUMNS if c in table.columns]
        table = table[keep].copy()
        best_naive = table[table.estimator != "shipped-model"]["crps"].min()
        shipped = float(table[table.estimator == "shipped-model"]["crps"].iloc[0])
        table["crps_vs_best_naive_%"] = (
            100.0 * (table["crps"] - best_naive) / best_naive
        ).round(2)
        print(f"\n-- {population} --")
        print(table.round(4).to_string(index=False))
        print(
            f"\n   best naive CRPS {best_naive:.4f}; shipped model {shipped:.4f} "
            f"({100.0 * (shipped - best_naive) / best_naive:+.2f}%)"
        )

    rows = []
    for fold in results.get("folds", []):
        entry = {"holdout": fold["holdout"]}
        for name, block in fold.get("estimators", {}).items():
            scored = block.get("relevant")
            if scored is not None:
                entry[name] = round(float(scored["crps"]), 4)
        rows.append(entry)
    if rows:
        print("\n-- CRPS by fold, relevant population --")
        print(pd.DataFrame(rows).to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str), "utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
