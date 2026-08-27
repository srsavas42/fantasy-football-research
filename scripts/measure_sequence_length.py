"""How long is a player's season-level series, and what does that permit?

Any classical time-series method needs a series. The question is whether these
players have one. Per-player ARIMA needs enough observations to identify
`(p, d, q)`; Holt's trend needs two; simple exponential smoothing with a fixed
alpha needs one and degrades gracefully to it.

This counts, for every row the pipeline would predict, how many prior seasons
that player actually has in the frame, and reports what each method's minimum
would admit. Replacement rows are excluded -- they are a synthetic bucket, not
a player with a history.

The answer decides the architecture question rather than informing it: if the
modal player has one prior season, a method that needs five is not a modelling
choice, it is a method that cannot run on most of the board.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# What each candidate needs before it can be fitted per player.
REQUIREMENTS = (
    (1, "simple exponential smoothing / AR(1)"),
    (2, "Holt's linear trend / AR(2)"),
    (5, "per-player ARIMA, minimum identification"),
    (8, "per-player ARIMA, comfortably"),
)


def prior_season_counts(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    replacement = pd.to_numeric(
        out.get("is_replacement_player", pd.Series(0, index=out.index)),
        errors="coerce",
    ).fillna(0)
    out = out[replacement.ne(1)]
    out = out.sort_values(["player_key", "season"])
    out["prior_seasons"] = out.groupby("player_key").cumcount()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player-rows", type=Path, default=Path(".cache/player_rows_2014_2025.pkl")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.player_rows.exists():
        raise SystemExit(f"no frame at {args.player_rows}")
    rows = prior_season_counts(pd.read_pickle(args.player_rows))
    total = len(rows)
    seasons = f"{int(rows.season.min())}-{int(rows.season.max())}"

    print(f"=== Prior seasons available at prediction time ({seasons}, n={total}) ===")
    counts = rows["prior_seasons"].value_counts().sort_index()
    cumulative = 100.0 * counts.cumsum() / total
    for value in counts.index[:9]:
        print(
            f"  {value} prior season(s): {counts[value]:5d} rows "
            f"({100.0 * counts[value] / total:5.1f}%)   cumulative {cumulative[value]:5.1f}%"
        )
    median = float(rows["prior_seasons"].median())
    print(f"  median prior seasons: {median:.0f}")

    print("\n=== What each method's minimum admits ===")
    admitted = {}
    for need, label in REQUIREMENTS:
        share = 100.0 * float((rows["prior_seasons"] >= need).mean())
        admitted[label] = share
        print(f"  {label:<44s} needs {need}:  {share:5.1f}% of rows qualify")

    lengths = rows.groupby("player_key").size()
    print(f"\n=== Total seasons per player in-window (n={len(lengths)} players) ===")
    print(lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).round(2).to_string())

    print(
        "\nReading it: a method whose minimum admits a minority of the board is not\n"
        "a modelling choice. Simple exponential smoothing is the one classical\n"
        "method that survives this, and the package already applies it --\n"
        "features/season_pathways.py, HISTORY_ALPHA = 0.50."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "rows": total,
                    "seasons": seasons,
                    "median_prior_seasons": median,
                    "prior_season_counts": {
                        int(k): int(v) for k, v in counts.items()
                    },
                    "share_admitted": admitted,
                    "player_career_lengths": lengths.describe().to_dict(),
                },
                indent=2,
                default=str,
            ),
            "utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
