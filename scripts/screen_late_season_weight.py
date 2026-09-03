"""Should the late-season role weight depend on how young the player is?

``prior_target_role`` and its siblings are already a blend:

    0.65 * prior_full_season_share + 0.35 * prior_late_season_share

with "late" meaning weeks >= LATE_SEASON_START_WEEK (10). The weight is a single
constant applied to everyone, and nothing in the repo records it being swept.

The hypothesis is that it should not be a constant. A rookie's full season
averages over the weeks he spent buried on a depth chart, so his late-season
share is the better description of the role he actually holds going into year
two; a nine-year veteran's full season is a clean measurement of a stable role
and its late block is just a noisier subsample of it. If that is right, the
optimal weight falls with experience.

For each experience bucket this solves for the ``w`` minimising squared error of

    w * late + (1 - w) * full   ->   next season's share

in closed form, and reports it with a bootstrap interval. Two guards matter:

* ``late_*_share`` is filled with 0.0 when a player recorded nothing after week
  10, which is not the same as a measured zero. A player hurt from week 9 gets a
  0 that means "absent", and treating it as a role would push the optimal weight
  down. This is 10% of rows, spread evenly across experience buckets (8.3% to
  11.6%) and concentrated in very low-volume seasons -- the median suspicious
  row had 3 to 8 prior-season targets. ``--drop-absent`` removes them so the
  result can be read with and without.
* The comparison is only meaningful where the two predictors disagree, so the
  spread of ``late - full`` is reported per bucket. A bucket where they barely
  differ cannot identify a weight no matter how many rows it has.
* ``--min-prior-exposure`` matters more than it looks. At a floor of 1 the
  population is mostly players with no role in either block, and they carry no
  information about which block describes a role better.

    python scripts/screen_late_season_weight.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BUCKETS = (
    ("entering year 2", 1, 1),
    ("year 3-4", 2, 3),
    ("year 5-7", 4, 6),
    ("veteran (8+)", 7, 99),
)


def optimal_weight(full: np.ndarray, late: np.ndarray, y: np.ndarray) -> float:
    """argmin_w ||(1-w)*full + w*late - y||^2, unclipped so the sign is honest."""
    d = late - full
    denominator = float(d @ d)
    if denominator < 1e-12:
        return float("nan")
    return float(d @ (y - full) / denominator)


def bootstrap(full, late, y, draws: int = 400, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        w = optimal_weight(full[idx], late[idx], y[idx])
        if np.isfinite(w):
            values.append(w)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-walkforward")
    )
    parser.add_argument("--stream", default="target", choices=("target", "carry"))
    parser.add_argument("--min-prior-exposure", type=int, default=1)
    parser.add_argument(
        "--drop-absent",
        action="store_true",
        help="drop rows whose late share is a zero-fill standing in for absence",
    )
    args = parser.parse_args(argv)

    rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    rows = rows[
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce")
        .fillna(0).ne(1)
    ].reset_index(drop=True)

    full_col = f"prior_{args.stream}_share" if args.stream == "carry" else "prior_target_share"
    late_col = f"prior_late_{args.stream}_share" if args.stream == "carry" else "prior_late_target_share"
    response = "target_share" if args.stream == "target" else "carry_share"
    count_col = "targets" if args.stream == "target" else "rush_att"

    # A player's late-season share only exists if his team played late-season
    # games he was on the roster for. The frame fills the rest with 0.0, which
    # reads as "held no role" rather than "was not there".
    prior_count = pd.to_numeric(
        rows.groupby("player_key")[count_col].shift(1), errors="coerce"
    )
    keep = (
        rows[full_col].notna()
        & rows[late_col].notna()
        & rows[response].notna()
        & prior_count.ge(args.min_prior_exposure)
    )
    if args.drop_absent:
        keep = keep & ~(rows[late_col].eq(0) & rows[full_col].gt(0))
    d = rows[keep].reset_index(drop=True)
    full = pd.to_numeric(d[full_col], errors="coerce").to_numpy(float)
    late = pd.to_numeric(d[late_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(d[response], errors="coerce").to_numpy(float)
    experience = pd.to_numeric(d["experience"], errors="coerce").to_numpy(float)

    print(f"stream={args.stream}  {len(d)} player-seasons  "
          f"(prior exposure >= {args.min_prior_exposure}"
          f"{', absence zeros dropped' if args.drop_absent else ''})")
    print(f"shipped weight on the late block: 0.35, applied to everyone\n")
    print(f"{'bucket':<18} {'n':>5} {'optimal w':>10} {'90% interval':>18} "
          f"{'sd(late-full)':>14}")
    for label, low, high in BUCKETS:
        at = np.isfinite(experience) & (experience >= low) & (experience <= high)
        if at.sum() < 100:
            print(f"{label:<18} {int(at.sum()):>5} {'too few':>10}")
            continue
        w = optimal_weight(full[at], late[at], y[at])
        lo, hi = bootstrap(full[at], late[at], y[at])
        print(f"{label:<18} {int(at.sum()):>5} {w:>10.3f} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>18} {np.std(late[at]-full[at]):>14.4f}")

    at = np.isfinite(experience)
    w = optimal_weight(full[at], late[at], y[at])
    lo, hi = bootstrap(full[at], late[at], y[at])
    print(f"\n{'pooled':<18} {int(at.sum()):>5} {w:>10.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
