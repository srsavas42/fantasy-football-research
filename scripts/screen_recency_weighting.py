"""Is an exponentially-weighted prior-season role better than the two-block one?

The shipped role prior is a hard two-block split: full-season share weighted
0.65, weeks-10-and-later share weighted 0.35. That is a crude approximation to
"recent weeks matter more" -- week 9 and week 10 are treated as different kinds
of evidence, and week 10 and week 18 as the same kind.

An exponentially-weighted share is the smooth version. Following the career
priors' construction, the player's weekly counts and his team's weekly counts
are accumulated *separately* under the decay and divided at the end, rather than
decaying a weekly share directly -- so a week he played two snaps and a week he
played sixty count for what they are worth instead of equally.

    rho^(last_week - week),  half-life h  =>  rho = 0.5 ** (1 / h)

h -> infinity recovers the flat full-season share, so the sweep contains the
current full-season block as a limit and any gain over it is real.

Scored as out-of-sample R^2 against the *next* season's share, which is the job
this column has. The two-block shipped blend is scored on the same rows as a
reference point, so the comparison is like for like.

Broken out by experience, because the question that prompted this is whether
young players want more recency than veterans.

    python scripts/screen_recency_weighting.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ffmodel.data import load_player_weeks  # noqa: E402
from ffmodel.features import crossseason  # noqa: E402
from ffmodel.features.season_average import normalize_model_positions  # noqa: E402

HALF_LIVES = (2.0, 3.0, 4.0, 6.0, 9.0, 14.0, 25.0, 1e6)
BUCKETS = (("entering year 2", 1, 1), ("year 3-4", 2, 3),
           ("year 5-7", 4, 6), ("veteran (8+)", 7, 99))


def ewma_shares(pw: pd.DataFrame, count_col: str, half_life: float) -> pd.DataFrame:
    """Decayed player counts over decayed team counts, per player-season."""
    d = pw[["season", "week", "team", "player_key", count_col]].copy()
    d[count_col] = pd.to_numeric(d[count_col], errors="coerce").fillna(0.0)
    # Anchor the decay at each season's own last week, so a season that ended at
    # week 17 and one that ended at 18 are weighted on the same footing.
    last = d.groupby("season")["week"].transform("max")
    rho = 0.5 ** (1.0 / float(half_life))
    d["_w"] = rho ** (last - d["week"]).astype(float)
    d["_num"] = d[count_col] * d["_w"]

    team = (
        d.groupby(["season", "week", "team"], dropna=False)[count_col]
        .sum().rename("_team_count").reset_index()
    )
    team = team.merge(
        d[["season", "week", "_w"]].drop_duplicates(["season", "week"]),
        on=["season", "week"], how="left",
    )
    team["_den"] = team["_team_count"] * team["_w"]
    team_totals = team.groupby(["season", "team"], dropna=False)["_den"].sum()

    player = d.groupby(["season", "team", "player_key"], dropna=False)["_num"].sum()
    out = player.reset_index().merge(
        team_totals.rename("_den").reset_index(), on=["season", "team"], how="left"
    )
    out["share"] = out["_num"] / out["_den"].where(out["_den"] > 0)
    return out[["season", "team", "player_key", "share"]]


def r2(y: np.ndarray, x: np.ndarray) -> float:
    """R^2 of the best linear rescaling of x -- scale-free, so half-lives that
    shrink or inflate the share are not penalised for the level."""
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < 50 or x.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-walkforward"))
    parser.add_argument("--count-col", default="targets")
    parser.add_argument("--min-prior-exposure", type=int, default=25)
    parser.add_argument("--seasons", type=int, nargs=2, default=(2015, 2025))
    args = parser.parse_args(argv)

    pw = normalize_model_positions(
        load_player_weeks(range(args.seasons[0], args.seasons[1] + 1), source="nflverse")
    )
    # The weekly frame carries provider ids, not the model's player_key; build
    # it the same way season_average does so the join lands on the same rows.
    pw["player_key"] = crossseason.player_key(pw)
    pw = pw[pw["week"].notna() & pw["team"].notna() & pw["player_key"].notna()]

    rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    rows = rows[
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce")
        .fillna(0).ne(1)
    ].reset_index(drop=True)
    response = "target_share" if args.count_col == "targets" else "carry_share"

    print(f"count={args.count_col}  response={response}  "
          f"prior exposure >= {args.min_prior_exposure}")
    print("R^2 against next season's share; h=1e6 is the flat full-season block\n")

    prior_count = pd.to_numeric(
        rows.groupby("player_key")[args.count_col].shift(1), errors="coerce"
    )
    base = rows[prior_count.ge(args.min_prior_exposure)].copy()
    base["_prior_season"] = base["season"] - 1

    header = f"{'half-life':>10}" + "".join(f"{b[0]:>17}" for b in BUCKETS) + f"{'all':>10}"
    print(header)
    for half_life in HALF_LIVES:
        shares = ewma_shares(pw, args.count_col, half_life).rename(
            columns={"season": "_prior_season", "share": "_ewma"}
        )
        # Join on the player's prior season only; team is deliberately not a key,
        # because a player who changed teams still carries his own prior role.
        merged = base.merge(
            shares.groupby(["_prior_season", "player_key"], as_index=False)["_ewma"].sum(),
            on=["_prior_season", "player_key"], how="left",
        )
        y = pd.to_numeric(merged[response], errors="coerce").to_numpy(float)
        x = pd.to_numeric(merged["_ewma"], errors="coerce").to_numpy(float)
        experience = pd.to_numeric(merged["experience"], errors="coerce").to_numpy(float)
        cells = []
        for _, low, high in BUCKETS:
            at = np.isfinite(experience) & (experience >= low) & (experience <= high)
            cells.append(f"{r2(y[at], x[at]):>17.4f}")
        label = "flat" if half_life > 1e5 else f"{half_life:g}"
        print(f"{label:>10}" + "".join(cells) + f"{r2(y, x):>10.4f}")

    # The shipped two-block blend, on the same rows, as the reference.
    shipped = 0.65 * pd.to_numeric(base["prior_target_share"], errors="coerce") + \
        0.35 * pd.to_numeric(base["prior_late_target_share"], errors="coerce") \
        if args.count_col == "targets" else \
        0.65 * pd.to_numeric(base["prior_carry_share"], errors="coerce") + \
        0.35 * pd.to_numeric(base["prior_late_carry_share"], errors="coerce")
    y = pd.to_numeric(base[response], errors="coerce").to_numpy(float)
    x = shipped.to_numpy(float)
    experience = pd.to_numeric(base["experience"], errors="coerce").to_numpy(float)
    cells = []
    for _, low, high in BUCKETS:
        at = np.isfinite(experience) & (experience >= low) & (experience <= high)
        cells.append(f"{r2(y[at], x[at]):>17.4f}")
    print(f"{'shipped':>10}" + "".join(cells) + f"{r2(y, x):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
