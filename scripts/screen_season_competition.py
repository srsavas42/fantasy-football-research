"""Is a prior season's usage share worth more when it is competition-weighted?

The weekly arm of this idea is a null (``docs/teammate-competition-2026-09.md``):
the next-week model already carries ``ahead_out`` and its lag, so it repairs the
contaminated average before the prediction leaves the building. The season layer
has no such escape. Its prior-season shares are plain sums over weeks -- see
``ffmodel.features.crossseason.season_usage`` -- and nothing anywhere in the
preseason feature set says which of those weeks were played against a depleted
room. If the effect is real and unhandled anywhere, it is here.

The screen asks the question directly and cheaply, on the weekly panel rather
than through the season pipeline, so a null costs an hour instead of a day.
For each player-season it builds two versions of the same prior-season share --

    raw       sum_i x_i        / sum_i T_i
    adjusted  sum_i w_i x_i    / sum_i w_i T_i

where ``T_i`` is his team's weekly total and ``w_i`` is ``kappa`` in a week
somebody ahead of him was ruled out -- and scores both against what he actually
did the following season. The adjusted form is a weighted average of the same
weekly shares, so it stays a share, and it collapses to the raw one when the
weights are uniform.
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
from ffmodel.weekly.competition import competition_state

SHARES = (
    ("target_share", "targets", "team_targets", ("WR", "TE", "RB")),
    ("carry_share", "rush_att", "team_rush_att", ("RB", "WR")),
)


def season_shares(panel: pd.DataFrame, kappa: float) -> pd.DataFrame:
    """One row per (player, season, main team) with raw and adjusted shares."""
    frame = panel.copy()
    frame["boosted"] = competition_state(frame).to_numpy(float)
    frame["w"] = np.where(frame["boosted"] > 0, float(kappa), 1.0)
    frame = frame[frame["played"].eq(1)]

    # A player traded mid-year is attributed to the club he saw the most work
    # for, matching `crossseason.season_usage`.
    frame["role_volume"] = frame["targets"].fillna(0) + frame["rush_att"].fillna(0)
    main = (
        frame.groupby(["player_key", "season", "team"], dropna=False)["role_volume"]
        .sum().reset_index().sort_values("role_volume")
        .groupby(["player_key", "season"]).tail(1)[["player_key", "season", "team"]]
    )
    frame = frame.merge(main, on=["player_key", "season", "team"], how="inner")

    for name, count, total, _ in SHARES:
        frame[f"_num_{name}"] = frame[count].astype(float)
        frame[f"_den_{name}"] = frame[total].astype(float)
        frame[f"_wnum_{name}"] = frame["w"] * frame[count].astype(float)
        frame[f"_wden_{name}"] = frame["w"] * frame[total].astype(float)

    agg = {"games": ("played", "sum"), "boosted_weeks": ("boosted", "sum"),
           "position": ("position", "first")}
    for name, *_ in SHARES:
        for stem in ("_num_", "_den_", "_wnum_", "_wden_"):
            agg[f"{stem}{name}"] = (f"{stem}{name}", "sum")
    out = frame.groupby(["player_key", "season", "team"], dropna=False).agg(**agg).reset_index()

    for name, *_ in SHARES:
        out[name] = np.divide(out[f"_num_{name}"], out[f"_den_{name}"],
                              out=np.zeros(len(out)), where=out[f"_den_{name}"] > 0)
        out[f"{name}_adj"] = np.divide(out[f"_wnum_{name}"], out[f"_wden_{name}"],
                                       out=np.zeros(len(out)), where=out[f"_wden_{name}"] > 0)
    return out.drop(columns=[c for c in out.columns if c.startswith(("_num_", "_den_", "_wnum_", "_wden_"))])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL_CACHE)
    parser.add_argument("--kappa", type=float, nargs="+", default=[0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--min-games", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("reports/season_competition_screen.json"))
    args = parser.parse_args(argv)

    panel = pd.read_pickle(args.panel)
    rows = []
    for kappa in args.kappa:
        table = season_shares(panel, kappa)
        nxt = table[["player_key", "season", *[n for n, *_ in SHARES]]].copy()
        nxt["season"] = nxt["season"] - 1
        nxt = nxt.rename(columns={n: f"next_{n}" for n, *_ in SHARES})
        joined = table.merge(nxt, on=["player_key", "season"], how="inner")
        joined = joined[joined["games"] >= args.min_games]

        for name, _c, _t, positions in SHARES:
            sub = joined[joined["position"].isin(positions)].dropna(
                subset=[name, f"{name}_adj", f"next_{name}"]
            )
            # Only players who saw both kinds of week can move at all; reporting
            # the pooled number alone would dilute the effect by the ~85% of
            # rows the weights leave exactly where they were.
            moved = sub[(sub[name] - sub[f"{name}_adj"]).abs() > 1e-9]
            for label, block in (("all", sub), ("weights bite", moved)):
                if len(block) < 50:
                    continue
                y = block[f"next_{name}"].to_numpy(float)
                rows.append({
                    "kappa": kappa, "share": name, "population": label, "n": len(block),
                    "r_raw": float(np.corrcoef(block[name], y)[0, 1]),
                    "r_adj": float(np.corrcoef(block[f"{name}_adj"], y)[0, 1]),
                    "mae_raw": float(np.abs(block[name] - y).mean()),
                    "mae_adj": float(np.abs(block[f"{name}_adj"] - y).mean()),
                })
        print(f"  kappa {kappa:.2f} done", flush=True)

    table = pd.DataFrame(rows)
    table["d_r"] = table["r_adj"] - table["r_raw"]
    table["d_mae_pct"] = 100.0 * (table["mae_adj"] - table["mae_raw"]) / table["mae_raw"]
    print("\n=== prior-season share against the following season (negative MAE is better) ===")
    print(table[["kappa", "share", "population", "n", "r_raw", "r_adj", "d_r", "d_mae_pct"]]
          .round(4).to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, default=str), "utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
