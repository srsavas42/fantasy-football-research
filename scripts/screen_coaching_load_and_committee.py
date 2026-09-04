"""Does a play-caller carry load management or backfield concentration?

Two ideas that survived the first round of triage, screened together because
they share a design with scripts/screen_coaching_tree_transfer.py.

**Load.** Availability is the layer with the worst persistence in this package
-- a starter's prior-to-next availability correlates at only +0.079 -- so there
is more unexplained variance there than anywhere else. If some staffs manage
snaps differently, that would be a rare thing that actually predicts it.

**Committee.** "Is this coach a committee guy, or does he ride a bell cow?" is
probably the single most widely held coaching belief in fantasy football, and
it sits right next to the one shape that *did* transfer (rb_target_share). If
back usage travels, backfield concentration is the obvious neighbour to check.

Shapes, all era-normalised within season per the lesson in
screen_coaching_deep_history.py (league norms drift, so raw averages across
eras add eras rather than tendencies):

  starter_availability  mean availability of skill players holding a real snap
                        share -- the load-management signature
  rb_carry_top_share    the lead back's share of team carries: 1.0 is a bell
                        cow, low is a committee
  rb_carry_hhi          Herfindahl concentration over the whole backfield, which
                        unlike the top share also sees how the rest splits
  top5_snap_rate        mean snap share of the five most-used skill players --
                        how hard this staff rides its starters

Result: all four null, and three of them flat. Backfield concentration is the
emphatic one at +0.003 (p=0.97) -- whatever a coach's committee reputation, it
does not follow him. Read alongside the one hit, the pattern is that a
play-caller carries *how often the backfield is thrown to* and not *which back
gets the work*, which is a personnel decision made with the roster he is given.

    python scripts/screen_coaching_load_and_committee.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ffmodel.data.coaching import load_scheme_lineage  # noqa: E402

METRICS = ("starter_availability", "rb_carry_top_share", "rb_carry_hhi", "top5_snap_rate")
SCHEME_ROLES = "offensive coordinator|head coach|quarterbacks coach"
EXCLUDED_ROLES = "assistant|quality control|intern"
STARTER_SNAP_SHARE = 0.35


def staff_shapes(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows[
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    ].copy()
    for column in ("observed_availability", "snap_share", "rush_att"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

    skill = frame[frame["position"].isin(["RB", "WR", "TE"])]
    starters = skill[skill["snap_share"].fillna(0).ge(STARTER_SNAP_SHARE)]
    shapes = starters.groupby(["season", "team"], as_index=False)[
        "observed_availability"
    ].mean().rename(columns={"observed_availability": "starter_availability"})

    backs = frame[frame["position"].eq("RB")].copy()
    room = backs.groupby(["season", "team"])["rush_att"].transform("sum")
    backs["_share"] = backs["rush_att"] / room.where(room > 0)
    shapes = shapes.merge(
        backs.groupby(["season", "team"], as_index=False)["_share"].max().rename(
            columns={"_share": "rb_carry_top_share"}
        ), on=["season", "team"], how="inner",
    ).merge(
        backs.assign(_sq=backs["_share"] ** 2).groupby(
            ["season", "team"], as_index=False
        )["_sq"].sum().rename(columns={"_sq": "rb_carry_hhi"}),
        on=["season", "team"], how="inner",
    ).merge(
        skill.sort_values("snap_share", ascending=False)
        .groupby(["season", "team"]).head(5)
        .groupby(["season", "team"], as_index=False)["snap_share"].mean()
        .rename(columns={"snap_share": "top5_snap_rate"}),
        on=["season", "team"], how="inner",
    )
    for metric in METRICS:
        by_season = shapes.groupby("season")[metric]
        spread = by_season.transform("std")
        shapes[metric] = (shapes[metric] - by_season.transform("mean")) / spread.where(
            spread > 0
        )
    return shapes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-walkforward"))
    args = parser.parse_args(argv)

    shapes = staff_shapes(pd.read_pickle(args.cache_dir / "player_rows.pkl"))
    lineage = load_scheme_lineage()
    role = lineage["prior_role"].astype(str).str.lower()
    lineage = lineage[
        role.str.contains(SCHEME_ROLES, na=False)
        & ~role.str.contains(EXCLUDED_ROLES, na=False)
    ]
    lineage = lineage[lineage["prior_team_code"] != lineage["franchise_code"]]
    base = shapes.rename(columns={"team": "franchise_code"})

    print("staff load and concentration shapes: does the play-caller carry them?")
    print("(external stops, era-normalised, controls = team seasons -1, -2, -3)\n")
    print(f"{'shape':<26}{'n':>5}{'raw r':>9}{'p':>10}{'partial r':>12}{'p':>10}")
    for metric in METRICS:
        stops = lineage.merge(
            shapes.rename(columns={"season": "prior_season", "team": "prior_team_code"}),
            on=["prior_season", "prior_team_code"], how="inner",
        )
        values = pd.to_numeric(stops[metric], errors="coerce")
        carried = stops[values.notna()].groupby(
            ["season", "franchise_code"], as_index=False
        )[metric].mean().rename(columns={metric: "coach"})
        panel = base[["season", "franchise_code", metric]].merge(
            carried, on=["season", "franchise_code"], how="inner"
        )
        for lag in (1, 2, 3):
            lagged = base[["season", "franchise_code", metric]].copy()
            lagged["season"] = lagged["season"] + lag
            panel = panel.merge(
                lagged.rename(columns={metric: f"t{lag}"}),
                on=["season", "franchise_code"], how="left",
            )
        block = panel[[metric, "coach", "t1", "t2", "t3"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        n = len(block)
        if n < 40:
            print(f"{metric:<26}{n:>5}  too few")
            continue
        y = block[metric].to_numpy(float)
        x = block["coach"].to_numpy(float)
        raw = float(np.corrcoef(x, y)[0, 1])
        p_raw = 2 * stats.norm.sf(abs(np.arctanh(raw) * np.sqrt(max(n - 3, 1))))
        design = np.column_stack([block[["t1", "t2", "t3"]].to_numpy(float), np.ones(n)])
        xr = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        yr = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        part = float(np.corrcoef(xr, yr)[0, 1])
        p_part = 2 * stats.norm.sf(
            abs(np.arctanh(part) * np.sqrt(max(n - 3 - design.shape[1], 1)))
        )
        print(f"{metric:<26}{n:>5}{raw:>+9.4f}{p_raw:>10.2e}{part:>+12.4f}{p_part:>10.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
