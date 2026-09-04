"""Does the carried backfield tendency sharpen with a deeper career window?

screen_coaching_tree_transfer.py measures the transfer against team shapes
computed from the walk-forward frames, which start in 2015. That silently
discards two thirds of the evidence: of 1,630 external play-calling stops
behind 2016-2025 response seasons, 1,087 are pre-2015 and have no shape to
attach, with a median around 2009. Only 65.8% of response team-seasons get any
usable stop at all.

So the binding limit on that feature was never the coaching data -- it was how
far back the *player* data reached. nflverse player weeks run to 1999 and carry
everything the shape needs (targets, position, team), which is a far lighter
requirement than the full feature frame.

Widening the window should do two things if the effect is real: raise coverage,
and sharpen each coach's carried value by averaging more of his career. A
result that only gains n is a result gaining power; one that gains n *and*
magnitude is one that was being measured with a noisy instrument.

    python scripts/screen_coaching_deep_history.py --from-season 1999
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ffmodel.data import load_player_weeks  # noqa: E402
from ffmodel.data.coaching import load_scheme_lineage  # noqa: E402
from ffmodel.features.season_average import normalize_model_positions  # noqa: E402

METRIC = "rb_target_share"
SCHEME_ROLES = "offensive coordinator|head coach|quarterbacks coach"
EXCLUDED_ROLES = "assistant|quality control|intern"


def deep_team_shapes(first_season: int, last_season: int) -> pd.DataFrame:
    """Running-back share of team targets per team-season, straight from weeks.

    Deliberately not the feature-frame path: this needs three columns, and
    building the full season-average frame back to 1999 would cost far more and
    add nothing the shape uses.
    """
    weeks = normalize_model_positions(
        load_player_weeks(range(first_season, last_season + 1), source="nflverse")
    )
    weeks = weeks[weeks["team"].notna() & weeks["season"].notna()].copy()
    weeks["targets"] = pd.to_numeric(weeks["targets"], errors="coerce").fillna(0.0)
    weeks["_rb"] = weeks["targets"].where(weeks["position"].eq("RB"), 0.0)
    grouped = weeks.groupby(["season", "team"], as_index=False).agg(
        _rb=("_rb", "sum"), _all=("targets", "sum")
    )
    grouped[METRIC] = grouped["_rb"] / grouped["_all"].where(grouped["_all"] > 0)
    grouped["season"] = grouped["season"].astype(int)
    return grouped[["season", "team", METRIC]]


def transfer(shapes: pd.DataFrame, label: str) -> None:
    lineage = load_scheme_lineage()
    role = lineage["prior_role"].astype(str).str.lower()
    lineage = lineage[
        role.str.contains(SCHEME_ROLES, na=False)
        & ~role.str.contains(EXCLUDED_ROLES, na=False)
    ]
    lineage = lineage[lineage["prior_team_code"] != lineage["franchise_code"]]

    stops = lineage.merge(
        shapes.rename(columns={"season": "prior_season", "team": "prior_team_code"}),
        on=["prior_season", "prior_team_code"], how="inner",
    )
    values = pd.to_numeric(stops[METRIC], errors="coerce")
    stops = stops[values.notna()]
    carried = stops.groupby(["season", "franchise_code"], as_index=False).agg(
        coach=(METRIC, "mean"), stops=(METRIC, "size")
    )

    base = shapes.rename(columns={"team": "franchise_code"})
    panel = base.merge(carried, on=["season", "franchise_code"], how="inner")
    for lag in (1, 2, 3):
        lagged = base.copy()
        lagged["season"] = lagged["season"] + lag
        panel = panel.merge(
            lagged.rename(columns={METRIC: f"t{lag}"}), on=["season", "franchise_code"], how="left"
        )
    # Score only the window the original screen scored, so the comparison is
    # about how much history each coach carries, not about which seasons are
    # being predicted.
    panel = panel[panel["season"].between(2016, 2025)]
    block = panel[[METRIC, "coach", "stops", "t1", "t2", "t3"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    n = len(block)
    y = block[METRIC].to_numpy(float)
    x = block["coach"].to_numpy(float)
    design = np.column_stack([block[["t1", "t2", "t3"]].to_numpy(float), np.ones(n)])
    xr = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    yr = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    r = float(np.corrcoef(xr, yr)[0, 1])
    p = 2 * stats.norm.sf(abs(np.arctanh(r) * np.sqrt(max(n - 3 - design.shape[1], 1))))
    print(f"  {label:<34} n={n:<5} stops/coach={block['stops'].median():>4.0f}"
          f"   partial r={r:+.4f}  p={p:.4f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-season", type=int, default=1999)
    parser.add_argument("--last-season", type=int, default=2025)
    args = parser.parse_args(argv)

    print(f"{METRIC} transfer, scored on 2016-2025, by how far back team shapes reach")
    print("(external play-calling stops, controls = team seasons -1, -2, -3)\n")
    for first in (2015, 2010, 2005, args.from_season):
        if first > args.last_season:
            continue
        shapes = deep_team_shapes(first, args.last_season)
        transfer(shapes, f"shapes from {first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
