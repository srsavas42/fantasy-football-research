"""Phase 2 feature diagnostics.

Builds features on legacy weekly data and prints distribution summaries plus
the key sanity check: does a player's *trailing* efficiency predict their
*next-week* change in opportunity share? A positive rank correlation supports
the modeling premise that efficient players earn future volume.

Run: python scripts/validate_features.py [start_season end_season]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ffmodel.features import build_features
from ffmodel.features.trailing import player_key


def main(start: int = 2015, end: int = 2020) -> None:
    seasons = list(range(start, end + 1))
    print(f"Building features for {seasons} (legacy source)...")
    df = build_features(seasons, source="legacy")
    print(f"  {len(df):,} player-weeks\n")

    print("Role tier distribution (skill positions):")
    skill = df[df["role_tier"].notna()]
    print(skill.groupby(["position", "role_tier"]).size().unstack(fill_value=0), "\n")

    print("Usage share summary (WR/RB/TE):")
    for pos in ("WR", "RB", "TE"):
        sub = df[(df["position"] == pos) & (df["opportunity"] > 0)]
        print(
            f"  {pos}: target_share med={sub['target_share'].median():.3f}  "
            f"carry_share med={sub['carry_share'].median():.3f}  "
            f"wopr med={sub['wopr'].median():.3f}"
        )
    print()

    print("Primary signal — trailing usage predicts next-week opportunity share:")
    _persistence_signal(df)
    print()
    print("Secondary signal — trailing efficiency vs next-week Δ(opportunity share):")
    print("  (expected weak/noisy at the weekly grain; efficiency acts slowly)")
    _efficiency_signal(df)


def _persistence_signal(df: pd.DataFrame) -> None:
    work = df.copy()
    work["_key"] = player_key(work)
    work = work.sort_values(["_key", "season", "week"])
    work["next_opp_share"] = work.groupby("_key")["opportunity_share"].shift(-1)
    for pos in ("WR", "RB", "TE"):
        sub = work[work["position"] == pos].dropna(
            subset=["ewma_opportunity_share", "next_opp_share"]
        )
        if len(sub) < 50:
            continue
        rho = sub["ewma_opportunity_share"].rank().corr(sub["next_opp_share"].rank())
        print(f"  {pos}: n={len(sub):,}  rho(trailing share -> next share) = {rho:+.3f}")


def _efficiency_signal(df: pd.DataFrame) -> None:
    work = df.copy()
    work["_key"] = player_key(work)
    work = work.sort_values(["_key", "season", "week"])
    # Next week's change in opportunity share for the same player.
    work["next_opp_share"] = work.groupby("_key")["opportunity_share"].shift(-1)
    work["delta_share"] = work["next_opp_share"] - work["opportunity_share"]

    for pos in ("WR", "RB", "TE"):
        sub = work[(work["position"] == pos)].dropna(
            subset=["ewma_yds_per_touch", "delta_share"]
        )
        if len(sub) < 50:
            continue
        # Spearman = Pearson on ranks (avoids a scipy dependency).
        rho = sub["ewma_yds_per_touch"].rank().corr(sub["delta_share"].rank())
        print(f"  {pos}: n={len(sub):,}  rho(yds/touch -> Δshare) = {rho:+.3f}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(int(sys.argv[1]), int(sys.argv[2]))
    else:
        main()
