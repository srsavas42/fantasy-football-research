"""Why cold-start rows are under-projected: the claim curve counts draft twice.

measure_cold_start_bias.py established the bias -- cold rows under-projected on
every fold and both streams, established starters over-projected to match, since
shares sum to one inside a team and the two are the same error. This locates it.

Three findings, each narrowing the last.

**It is the role prior, not the snap model.** Rebuilding the deterministic
allocation with each row's *observed* snap share -- perfect foresight on playing
time -- leaves the gap almost untouched: cold rows still land -39.3% on targets
and -26.9% on carries, with warm rows +7.2% and +5.8%. Nothing about projecting
playing time is responsible.

**Conditional on playing, draft capital barely moves per-snap usage.** Among
cold players with 50+ snaps, observed target rate runs 0.118 / 0.089 / 0.085 /
0.066 across round 1, rounds 2-3, rounds 4-7 and undrafted -- a 1.8x spread
end to end. The curve pays 0.149 / 0.074 / 0.025 / 0.009, a 17x spread. Late and
undrafted players who earn a role are paid a seventh of what they produce.

**The reason is a units mismatch, and it is visible in the source.**
ffmodel.features.draft_calibration fits each curve against ``target_share`` /
``carry_share`` -- volume shares, which already contain playing time. But
``_role_prior`` consumes the result as a per-snap rate; its own comment says so
("the cold-start prior stands in for per_snap_role ... so it has to be a
per-snap rate"). The softmax score is ``log(role_prior) + log(exposure)``, so
exposure gets applied twice: once inside a curve fitted on shares, and again as
the offset.

The scale of the double-count, round 1 against undrafted:

    observed snap share (exposure)      7.14x
    observed per-snap target rate       1.79x
    what the claim curve applies       29.96x

The model therefore applies about 214x of draft capital where the data supports
about 13x, and the softmax redistributes what it takes from late-round rows onto
whoever else is in the room -- which is why the same measurement shows
established starters over-projected.

A second, smaller defect sits beside it: ``rookie_seasons`` keeps only rows with
a non-null ``overall_pick``, so undrafted players are excluded from the fit
entirely and then served by extrapolating the exponential to a stand-in pick of
220. That is 61% of cold rows getting a value from beyond the end of the fitted
data, never validated against anything.

    python scripts/diagnose_cold_start_prior.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ffmodel.features.draft import _claim  # noqa: E402
from ffmodel.models.volume_season_average import STREAMS  # noqa: E402

SLOTS = ((1, 32, "round 1"), (33, 100, "rounds 2-3"),
         (101, 200, "rounds 4-7"), (999, 9999, "undrafted"))


def role_prior(rows: pd.DataFrame, stream: str) -> np.ndarray:
    """``_role_prior``'s fallback chain, reproduced."""
    spec = STREAMS[stream]
    weight = 0.75 if stream == "target" else 1.0
    per_snap = pd.to_numeric(rows.get(spec["per_snap_role"]), errors="coerce")
    role = pd.to_numeric(rows.get(spec["role"]), errors="coerce")
    draft = pd.to_numeric(rows.get(spec["draft"]), errors="coerce")
    cold = rows["position"].map(spec["fallback"]).astype(float)
    prior = per_snap.where(per_snap.gt(0))
    both = per_snap.gt(0) & role.gt(0)
    prior.loc[both] = np.exp(
        weight * np.log(per_snap.loc[both]) + (1 - weight) * np.log(role.loc[both])
    )
    prior = prior.where(prior.notna(), role.where(role.gt(0)))
    prior = prior.where(prior.notna(), draft.where(draft.gt(0)))
    prior = prior.where(prior.notna(), cold)
    return np.clip(prior.to_numpy(dtype=float), 1e-5, 1.0)


def slot_mask(picks: pd.Series, low: int, high: int) -> pd.Series:
    return picks.isna() if low >= 999 else picks.between(low, high)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-walkforward"))
    parser.add_argument("--min-snaps", type=int, default=50)
    args = parser.parse_args(argv)

    rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    rows = rows[
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    ].copy()
    rows["_cold"] = pd.to_numeric(rows.get("cold_start"), errors="coerce").fillna(0).eq(1)
    rows["_rookie"] = pd.to_numeric(rows.get("experience"), errors="coerce").fillna(99).le(0)

    print("1. Allocation from the role prior against OBSERVED snap share, so any")
    print("   remaining gap belongs to the prior and not to projecting playing time.\n")
    for stream, count in (("target", "targets"), ("carry", "rush_att")):
        frame = rows[~rows.position.eq("QB")].copy() if stream == "target" else rows.copy()
        frame["_prior"] = role_prior(frame, stream)
        frame["_score"] = frame["_prior"] * pd.to_numeric(
            frame.get("snap_share"), errors="coerce"
        )
        frame = frame[np.isfinite(frame["_score"]) & frame["_score"].gt(0)]
        frame["_alloc"] = frame["_score"] / frame.groupby(["season", "team"])[
            "_score"
        ].transform("sum")
        counts = pd.to_numeric(frame[count], errors="coerce").fillna(0.0)
        totals = counts.groupby([frame.season, frame.team]).transform("sum")
        frame["_observed"] = counts / totals.where(totals > 0)
        frame = frame[frame.season.between(2016, 2025)]
        print(f"   {stream:<8}" + "".join(
            f"  {label}: {100 * (frame[mask]._alloc.sum() / frame[mask]._observed.sum() - 1):+.1f}%"
            for label, mask in (("warm", ~frame._cold), ("cold", frame._cold),
                                ("rookies", frame._rookie))
        ))

    cold = rows[rows._cold & rows.season.between(2016, 2025)].copy()
    cold["_pick"] = pd.to_numeric(cold.get("overall_pick"), errors="coerce")
    snaps = pd.to_numeric(cold.get("offense_snaps"), errors="coerce").fillna(0)

    print(f"\n2. Among cold players with {args.min_snaps}+ snaps: what the curve pays,")
    print("   against what they actually do per snap.\n")
    for stream, count, positions in (("target", "targets", ("WR", "TE", "RB")),
                                     ("carry", "rush_att", ("RB",))):
        block = cold[cold.position.isin(positions) & snaps.ge(args.min_snaps)].copy()
        played = pd.to_numeric(block.offense_snaps, errors="coerce")
        block["_rate"] = pd.to_numeric(block[count], errors="coerce") / played.where(played > 0)
        block["_paid"] = [_claim(p, q, stream) for p, q in zip(block._pick, block.position)]
        block = block[block._rate.notna()]
        print(f"   --- {stream} (n={len(block)}) ---")
        print(f"   {'slot':<14}{'n':>5}{'curve pays':>12}{'observed':>10}{'paid/obs':>10}")
        for low, high, label in SLOTS:
            slot = block[slot_mask(block._pick, low, high)]
            if len(slot) < 10:
                continue
            print(f"   {label:<14}{len(slot):>5}{slot._paid.mean():>12.4f}"
                  f"{slot._rate.mean():>10.4f}{slot._paid.mean() / slot._rate.mean():>10.2f}")
        print()

    print("3. The double-count, round 1 against undrafted.\n")
    first = cold[cold._pick.between(1, 32)]
    undrafted = cold[cold._pick.isna()]
    share = pd.to_numeric(cold.get("snap_share"), errors="coerce")
    ratio = (
        share[first.index].fillna(0).mean() / share[undrafted.index].fillna(0).mean()
    )
    print(f"   observed snap share (exposure) {ratio:>8.2f}x")
    print(f"   what the claim curve applies   "
          f"{_claim(16, 'WR', 'target') / _claim(None, 'WR', 'target'):>8.2f}x")
    print("\n   The softmax multiplies prior by exposure, so draft capital lands twice:")
    print("   once inside a curve fitted on shares, and again as the exposure offset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
