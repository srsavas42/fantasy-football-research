"""Does the projected starting quarterback move a running back's volume?

``teammate_qb_quality_signal`` currently reaches three responses, all receiving
efficiency: rec_catch_rate, rec_yards_per_target, rec_td_rate. It touches no
volume model, no opportunity model, and no *rushing* response. So a running
back's carries, his target share and his yards per carry have no quarterback
input at all -- a back whose passer changes is projected as though nothing
happened.

Two mechanisms are worth separating, and they point opposite ways:

  quality   a better passer sustains drives, which means more snaps and more
            plays for everyone behind him, but also a pass-leaning script

  rushing   a quarterback who runs takes carries and short-yardage work off the
            back directly, and checks down less, which costs targets

This screens both against the residual the allocator leaves behind, using the
same deterministic-prior baseline and the same log-prior control as
screen_target_room_quality.py. Efficiency responses are screened against the
player's own lagged rate instead, which is what the efficiency models regress
from.

    python scripts/screen_qb_context_rb.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from screen_target_room_quality import entry_prior, residualise  # noqa: E402

from ffmodel.models.volume_season_average import STREAMS  # noqa: E402


def partial(x: pd.Series, y: np.ndarray, controls: np.ndarray) -> tuple:
    at = x.notna().to_numpy() & np.isfinite(y)
    if at.sum() < 100:
        return int(at.sum()), float("nan"), float("nan")
    xr = residualise(x[at].to_numpy(float), controls[at])
    yr = residualise(y[at], controls[at])
    if xr.std() < 1e-12 or yr.std() < 1e-12:
        return int(at.sum()), float("nan"), float("nan")
    r = float(np.corrcoef(xr, yr)[0, 1])
    n = int(at.sum())
    z = np.arctanh(r) * np.sqrt(max(n - 3 - controls.shape[1], 1))
    return n, r, float(2 * stats.norm.sf(abs(z)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-walkforward")
    )
    args = parser.parse_args(argv)

    rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    rows = rows[
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce")
        .fillna(0).ne(1)
    ].reset_index(drop=True)

    # The quarterback's own rushing load, built from the same frame. This is the
    # "rushing QB" mechanism the target model has no term for: it is a property
    # of the passer, joined onto his teammates' rows.
    qb = rows[rows.position.eq("QB")].copy()
    qb["_carries_per_game"] = pd.to_numeric(qb.rush_att, errors="coerce") / pd.to_numeric(
        qb.team_games, errors="coerce"
    )
    # One passer per team-season: the one who took the most snaps, which is the
    # same choice teammate_qb_quality_signal makes.
    lead = qb.sort_values("offense_snaps", ascending=False).groupby(
        ["season", "team"], as_index=False
    ).first()[["season", "team", "_carries_per_game"]]
    # Lag it: the response season's quarterback rushing is not known in advance,
    # so the honest feature is what that team's passer did the year before.
    lead["season"] = lead["season"] + 1
    rows = rows.merge(lead, on=["season", "team"], how="left")

    backs = rows[rows.position.eq("RB")].reset_index(drop=True)
    print(f"{len(backs)} running-back seasons, {backs.season.min()}-{backs.season.max()}\n")

    candidates = {
        "teammate_qb_quality_signal": "projected starter's prior passing quality",
        "_carries_per_game": "prior-season QB rushing load (carries/team game)",
    }

    for stream, count_col in (("target", "targets"), ("carry", "rush_att")):
        spec = STREAMS[stream]
        block = backs.copy()
        per_snap = pd.to_numeric(block.get(spec["per_snap_role"]), errors="coerce")
        role = pd.to_numeric(block.get(spec["role"]), errors="coerce")
        draft = pd.to_numeric(block.get(spec["draft"]), errors="coerce")
        cold = block["position"].map(spec["fallback"]).astype(float)
        prior = per_snap.where(per_snap.gt(0))
        prior = prior.where(prior.notna(), role.where(role.gt(0)))
        prior = prior.where(prior.notna(), draft.where(draft.gt(0)))
        prior = prior.where(prior.notna(), cold)
        exposure = pd.to_numeric(block.get("snap_share"), errors="coerce")
        block["_score"] = np.clip(prior, 1e-5, 1.0) * exposure

        # Renormalise over the whole roster the allocator sees, not just backs.
        full = rows.copy()
        fps = pd.to_numeric(full.get(spec["per_snap_role"]), errors="coerce")
        fr = pd.to_numeric(full.get(spec["role"]), errors="coerce")
        fd = pd.to_numeric(full.get(spec["draft"]), errors="coerce")
        fc = full["position"].map(spec["fallback"]).astype(float)
        fp = fps.where(fps.gt(0))
        fp = fp.where(fp.notna(), fr.where(fr.gt(0)))
        fp = fp.where(fp.notna(), fd.where(fd.gt(0)))
        fp = fp.where(fp.notna(), fc)
        full["_score"] = np.clip(fp, 1e-5, 1.0) * pd.to_numeric(
            full.get("snap_share"), errors="coerce"
        )
        totals = full.groupby(["season", "team"])["_score"].sum().rename("_team_score")
        block = block.merge(totals, on=["season", "team"], how="left")
        block["_prior_share"] = block["_score"] / block["_team_score"]

        counts = pd.to_numeric(block[count_col], errors="coerce").fillna(0.0)
        team_counts = pd.to_numeric(full[count_col], errors="coerce").fillna(0.0)
        team_totals = team_counts.groupby(
            [full.season, full.team]
        ).sum().rename("_team_count")
        block = block.merge(team_totals, on=["season", "team"], how="left")
        block["_observed_share"] = counts / block["_team_count"].where(
            block["_team_count"] > 0
        )

        keep = block["_observed_share"].gt(0.01) & block["_prior_share"].gt(0.01)
        block = block[keep & np.isfinite(block["_prior_share"])].reset_index(drop=True)
        residual = (
            np.log(block["_observed_share"]) - np.log(block["_prior_share"])
        ).to_numpy(float)
        log_prior = np.log(block["_prior_share"].to_numpy(float))
        controls = np.column_stack(
            [log_prior, log_prior**2, np.ones(len(block))]
        )
        print(f"RB {stream} share -- residual vs deterministic prior allocation "
              f"({len(block)} rows)")
        for name, label in candidates.items():
            n, r, p = partial(pd.to_numeric(block[name], errors="coerce"), residual, controls)
            print(f"  {name:<30} n={n:<5} r={r:+.4f}  p={p:.2e}   {label}")
        print()

    # Rushing efficiency: regressed from the player's own lagged rate, so that
    # is the control, not a share prior.
    print("RB rushing efficiency -- residual vs own lagged rate")
    for response, lagged in (
        ("rush_yards_per_carry", "prior_rush_yards_per_carry"),
        ("rush_td_rate", "prior_rush_td_rate"),
        # Receiving efficiency is where the signal was already promoted, so it
        # is the positive control: if it does not show up for backs here, the
        # screen is measuring nothing rather than finding nothing.
        ("rec_yards_per_target", "prior_rec_yards_per_target"),
    ):
        if response not in backs or lagged not in backs:
            print(f"  {response}: absent from the frame")
            continue
        block = backs.copy()
        exposure = pd.to_numeric(
            block.get("rush_att" if response.startswith("rush") else "targets"),
            errors="coerce",
        ).fillna(0)
        floor = 50 if response.startswith("rush") else 25
        block = block[exposure.ge(floor)].reset_index(drop=True)
        y = pd.to_numeric(block[response], errors="coerce").to_numpy(float)
        lag = pd.to_numeric(block[lagged], errors="coerce")
        at = np.isfinite(y) & lag.notna().to_numpy()
        block, y = block[at].reset_index(drop=True), y[at]
        lag = lag[at].to_numpy(float)
        controls = np.column_stack([lag, np.ones(len(block))])
        print(f"  {response} ({len(block)} rows)")
        for name, label in candidates.items():
            n, r, p = partial(pd.to_numeric(block[name], errors="coerce"), y, controls)
            print(f"    {name:<28} n={n:<5} r={r:+.4f}  p={p:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
