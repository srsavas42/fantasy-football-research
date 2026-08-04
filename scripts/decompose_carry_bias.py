"""Task 33: which input to the carry allocator carries the QB/RB bias?

The allocator over-projects quarterbacks by a third and under-projects running
backs by a twentieth, on every fold. The mean-preserving test settled that
softmax renormalization owns about one point of the thirty-six, so the cause is
one of the allocator's *inputs*, and there are four:

    carries_i  =  team rush total  x  softmax( log role_prior_i
                                               + log exposure_i
                                               + X_i beta )  x  eligibility_i

This substitutes each input with the truth in turn, one rung at a time, and
reports the per-position bias at each rung. The rung where the quarterback bias
collapses names the culprit; if none does, the role prior is what is left.

A note on why this is a ladder rather than four separate one-at-a-time swaps:
the softmax is a competition, so replacing one position's exposure changes every
other position's share. Cumulative substitution ends at a configuration that is
by construction unbiased, which makes the intermediate rungs readable as "how
much of the gap is closed by knowing this".
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import (
    SeasonAverageVolumePipeline,
    _align_group_draws,
    _allocate_season_counts,
)

HOLDOUT = 2025
OUTPUT = Path(f"scripts/validation_runs/carry_decomposition_{HOLDOUT}.json")
POSITIONS = ("QB", "RB", "WR", "TE")

pr = pd.read_pickle(".cache/ffmodel-wf-2025/player_rows.pkl")
tr = pd.read_pickle(".cache/ffmodel-wf-2025/team_rows.pkl")
train = SeasonAverageData(
    tr[tr.season < HOLDOUT].copy(), pr[pr.season < HOLDOUT].copy()
)
test = SeasonAverageData(
    tr[tr.season == HOLDOUT].copy(), pr[pr.season == HOLDOUT].copy()
)

pipeline = SeasonAverageVolumePipeline().fit(train, draws=1000, tune=1000, chains=4)
print("fitted", flush=True)

served = pipeline.predict_samples(test, seed=42)
rows = served.player_rows.reset_index(drop=True)
draws = served.carries_per_team_game.shape[1]
named = (
    pd.to_numeric(
        rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
        errors="coerce",
    )
    .fillna(0)
    .ne(1)
    .to_numpy()
)
games = pd.to_numeric(rows["team_games"], errors="coerce").to_numpy(float)
observed = (
    pd.to_numeric(rows["rush_att"], errors="coerce").fillna(0).to_numpy(float) / games
)
position = rows["position"].to_numpy()


def column(name: str, fill: float) -> np.ndarray:
    return (
        pd.to_numeric(rows.get(name, pd.Series(np.nan, index=rows.index)), errors="coerce")
        .fillna(fill)
        .to_numpy(float)
    )


# The three truths, broadcast to the posterior's draw axis. Using truth for an
# input means every draw sees the same value: the point of the rung is to remove
# that input's error, and its spread is part of its error.
observed_snap = np.repeat(
    np.clip(column("snap_share", 0.0), 1e-5, 1.0)[:, None], draws, axis=1
)
observed_eligible = np.repeat(
    (column("rush_att", 0.0) > 0).astype(float)[:, None], draws, axis=1
)

# Rebuild the model-side inputs exactly as predict_samples produced them, so a
# rung that substitutes nothing reproduces the served prediction.
availability = pipeline.availability_model.predict_samples(
    test.player_rows, seed=42 + 1
)
team_games = (
    pd.to_numeric(availability.rows["team_games"], errors="coerce")
    .fillna(17)
    .to_numpy(float)
)
model_snap = pipeline.snap_model.predict_samples(
    availability.rows,
    active_fraction_samples=availability.games_active / team_games[:, None],
    seed=42 + 2,
).snap_share
eligibility = pipeline.carry_eligibility_model.predict_samples(
    availability.rows, seed=42 + 5
)
model_eligible = eligibility.eligible
team = pipeline.team_model.predict_average_samples(
    test.team_rows, games=None, seed=42
)


def allocate(snap_samples, eligible_samples, *, observed_totals: bool) -> np.ndarray:
    carry = pipeline.carry_model.predict_share_samples(
        availability.rows,
        snap_samples=snap_samples,
        eligibility_samples=eligible_samples,
        seed=42 + 7,
    )
    group = carry.rows["_group_idx"].to_numpy(dtype=int)
    totals = _align_group_draws(carry.group_keys, team["rows"], team["rush_attempts"])
    team_games = _align_group_draws(carry.group_keys, team["rows"], team["games"])
    if observed_totals:
        indexed = test.team_rows.set_index(["season", "team"])
        wanted = pd.MultiIndex.from_frame(carry.group_keys[["season", "team"]])
        truth = pd.to_numeric(
            indexed["rush_attempts"].reindex(wanted), errors="coerce"
        ).to_numpy(float)
        played = pd.to_numeric(
            indexed["games"].reindex(wanted), errors="coerce"
        ).to_numpy(float)
        totals = np.repeat(np.rint(truth)[:, None].astype(int), draws, axis=1)
        team_games = np.repeat(played[:, None], draws, axis=1)
    counts = _allocate_season_counts(carry, totals, seed=42 + 10)
    return counts / team_games[group]


rungs = [
    ("as served", model_snap, model_eligible, False),
    ("+ observed snap share", observed_snap, model_eligible, False),
    ("+ observed eligibility", observed_snap, observed_eligible, False),
    ("+ observed team total", observed_snap, observed_eligible, True),
]

report: dict[str, object] = {"holdout": HOLDOUT, "rungs": {}}
print(f"\nCARRY BIAS BY SUBSTITUTION RUNG, holdout {HOLDOUT} (named rows)\n")
header = f"  {'rung':24s}" + "".join(f"{pos:>11s}" for pos in POSITIONS) + f"{'room':>9s}"
print(header)
print(
    f"  {'observed carries/gm':24s}"
    + "".join(f"{observed[named & (position == p)].mean():>11.3f}" for p in POSITIONS)
    + f"{observed[named].sum() / rows['team'].nunique():>9.2f}"
)
print("  " + "-" * (24 + 11 * len(POSITIONS) + 9))

for label, snap_samples, eligible_samples, observed_totals in rungs:
    predicted = allocate(snap_samples, eligible_samples, observed_totals=observed_totals)
    mean = predicted.mean(axis=1)
    line = f"  {label:24s}"
    entry: dict[str, float] = {}
    for pos in POSITIONS:
        mask = named & (position == pos)
        truth = observed[mask].mean()
        bias = (mean[mask].mean() - truth) / max(truth, 1e-9)
        entry[pos] = float(bias)
        line += f"{bias:>+10.1%} "
    entry["room"] = float(mean[named].sum() / rows["team"].nunique())
    print(line + f"{entry['room']:>8.2f}", flush=True)
    report["rungs"][label] = entry

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(f"\nwrote {OUTPUT}")

# The role prior, measured the way the allocator uses it: exposure-weighted
# within the room rather than as a median over rows.
print("\nROLE PRIOR vs REALIZED, weighted as the softmax weights it\n")
design = pipeline.carry_model._design(test.player_rows)
d = design["rows"]
role = pipeline.carry_model._role_prior(d)
snaps = pd.to_numeric(d["offense_snaps"], errors="coerce").to_numpy(float)
carries = pd.to_numeric(d["rush_att"], errors="coerce").fillna(0).to_numpy(float)
live = np.isfinite(snaps) & (snaps > 0)
print(f"  {'pos':4s} {'n':>4s} {'prior (snap-wtd)':>17s} {'realized':>10s} {'ratio':>8s}")
prior_summary: dict[str, dict[str, float]] = {}
for pos in POSITIONS:
    mask = live & (d["position"].to_numpy() == pos)
    if not mask.sum():
        continue
    weighted = float(np.average(role[mask], weights=snaps[mask]))
    realized = float(carries[mask].sum() / snaps[mask].sum())
    print(
        f"  {pos:4s} {int(mask.sum()):>4d} {weighted:>17.5f} {realized:>10.5f} "
        f"{weighted / realized:>8.3f}"
    )
    prior_summary[pos] = {
        "prior_snap_weighted": weighted,
        "realized": realized,
        "ratio": weighted / realized,
    }
report["role_prior"] = prior_summary
OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
