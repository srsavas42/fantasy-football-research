"""Does the mean-preserving correction move the carry *bias*, not just MAE?

The MAE/CRPS deltas from the matched walk-forward are -0.21%/-0.03% on carry,
which is below the gate's materiality floor. That does not answer whether the
standing per-position bias -- quarterbacks +26-36%, running backs -5-7% in every
fold -- actually shrank. A correction can trim error metrics while leaving the
split intact.

The flag is read only in ``_role_share_prediction``, never during fit, so one
fit can serve both arms. Predicting twice from the *same* posterior with the
*same* seed makes this an exactly paired test: the innovation draws are
identical and the only difference is the softmax renormalization correction.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

HOLDOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
CACHE = ".cache/ffmodel-wf-2025" if HOLDOUT >= 2025 else ".cache/ffmodel-walkforward"

pr = pd.read_pickle(f"{CACHE}/player_rows.pkl")
tr = pd.read_pickle(f"{CACHE}/team_rows.pkl")
train = SeasonAverageData(
    tr[tr.season < HOLDOUT].copy(), pr[pr.season < HOLDOUT].copy()
)
test = SeasonAverageData(
    tr[tr.season == HOLDOUT].copy(), pr[pr.season == HOLDOUT].copy()
)

pipe = SeasonAverageVolumePipeline().fit(train, draws=1000, tune=1000, chains=4)
print(f"fitted (holdout {HOLDOUT})", flush=True)

arms = {}
for label, flag in (("baseline", False), ("carry_mp", True)):
    pipe.carry_model.mean_preserving_innovation = flag
    arms[label] = pipe.predict_samples(test, seed=42)

rows = arms["baseline"].player_rows
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

# Sanity: the two arms must differ on carries and nowhere else.
same_targets = np.allclose(
    arms["baseline"].targets_per_team_game, arms["carry_mp"].targets_per_team_game
)
moved_carries = float(
    np.abs(
        arms["baseline"].carries_per_team_game - arms["carry_mp"].carries_per_team_game
    ).mean()
)
print(f"\ntargets identical across arms: {same_targets}")
print(f"mean |carry draw shift|: {moved_carries:.5f} per team game")

print(f"\nPER-POSITION CARRY BIAS, holdout {HOLDOUT} (named players only)\n")
header = (
    f"  {'pos':4s} {'n':>4s} {'obs/gm':>8s} "
    f"{'base pred':>10s} {'base bias':>10s} "
    f"{'mp pred':>10s} {'mp bias':>10s} {'change':>9s}"
)
print(header)
summary: dict[str, object] = {"holdout": HOLDOUT, "positions": {}}
for pos in ("QB", "RB", "WR", "TE"):
    mask = named & (position == pos)
    if not mask.sum():
        continue
    obs = observed[mask].mean()
    base = arms["baseline"].carries_per_team_game[mask].mean(axis=1).mean()
    mp = arms["carry_mp"].carries_per_team_game[mask].mean(axis=1).mean()
    base_rel = (base - obs) / max(obs, 1e-9)
    mp_rel = (mp - obs) / max(obs, 1e-9)
    print(
        f"  {pos:4s} {int(mask.sum()):>4d} {obs:>8.3f} "
        f"{base:>10.3f} {base_rel:>+9.1%} {mp:>10.3f} {mp_rel:>+9.1%} "
        f"{mp_rel - base_rel:>+8.1%}"
    )
    summary["positions"][pos] = {
        "n": int(mask.sum()),
        "observed": float(obs),
        "baseline_pred": float(base),
        "baseline_rel_bias": float(base_rel),
        "carry_mp_pred": float(mp),
        "carry_mp_rel_bias": float(mp_rel),
    }

print(f"\nPER-POSITION CARRY ERROR, holdout {HOLDOUT}\n")
print(
    f"  {'pos':4s} {'base MAE':>9s} {'mp MAE':>9s} {'d MAE':>8s} "
    f"{'base CRPS':>10s} {'mp CRPS':>9s} {'d CRPS':>8s}"
)
for pos in ("QB", "RB", "WR", "TE"):
    mask = named & (position == pos)
    if not mask.sum():
        continue
    truth = observed[mask]
    metrics = {}
    for label in ("baseline", "carry_mp"):
        draws = arms[label].carries_per_team_game[mask]
        metrics[label] = (
            float(np.abs(truth - draws.mean(axis=1)).mean()),
            float(empirical_crps(truth, draws).mean()),
        )
    (bm, bc), (mm, mc) = metrics["baseline"], metrics["carry_mp"]
    print(
        f"  {pos:4s} {bm:>9.4f} {mm:>9.4f} {(mm - bm) / bm:>+7.2%} "
        f"{bc:>10.4f} {mc:>9.4f} {(mc - bc) / bc:>+7.2%}"
    )
    summary["positions"].setdefault(pos, {}).update(
        baseline_mae=bm, carry_mp_mae=mm, baseline_crps=bc, carry_mp_crps=mc
    )

# Team-level conservation: the room total is fixed by the count allocator, so a
# room where QBs are over-projected must under-project someone else.
print("\nROOM TOTALS (all rows, per team game)\n")
for label in ("baseline", "carry_mp"):
    total = arms[label].carries_per_team_game.mean(axis=1).sum() / len(
        np.unique(rows["team"])
    )
    print(f"  {label:9s} {total:8.3f}")
print(f"  observed  {np.nansum(observed) / len(np.unique(rows['team'])):8.3f}")

out = Path(f"scripts/validation_runs/carry_bias_{HOLDOUT}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(f"\nwrote {out}")
