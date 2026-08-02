"""Volume-v3 walk-forward over cached nflverse rows: holdouts 2022/2023/2024.

Used to validate the S0/S1/S2 fixes and the availability-coupled QB gate; the
results are in docs/volume-fix-validation-2026-08.md and scripts/validation_runs.

Expects nfl_pr.pkl / nfl_tr.pkl built by::

    build_season_average_data(range(2014, 2025), source="nflverse",
                              roster_mode="point_in_time")

Usage: validate_volume_fix_walkforward.py LABEL DRAWS TUNE CHAINS [--cache DIR]
Set FFMODEL_COUPLE_GATE=1 to enable the candidate availability coupling.
"""
import warnings; warnings.filterwarnings("ignore")
import json, os, sys, time
import numpy as np, pandas as pd

from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline
from ffmodel.evaluation.metrics import empirical_crps, interval_coverage

label, draws, tune, chains = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
SCRATCH = sys.argv[6] if len(sys.argv) > 6 and sys.argv[5] == "--cache" else "."

pr = pd.read_pickle(f"{SCRATCH}/nfl_pr.pkl")
tr = pd.read_pickle(f"{SCRATCH}/nfl_tr.pkl")


def dist(observed, samples):
    observed = np.asarray(observed, dtype=float)
    mean = samples.mean(axis=1)
    return {
        "n": int(len(observed)),
        "mae": float(np.abs(observed - mean).mean()),
        "crps": float(empirical_crps(observed, samples).mean()),
        "cov80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
        "cov95": float(interval_coverage(observed, samples, level=0.95)["coverage"]),
    }


out = {}
for holdout in (2022, 2023, 2024):
    started = time.perf_counter()
    train = SeasonAverageData(tr[tr.season < holdout].copy(), pr[pr.season < holdout].copy())
    test = SeasonAverageData(tr[tr.season == holdout].copy(), pr[pr.season == holdout].copy())
    pipeline = SeasonAverageVolumePipeline()
    if os.environ.get('FFMODEL_COUPLE_GATE') == '1':
        pipeline.workload_model.couple_gate_to_availability = True
    pipeline = pipeline.fit(train, draws=draws, tune=tune, chains=chains)
    pred = pipeline.predict_samples(test, seed=42)
    rows = pred.player_rows
    named = pd.to_numeric(
        rows.get("is_replacement_player", pd.Series(0, index=rows.index)), errors="coerce"
    ).fillna(0).ne(1).to_numpy()
    qb = rows["position"].eq("QB").to_numpy()
    games = pd.to_numeric(rows["team_games"], errors="coerce").to_numpy(float)
    snaps_obs = pd.to_numeric(rows["snap_counts_observed"], errors="coerce").fillna(0).gt(0).to_numpy()

    fold = {
        "target": dist(pd.to_numeric(rows["targets"], errors="coerce").to_numpy(float)[named] / games[named],
                       pred.targets_per_team_game[named]),
        "carry": dist(pd.to_numeric(rows["rush_att"], errors="coerce").to_numpy(float)[named] / games[named],
                      pred.carries_per_team_game[named]),
        "pass_qb": dist(pd.to_numeric(rows["pass_att"], errors="coerce").to_numpy(float)[qb & named] / games[qb & named],
                        pred.pass_attempts_per_team_game[qb & named]),
        "snap": dist(pd.to_numeric(rows.loc[snaps_obs & named, "snap_share"], errors="coerce").to_numpy(float),
                     pred.snap_share[snaps_obs & named]),
        "availability": dist(pd.to_numeric(rows.loc[named, "observed_availability"], errors="coerce").to_numpy(float),
                             pred.availability[named]),
    }
    any_carry = pd.to_numeric(rows.loc[named, "rush_att"], errors="coerce").fillna(0).gt(0).to_numpy(float)
    fold["carry_eligibility_brier"] = float(
        np.mean((any_carry - pred.carry_eligibility_probability[named].mean(axis=1)) ** 2)
    )
    # QB workload share, the layer S2 changes most directly.
    wl = rows["observed_qb_workload_share"]
    fold["qb_workload"] = dist(pd.to_numeric(wl, errors="coerce").fillna(0.0).to_numpy(float)[qb & named],
                               pred.qb_workload_share[qb & named])
    fold["seconds"] = round(time.perf_counter() - started, 1)
    fold["diagnostics"] = {
        name: {"max_rhat": float(r["max_rhat"]), "min_ess": float(r["min_bulk_ess"]),
               "divergences": int(r["divergences"])}
        for name, r in pipeline.diagnostics().items()
    }
    out[str(holdout)] = fold
    print(f"[{label}] holdout {holdout} done in {fold['seconds']}s", flush=True)

json.dump(out, open(f"{SCRATCH}/wf_{label}.json", "w"), indent=2, sort_keys=True)
print(f"[{label}] wrote wf_{label}.json")
