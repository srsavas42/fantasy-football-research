"""Do the room-quality features carry target-share signal the allocator misses?

The softmax score is ``log(role_prior) + log(snap_share) + X.beta + innovation``.
Nothing in ``X`` says who the best receiver in the room is, so when a room loses
its alpha the vacated share is redistributed in proportion to the survivors'
own priors -- there is no term that lets the best remaining option take more
than its proportional cut. ``prior_rec_room_quality_advantage`` is exactly that
missing term: a player's receiving quality minus the role-weighted quality of
the *rest of his current room*, so it moves when a teammate leaves even though
none of the player's own history changed.

This screens it before spending a walk-forward. The baseline is the
deterministic prior allocation -- role prior times projected exposure,
renormalised over the roster -- which is what the softmax would return with no
covariates and no noise. Regress the residual log-ratio of observed to prior
share on each candidate, controlling for position, and report the partial
correlation. A feature with nothing here cannot help the fitted model either.

The control block includes ``log(prior_share)`` as well as position, and it is
not optional. The residual is a log ratio, so a player with a small prior share
has far more room above than below and shows a positive residual on average --
pure mean reversion. Several candidates here are monotone functions of that
same prior share (``role_uncertainty`` is literally ``1 - room_share``), so
without the control they inherit the artifact and report it as signal.

Deliberately *not* a claim about the fitted pipeline: X.beta and the innovation
both move shares, so a signal that survives here still has to clear the
walk-forward against a model that may already capture it another way.

    python scripts/screen_target_room_quality.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ffmodel.models.volume_season_average import STREAMS  # noqa: E402

CANDIDATES = (
    "prior_rec_room_quality_advantage",
    "prior_role_room_quality_advantage",
    "prior_target_role_uncertainty",
    "prior_target_room_competition",
    "prior_target_team_competition",
)
PER_SNAP_WEIGHT = 0.75  # SeasonRosterShareModel default for the target stream


def entry_prior(rows: pd.DataFrame) -> np.ndarray:
    """``_role_prior``'s fallback chain, reproduced on a frame."""
    spec = STREAMS["target"]
    per_snap = pd.to_numeric(rows.get(spec["per_snap_role"]), errors="coerce")
    role = pd.to_numeric(rows.get(spec["role"]), errors="coerce")
    draft = pd.to_numeric(rows.get(spec["draft"]), errors="coerce")
    cold = rows["position"].map(spec["fallback"]).astype(float)

    valid_per_snap, valid_role = per_snap.gt(0), role.gt(0)
    prior = per_snap.where(valid_per_snap)
    both = valid_per_snap & valid_role
    prior.loc[both] = np.exp(
        PER_SNAP_WEIGHT * np.log(per_snap.loc[both])
        + (1.0 - PER_SNAP_WEIGHT) * np.log(role.loc[both])
    )
    prior = prior.where(prior.notna(), role.where(valid_role))
    prior = prior.where(prior.notna(), draft.where(draft.gt(0)))
    prior = prior.where(prior.notna(), cold)
    return np.clip(prior.to_numpy(dtype=float), 1e-5, 1.0)


def residualise(values: np.ndarray, dummies: np.ndarray) -> np.ndarray:
    """Residual after least-squares removal of the control block."""
    coefficients, *_ = np.linalg.lstsq(dummies, values, rcond=None)
    return values - dummies @ coefficients


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--min-share", type=float, default=0.01)
    args = parser.parse_args(argv)

    rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    rows = rows[rows.season < 2026].copy()
    rows = rows[~rows.position.eq("QB")]
    rows = rows[
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce")
        .fillna(0).ne(1)
    ].reset_index(drop=True)

    rows["_prior"] = entry_prior(rows)
    # Observed exposure, not projected: this screens the *feature*, and routing
    # it through a snap projection would fold that model's error into the answer.
    exposure = pd.to_numeric(rows.get("snap_share"), errors="coerce")
    rows["_score"] = rows["_prior"] * exposure
    rows = rows[np.isfinite(rows["_score"]) & rows["_score"].gt(0)].reset_index(drop=True)

    grouped = rows.groupby(["season", "team"])["_score"]
    rows["_prior_share"] = rows["_score"] / grouped.transform("sum")
    targets = pd.to_numeric(rows["targets"], errors="coerce").fillna(0.0)
    team_targets = targets.groupby([rows.season, rows.team]).transform("sum")
    rows["_observed_share"] = targets / team_targets.where(team_targets > 0)

    keep = (
        rows["_observed_share"].gt(args.min_share)
        & rows["_prior_share"].gt(args.min_share)
    )
    rows = rows[keep].reset_index(drop=True)
    residual = np.log(rows["_observed_share"]) - np.log(rows["_prior_share"])

    controls = pd.get_dummies(rows["position"], drop_first=False).to_numpy(float)
    log_prior = np.log(rows["_prior_share"].to_numpy(float))
    controls = np.column_stack(
        [controls, log_prior, log_prior**2, np.ones(len(rows))]
    )
    residual_y = residualise(residual.to_numpy(float), controls)

    print(f"{len(rows)} player-seasons, {rows.season.min()}-{rows.season.max()}")
    print("residual log(observed share / prior-allocation share)")
    print("controlled for position and log prior share (quadratic)\n")
    print(f"{'feature':<40} {'n':>6} {'r':>8} {'p':>10}")
    for name in CANDIDATES:
        if name not in rows:
            print(f"{name:<40} {'--':>6} {'absent':>8}")
            continue
        values = pd.to_numeric(rows[name], errors="coerce")
        at = values.notna().to_numpy()
        if at.sum() < 200:
            print(f"{name:<40} {int(at.sum()):>6} {'too few':>8}")
            continue
        x = residualise(values[at].to_numpy(float), controls[at])
        y = residualise(residual_y[at], controls[at])
        if x.std() < 1e-12 or y.std() < 1e-12:
            print(f"{name:<40} {int(at.sum()):>6} {'constant':>8}")
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        # Fisher z, two-sided.
        from scipy import stats
        n = int(at.sum())
        z = np.arctanh(r) * np.sqrt(max(n - 3 - controls.shape[1], 1))
        p = float(2 * stats.norm.sf(abs(z)))
        print(f"{name:<40} {n:>6} {r:>+8.4f} {p:>10.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
