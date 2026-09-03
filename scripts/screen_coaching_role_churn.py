"""Does a change of scheme carrier break role continuity for snaps and volume?

The earlier coach screen (scripts/screen_zone_teammate_coach.py) asked whether
coach *identity* explains residual variance in efficiency, and found a team
effect wearing a hat: 6.95% on rec_yards_per_target collapsing to 3.82% at
p=0.305 once team identity was absorbed. It also noted the coaching *tree*
question was untestable, because the scraper that fills the lineage tables
could not run in that environment.

This asks a different question, aimed at the layer that screen did not cover.
Not "which coach" -- a ~60-level dummy nearly collinear with franchise -- but
"did the scheme carrier change", a binary that is known before the season
starts and is not a team effect at all, since the same franchise takes both
values across its own history.

The hypothesis is about *dispersion*, not level: when a new play-caller
arrives, last season's role should describe this season's role less well.
That is the shape the volume model can actually use. It already carries
``role_innovation_scale`` and ``cold_role_innovation`` -- machinery that says
"rows with no prior role of their own deserve a wider innovation" -- so
"rows whose offense just changed hands" is the same kind of claim about the
same parameter, rather than a new covariate fighting the softmax offset (which
is how room structure failed: see docs/target-competition-2026-09.md).

Three responses, all preseason-known inputs:

  target share   the roster softmax residual, as in screen_target_room_quality
  carry share    same
  snap share     the snap model's own response

LEAKAGE: ``has_midseason_change`` is deliberately NOT used as a predictor. A
coach fired in week 8 is not knowable in August, and a feature that reads it
would score well here and be unavailable when serving. ``new_head_coach`` and
``new_offensive_coordinator`` compare against the *ending* staff of season
Y-1, which offseason hiring makes known before season Y -- those are fair.

    python scripts/screen_coaching_role_churn.py
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

from ffmodel.data.coaching import load_scheme_sources  # noqa: E402
from ffmodel.models.volume_season_average import STREAMS  # noqa: E402

# Preseason-known only. has_midseason_change is excluded on purpose; see the
# module docstring.
COACH_FLAGS = ("new_head_coach", "new_offensive_coordinator", "new_scheme_coach")


def stream_residual(rows: pd.DataFrame, stream: str) -> pd.DataFrame:
    """Log residual of observed share against the deterministic prior allocation."""
    spec = STREAMS[stream]
    count_col = spec["count"]
    d = rows.copy()
    d["_prior"] = entry_prior(d) if stream == "target" else _generic_prior(d, stream)
    exposure = pd.to_numeric(d.get("snap_share"), errors="coerce")
    d["_score"] = d["_prior"] * exposure
    d = d[np.isfinite(d["_score"]) & d["_score"].gt(0)].reset_index(drop=True)
    d["_prior_share"] = d["_score"] / d.groupby(["season", "team"])["_score"].transform("sum")
    counts = pd.to_numeric(d[count_col], errors="coerce").fillna(0.0)
    totals = counts.groupby([d.season, d.team]).transform("sum")
    d["_observed"] = counts / totals.where(totals > 0)
    d = d[d["_observed"].gt(0.01) & d["_prior_share"].gt(0.01)].reset_index(drop=True)
    d["_residual"] = np.log(d["_observed"]) - np.log(d["_prior_share"])
    return d


def _generic_prior(rows: pd.DataFrame, stream: str) -> np.ndarray:
    spec = STREAMS[stream]
    per_snap = pd.to_numeric(rows.get(spec["per_snap_role"]), errors="coerce")
    role = pd.to_numeric(rows.get(spec["role"]), errors="coerce")
    draft = pd.to_numeric(rows.get(spec["draft"]), errors="coerce")
    cold = rows["position"].map(spec["fallback"]).astype(float)
    prior = per_snap.where(per_snap.gt(0))
    prior = prior.where(prior.notna(), role.where(role.gt(0)))
    prior = prior.where(prior.notna(), draft.where(draft.gt(0)))
    prior = prior.where(prior.notna(), cold)
    return np.clip(prior.to_numpy(dtype=float), 1e-5, 1.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-walkforward"))
    args = parser.parse_args(argv)

    rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    rows = rows[
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    ].reset_index(drop=True)

    sources = load_scheme_sources()
    # The scheme carrier is the modeled play-caller: HC when his prior history
    # includes OC, else the OC. "new_scheme_coach" is the union that matters --
    # whoever the model would treat as owning the scheme changed hands.
    sources = sources.sort_values(["franchise_code", "season"]).copy()
    previous = sources.groupby("franchise_code")["scheme_coach_page_title"].shift(1)
    sources["new_scheme_coach"] = (
        sources["scheme_coach_page_title"].notna()
        & previous.notna()
        & sources["scheme_coach_page_title"].ne(previous)
    )
    keep = ["season", "franchise_code", *COACH_FLAGS, "scheme_basis", "hc_was_prior_oc"]
    coach = sources[keep].rename(columns={"franchise_code": "team"})

    merged = rows.merge(coach, on=["season", "team"], how="left")
    covered = merged[COACH_FLAGS[0]].notna().mean()
    print(f"{len(merged)} player-seasons, coach flags present on {covered:.1%}\n")

    for stream, label in (("target", "target share"), ("carry", "carry share")):
        block = stream_residual(merged, stream)
        block = block[block[COACH_FLAGS[0]].notna()]
        if stream == "target":
            block = block[~block.position.eq("QB")]
        print(f"=== {label} ({len(block)} rows) ===")
        log_prior = np.log(block["_prior_share"].to_numpy(float))
        controls = np.column_stack([
            pd.get_dummies(block["position"]).to_numpy(float),
            log_prior, log_prior**2, np.ones(len(block)),
        ])
        residual = block["_residual"].to_numpy(float)
        report(block, residual, controls)
        print()

    # Snap share: the snap model's own response, against its own lagged value.
    snaps = merged[
        merged[COACH_FLAGS[0]].notna()
        & pd.to_numeric(merged.get("prior_snap_share"), errors="coerce").gt(0.05)
        & pd.to_numeric(merged.get("snap_share"), errors="coerce").gt(0)
    ].reset_index(drop=True)
    prior_snap = pd.to_numeric(snaps["prior_snap_share"], errors="coerce").to_numpy(float)
    observed_snap = pd.to_numeric(snaps["snap_share"], errors="coerce").to_numpy(float)
    residual = np.log(observed_snap) - np.log(prior_snap)
    controls = np.column_stack([
        pd.get_dummies(snaps["position"]).to_numpy(float),
        np.log(prior_snap), np.log(prior_snap) ** 2, np.ones(len(snaps)),
    ])
    print(f"=== snap share ({len(snaps)} rows, prior snap share > 0.05) ===")
    report(snaps, residual, controls)
    return 0


def report(block: pd.DataFrame, residual: np.ndarray, controls: np.ndarray) -> None:
    """Level and dispersion effects for each preseason-known coach flag."""
    print(f"{'flag':<28}{'n_true':>7}{'level r':>10}{'p':>10}"
          f"{'|resid| true':>14}{'|resid| false':>15}{'disp r':>9}{'p':>10}")
    for flag in COACH_FLAGS:
        values = block[flag].astype(float).to_numpy()
        at = np.isfinite(values) & np.isfinite(residual)
        if at.sum() < 200 or len(set(values[at])) < 2:
            print(f"{flag:<28}{int(at.sum()):>7}{'too few':>10}")
            continue
        x = residualise(values[at], controls[at])
        # Level: does the coach change move the share up or down?
        y_level = residualise(residual[at], controls[at])
        r_level, p_level = partial(x, y_level, controls.shape[1], at.sum())
        # Dispersion: does it make the prior a worse description, either way?
        y_disp = residualise(np.abs(residual[at]), controls[at])
        r_disp, p_disp = partial(x, y_disp, controls.shape[1], at.sum())
        true_mask = values[at] > 0.5
        print(f"{flag:<28}{int(true_mask.sum()):>7}{r_level:>+10.4f}{p_level:>10.2e}"
              f"{np.abs(residual[at])[true_mask].mean():>14.4f}"
              f"{np.abs(residual[at])[~true_mask].mean():>15.4f}"
              f"{r_disp:>+9.4f}{p_disp:>10.2e}")


def partial(x: np.ndarray, y: np.ndarray, n_controls: int, n: int) -> tuple[float, float]:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    z = np.arctanh(r) * np.sqrt(max(n - 3 - n_controls, 1))
    return r, float(2 * stats.norm.sf(abs(z)))


if __name__ == "__main__":
    raise SystemExit(main())
