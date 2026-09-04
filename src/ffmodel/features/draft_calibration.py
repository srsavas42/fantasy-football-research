"""Fit the rookie draft-capital prior to realized rookie seasons.

The cold-start claim curves in :mod:`ffmodel.features.draft` were hand-set and
documented as calibratable. They are not merely an ordering: the model consumes
them as a prior wherever a lagged role is missing, so their magnitude is used
directly and being wrong by a factor of two matters.

The functional form is kept — ``base * exp(-(pick - 1) / scale)`` — because it
is monotone in draft capital, has two interpretable parameters, and cannot go
negative. Only the parameters are learned. Each position/stream is fit
separately rather than splitting one claim by a fixed carry fraction, since a
back's receiving role and rushing role do not decay at the same rate.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# Streams a rookie can claim, and the realized column each is fit against.
#
# These are volume *shares*, and fitting against them is the defect
# scripts/diagnose_cold_start_prior.py traced the cold-start under-projection
# to. ``_role_prior`` consumes the result as a per-snap rate -- its own comment
# says it has to be one -- and the softmax score is
# ``log(role_prior) + log(exposure)``. A share already contains exposure, so
# fitting on one and then multiplying by exposure applies draft capital twice:
# 29.96x from the curve on top of 7.14x from exposure, where the per-snap rate
# actually varies by 1.79x. Kept for the ``target="share"`` comparison arm only.
CLAIM_STREAMS = {
    "target": "target_share",
    "carry": "carry_share",
    "pass": "pass_attempt_share",
}

# The same streams as a per-snap rate: (numerator, exposure). This is the
# quantity ``_role_prior`` actually wants, so a curve fitted here is on the
# scale it is consumed on.
CLAIM_RATE_STREAMS = {
    "target": ("targets", "offense_snaps"),
    "carry": ("rush_att", "offense_snaps"),
    "pass": ("pass_att", "offense_snaps"),
}

# A rate computed over a handful of snaps is noise, not a tendency. Rookies
# below this are dropped from a rate fit; they are exactly the rows whose
# exposure the snap model is already projecting near zero.
MIN_RATE_SNAPS = 50

# Search grid for the e-folding pick distance. Bounded well outside anything the
# data has supported so a fit that runs to an edge is visible rather than silent.
_SCALE_GRID = np.arange(10.0, 400.0, 2.0)

# The pick a player with none stands in at, shared with ffmodel.features.draft.
UNDRAFTED_PICK = 220.0

# The hand-set curves this calibration replaced, frozen so the validation script
# has a fixed baseline to beat. Derived from the original constants: one claim
# per position on a scale of 60, split target/carry by a fixed carry fraction,
# with a separate passing claim for quarterbacks.
HAND_SET_CLAIM_CURVES: dict[tuple[str, str], tuple[float, float]] = {
    ("QB", "target"): (0.0, 60.0),
    ("QB", "carry"): (0.0, 60.0),
    ("QB", "pass"): (0.78, 60.0),
    ("RB", "target"): (0.34 * 0.25, 60.0),
    ("RB", "carry"): (0.34 * 0.75, 60.0),
    ("RB", "pass"): (0.0, 60.0),
    ("WR", "target"): (0.22, 60.0),
    ("WR", "carry"): (0.0, 60.0),
    ("WR", "pass"): (0.0, 60.0),
    ("TE", "target"): (0.12, 60.0),
    ("TE", "carry"): (0.0, 60.0),
    ("TE", "pass"): (0.0, 60.0),
}


def rookie_seasons(
    player_rows: pd.DataFrame, *, include_undrafted: bool = False
) -> pd.DataFrame:
    """Rookie-season rows: draft capital is merged season-matched, so a present
    ``overall_pick`` marks a player's first year.

    ``include_undrafted`` also keeps first-year players with no pick, at the
    ``_UNDRAFTED_PICK`` stand-in. Without it the undrafted tail is excluded from
    every fit and then served by extrapolating the exponential past the end of
    its own data -- which is 61% of cold-start rows taking a value nothing ever
    validated. Requires an ``experience`` column to identify them.
    """
    if "overall_pick" not in player_rows.columns:
        return player_rows.iloc[0:0]
    out = player_rows.copy()
    out["overall_pick"] = pd.to_numeric(out["overall_pick"], errors="coerce")
    drafted = out[out["overall_pick"].notna()]
    if not include_undrafted or "experience" not in out.columns:
        return drafted
    experience = pd.to_numeric(out["experience"], errors="coerce")
    undrafted = out[out["overall_pick"].isna() & experience.le(0)].copy()
    undrafted["overall_pick"] = UNDRAFTED_PICK
    return pd.concat([drafted, undrafted], ignore_index=True)


def fit_claim_curve(picks, shares) -> tuple[float, float]:
    """Least-squares ``(base, scale)`` for ``base * exp(-(pick - 1) / scale)``.

    For any fixed scale the optimal base is closed form, so the search is a
    one-dimensional scan rather than a general optimiser — no new dependency,
    and the result is deterministic.
    """
    pick = np.asarray(picks, dtype=float)
    share = np.asarray(shares, dtype=float)
    keep = np.isfinite(pick) & np.isfinite(share)
    pick, share = pick[keep], share[keep]
    if len(pick) < 2:
        return 0.0, float(_SCALE_GRID[0])

    best = (0.0, float(_SCALE_GRID[0]), np.inf)
    for scale in _SCALE_GRID:
        decay = np.exp(-(pick - 1.0) / scale)
        denominator = float(np.dot(decay, decay))
        if denominator <= 0:
            continue
        base = float(np.dot(share, decay) / denominator)
        if base < 0:
            base = 0.0
        residual = share - base * decay
        sse = float(np.dot(residual, residual))
        if sse < best[2]:
            best = (base, float(scale), sse)
    return best[0], best[1]


def fit_rookie_priors(
    player_rows: pd.DataFrame,
    *,
    positions: Iterable[str] = ("QB", "RB", "WR", "TE"),
    min_rows: int = 20,
    target: str = "share",
) -> dict[tuple[str, str], tuple[float, float]]:
    """Fit every position/stream curve from realized rookie seasons.

    ``target="rate"`` fits per-snap rates rather than volume shares, which is
    the scale ``_role_prior`` consumes and the correction for the double-count
    described on CLAIM_STREAMS. It also admits the undrafted tail and drops
    rookies below MIN_RATE_SNAPS, whose rate would be noise.

    Streams a position does not meaningfully claim, and those with too few
    observations to fit, are returned as a zero claim rather than a noisy one.
    """
    if target not in {"share", "rate"}:
        raise ValueError(f"target must be 'share' or 'rate', not {target!r}")
    rookies = rookie_seasons(player_rows, include_undrafted=target == "rate")
    if target == "rate":
        snaps = pd.to_numeric(rookies.get("offense_snaps"), errors="coerce").fillna(0.0)
        rookies = rookies[snaps.ge(MIN_RATE_SNAPS)]

    fitted: dict[tuple[str, str], tuple[float, float]] = {}
    for position in positions:
        sub = rookies[rookies["position"].astype(str).eq(position)]
        streams = CLAIM_STREAMS if target == "share" else CLAIM_RATE_STREAMS
        for stream, spec in streams.items():
            if target == "share":
                realized = sub[spec] if spec in sub.columns else None
            else:
                numerator, exposure = spec
                if numerator in sub.columns and exposure in sub.columns:
                    denominator = pd.to_numeric(sub[exposure], errors="coerce")
                    realized = pd.to_numeric(sub[numerator], errors="coerce") / (
                        denominator.where(denominator > 0)
                    )
                else:
                    realized = None
            if realized is None or len(sub) < min_rows:
                fitted[(position, stream)] = (0.0, float(_SCALE_GRID[0]))
                continue
            fitted[(position, stream)] = fit_claim_curve(sub["overall_pick"], realized)
    return fitted


def claim_from_curve(overall_pick, curve: tuple[float, float]) -> float:
    """Evaluate a fitted curve, treating a missing pick as undrafted."""
    base, scale = curve
    if base <= 0:
        return 0.0
    pick = (
        UNDRAFTED_PICK
        if overall_pick is None or pd.isna(overall_pick)
        else float(overall_pick)
    )
    return float(base * np.exp(-(pick - 1.0) / scale))


def score_prior(
    player_rows: pd.DataFrame, priors: dict[tuple[str, str], tuple[float, float]]
) -> pd.DataFrame:
    """Mean absolute and squared error per position/stream on held-out rookies."""
    rookies = rookie_seasons(player_rows)
    rows = []
    for (position, stream), curve in sorted(priors.items()):
        column = CLAIM_STREAMS[stream]
        sub = rookies[rookies["position"].astype(str).eq(position)]
        if column not in sub.columns or sub.empty:
            continue
        actual = pd.to_numeric(sub[column], errors="coerce").to_numpy(dtype=float)
        predicted = np.array(
            [claim_from_curve(p, curve) for p in sub["overall_pick"]], dtype=float
        )
        keep = np.isfinite(actual) & np.isfinite(predicted)
        if not keep.any():
            continue
        error = predicted[keep] - actual[keep]
        rows.append(
            {
                "position": position,
                "stream": stream,
                "n": int(keep.sum()),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt((error**2).mean())),
                "mean_actual": float(actual[keep].mean()),
                "mean_predicted": float(predicted[keep].mean()),
            }
        )
    return pd.DataFrame(rows)
