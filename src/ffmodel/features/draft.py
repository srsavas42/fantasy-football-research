"""Rookie draft capital — the "drafted competition" half of share competition.

A returning player's opportunity isn't just about volume *vacated* by departures;
it's also diluted by incoming talent. Rookies are the hardest incoming source to
see (they have no prior NFL usage), so we proxy their expected claim from draft
capital: earlier picks command more early-career opportunity.

Source is hybrid, matching the rest of the package: nflverse draft picks when
reachable (complete, abbreviated teams), else the committed combine CSV, whose
`Drafted (tm/rnd/yr)` string parses cleanly and whose team names all map to PFR
abbreviations via `misc/abbrev.csv`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ffmodel.config import REPO_ROOT
from ffmodel.features.volume import SKILL_POSITIONS

_COMBINE_CSV = REPO_ROOT / "combine" / "combine00to20.csv"
_ABBREV_CSV = REPO_ROOT / "misc" / "abbrev.csv"

DRAFT_COLUMNS = [
    "player_name",
    "position",
    "season",
    "team",
    "round",
    "overall_pick",
    "player_id",
]

# A rookie's first-year claim on each stream as a function of overall pick,
# ``base * exp(-(pick - 1) / scale)``. These are consumed as a prior wherever a
# lagged role is missing, so the magnitude is used directly, not just the
# ordering by draft capital.
#
# Fit on realized rookie seasons 2015-2024 by
# :mod:`ffmodel.features.draft_calibration`, holding 2025 out entirely. A fitted
# curve ships only where it beat the hand-set value on both MAE and RMSE twice:
# in a walk-forward over 2020-2024 run inside the training seasons, and again on
# the untouched 2025 holdout. Everything else keeps the hand-set curve.
#
# Two candidates the walk-forward promoted did not survive the holdout and were
# put back: a quarterback passing curve, and a small but non-zero receiver
# rushing claim. See ``scripts/validate_rookie_prior.py``.
#
# Refit before projecting a season once its predecessor has been played.
_HAND_SET_SCALE = 60.0
ROOKIE_CLAIM_CURVES: dict[tuple[str, str], tuple[float, float]] = {
    # position, stream: (base, scale)
    #
    # Refitted 2026-09 against a per-snap *rate* rather than a volume share.
    # The share fit was the cold-start under-projection's root cause: a share
    # already contains playing time, _role_prior consumes this as a per-snap
    # rate, and the softmax then multiplies by exposure again -- applying about
    # 214x of draft capital where the data supports about 13x. See
    # scripts/diagnose_cold_start_prior.py and
    # docs/target-competition-2026-09.md.
    #
    # Every fitted scale runs to 398, the top of _SCALE_GRID. That is the grid
    # doing its job rather than a failure: the module bounds it "well outside
    # anything the data has supported so a fit that runs to an edge is visible
    # rather than silent". What it makes visible is that an exponential is the
    # wrong shape for a per-snap rate -- conditional on playing, usage is very
    # nearly flat in draft slot (observed round-1 : undrafted is 1.79x, and the
    # refit curves land at 1.7x against the old 6-30x). A long scale is how an
    # exponential expresses "almost constant".
    #
    # The structural zeros are kept deliberately. The rate fit returns small
    # non-zero claims for them (WR carry 0.0097, TE carry 0.0010, QB target
    # 0.0004), but those streams were zeroed by earlier holdout results, and a
    # refit aimed at a different defect is not a reason to reopen them.
    ("QB", "pass"): (0.6198, 398.0),  # rate-fit
    ("QB", "carry"): (0.0886, 398.0),  # rate-fit
    ("QB", "target"): (0.0, _HAND_SET_SCALE),  # retained: structural zero
    ("RB", "carry"): (0.4734, 398.0),  # rate-fit
    ("RB", "target"): (0.1226, 398.0),  # rate-fit
    ("RB", "pass"): (0.0, _HAND_SET_SCALE),  # retained: structural zero
    ("WR", "target"): (0.1412, 398.0),  # rate-fit
    ("WR", "carry"): (0.0, _HAND_SET_SCALE),  # retained: lost the holdout
    ("WR", "pass"): (0.0, _HAND_SET_SCALE),  # retained: structural zero
    ("TE", "target"): (0.0970, 398.0),  # rate-fit
    ("TE", "carry"): (0.0, _HAND_SET_SCALE),  # retained: structural zero
    ("TE", "pass"): (0.0, _HAND_SET_SCALE),  # retained: structural zero
}

# The curves this replaced, kept so the paired walk-forward can rebuild the
# old draft priors on identical frames rather than needing a separate build.
LEGACY_SHARE_FIT_CURVES: dict[tuple[str, str], tuple[float, float]] = {
    ("QB", "pass"): (0.78, _HAND_SET_SCALE),
    ("QB", "carry"): (0.1153, 66.0),
    ("QB", "target"): (0.0, _HAND_SET_SCALE),
    ("RB", "carry"): (0.4835, 112.0),
    ("RB", "target"): (0.1062, 106.0),
    ("RB", "pass"): (0.0, _HAND_SET_SCALE),
    ("WR", "target"): (0.22, _HAND_SET_SCALE),
    ("WR", "carry"): (0.0, _HAND_SET_SCALE),
    ("WR", "pass"): (0.0, _HAND_SET_SCALE),
    ("TE", "target"): (0.1735, 68.0),
    ("TE", "carry"): (0.0, _HAND_SET_SCALE),
    ("TE", "pass"): (0.0, _HAND_SET_SCALE),
}

# A pick this late stands in for undrafted, matching the previous behaviour.
_UNDRAFTED_PICK = 220.0


def _claim(overall_pick, position: str, stream: str) -> float:
    import math

    base, scale = ROOKIE_CLAIM_CURVES.get((position, stream), (0.0, _HAND_SET_SCALE))
    if base <= 0:
        return 0.0
    pick = (
        _UNDRAFTED_PICK
        if overall_pick is None or pd.isna(overall_pick)
        else float(overall_pick)
    )
    return base * math.exp(-(pick - 1.0) / scale)


def _abbrev_map() -> dict[str, str]:
    ab = pd.read_csv(_ABBREV_CSV)
    return dict(zip(ab["Team"], ab["Abbrev"]))


def _parse_combine(seasons: set[int]) -> pd.DataFrame:
    df = pd.read_csv(_COMBINE_CSV)
    df = df[["Year", "Player", "Pos", "Drafted (tm/rnd/yr)"]].dropna(
        subset=["Drafted (tm/rnd/yr)"]
    )
    amap = _abbrev_map()
    rows = []
    for _, r in df.iterrows():
        parts = [p.strip() for p in str(r["Drafted (tm/rnd/yr)"]).split("/")]
        if len(parts) < 4:
            continue
        team = amap.get(parts[0])
        rnd = _digits(parts[1])
        pick = _digits(parts[2])
        year = _digits(parts[3])
        if team is None or year is None or int(year) not in seasons:
            continue
        rows.append(
            {
                "player_name": r["Player"],
                "position": r["Pos"],
                "season": int(year),
                "team": team,
                "round": int(rnd) if rnd else pd.NA,
                "overall_pick": int(pick) if pick else pd.NA,
            }
        )
    return pd.DataFrame(rows, columns=DRAFT_COLUMNS)


def _digits(s: str):
    m = re.sub(r"\D", "", str(s))
    return m or None


def load_draft_capital(seasons: Iterable[int], source: str = "auto") -> pd.DataFrame:
    """Per-rookie draft capital for the given draft years, skill positions only."""
    seasons = set(int(s) for s in seasons)
    if source in ("auto", "nflverse"):
        try:
            df = _load_nflverse(seasons)
            if df is not None:
                return _finalize(df, seasons)
        except Exception:
            if source == "nflverse":
                raise
    return _finalize(_parse_combine(seasons), seasons)


def _load_nflverse(seasons: set[int]):
    from ffmodel.data import ingest
    from ffmodel.data.identity import is_gsis_id, resolve_player_ids

    picks = ingest.load_draft_picks(list(seasons))
    out = picks.rename(
        columns={"pfr_player_name": "player_name", "pick": "overall_pick"}
    )
    # The feed's own ``gsis_id`` is not dependable for a recent class: it carries
    # PFR-style values there, which match nothing in the roster or depth feeds.
    # Keep it only where it is actually GSIS-shaped and resolve the rest.
    native = out.get("gsis_id", pd.Series(pd.NA, index=out.index)).astype("string")
    out["player_id"] = native.where(is_gsis_id(native))
    try:
        bridged = resolve_player_ids(out)
    except Exception:
        # Identity enrichment is an optimisation over the name join, never a
        # precondition for loading draft capital.
        bridged = pd.Series(pd.NA, index=out.index, dtype="string")
    out["player_id"] = out["player_id"].where(out["player_id"].notna(), bridged)
    keep = [
        "player_name",
        "position",
        "season",
        "team",
        "round",
        "overall_pick",
        "player_id",
    ]
    return out[[c for c in keep if c in out.columns]]


def _finalize(df: pd.DataFrame, seasons: set[int]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=DRAFT_COLUMNS)
    df = df[df["position"].isin(SKILL_POSITIONS + ("QB",))]
    return df[df["season"].isin(seasons)].reset_index(drop=True)


def expected_rookie_claim(overall_pick, position: str) -> tuple[float, float]:
    """(target_claim, carry_claim) an incoming rookie is expected to take.

    Each stream decays exponentially with overall pick on its own fitted curve,
    rather than one claim split by a fixed carry fraction: a back's receiving
    role and rushing role do not decay at the same rate. Missing pick -> treated
    as undrafted (small claim).
    """
    return (
        _claim(overall_pick, position, "target"),
        _claim(overall_pick, position, "carry"),
    )


def expected_rookie_pass_claim(overall_pick, position: str) -> float:
    """Prior share of team attempts for a rookie passer.

    Only quarterbacks receive a material cold-start passing prior. The roster
    allocator still retains every modeled offensive position so genuine trick
    attempts remain representable.
    """
    return _claim(overall_pick, position, "pass")
