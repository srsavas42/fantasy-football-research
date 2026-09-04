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
    ("QB", "pass"): (0.78, _HAND_SET_SCALE),  # retained: lost the holdout
    ("QB", "carry"): (0.1153, 66.0),  # fit: rookie passers do carry
    ("QB", "target"): (0.0, _HAND_SET_SCALE),  # retained
    ("RB", "carry"): (0.4835, 112.0),  # fit
    ("RB", "target"): (0.1062, 106.0),  # fit
    ("RB", "pass"): (0.0, _HAND_SET_SCALE),  # retained
    ("WR", "target"): (0.22, _HAND_SET_SCALE),  # retained: lost the walk-forward
    ("WR", "carry"): (0.0, _HAND_SET_SCALE),  # retained: lost the holdout
    ("WR", "pass"): (0.0, _HAND_SET_SCALE),  # retained
    ("TE", "target"): (0.1735, 68.0),  # fit
    ("TE", "carry"): (0.0, _HAND_SET_SCALE),  # retained
    ("TE", "pass"): (0.0, _HAND_SET_SCALE),  # retained
}

# A 2026-09 refit against a per-snap *rate* rather than a volume share was
# measured and reverted. The units argument behind it is correct as far as it
# goes -- a share already contains playing time, ``_role_prior`` consumes this
# as a per-snap rate, and the softmax multiplies by exposure again -- but the
# rate fit has to condition on 50+ snaps to get a rate worth fitting, so it
# describes rookies who earned a role and is then applied to every rookie.
# Flattening moved undrafted players from 22% to 55% of all cold prior mass and
# the gate rejected it on every fold of every volume stream (target MAE +5.55%,
# carry +2.94%, pass +1.92%). Held out on 2024 it turned a cold target bias of
# -1.5% into +53.1% and doubled cold MAE. The steepness here is not only
# pricing per-snap usage; it is also pricing whether draft capital converts
# into a role at all, which projected exposure does not fully carry. See
# docs/target-competition-2026-09.md.

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
