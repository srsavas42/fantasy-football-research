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

DRAFT_COLUMNS = ["player_name", "position", "season", "team", "round", "overall_pick"]

# Coarse, documented prior for a rookie's first-year opportunity claim as a
# function of overall pick, split into target vs carry competition by position.
# The model learns the coefficient on aggregated competition, so what matters
# here is the *ordering* by draft capital, not exact magnitudes. Calibratable.
_CLAIM_BASE = {"RB": 0.34, "WR": 0.22, "TE": 0.12, "QB": 0.0}
_CLAIM_SCALE = 60.0  # e-folding pick distance
_CARRY_FRACTION = {"RB": 0.75, "WR": 0.0, "TE": 0.0, "QB": 0.0}


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

    picks = ingest.load_draft_picks(list(seasons))
    out = picks.rename(
        columns={"pfr_player_name": "player_name", "pick": "overall_pick"}
    )
    keep = ["player_name", "position", "season", "team", "round", "overall_pick"]
    return out[[c for c in keep if c in out.columns]]


def _finalize(df: pd.DataFrame, seasons: set[int]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=DRAFT_COLUMNS)
    df = df[df["position"].isin(SKILL_POSITIONS + ("QB",))]
    return df[df["season"].isin(seasons)].reset_index(drop=True)


def expected_rookie_claim(overall_pick, position: str) -> tuple[float, float]:
    """(target_claim, carry_claim) an incoming rookie is expected to take.

    Decays exponentially with overall pick; split into receiving vs rushing by
    position. Missing pick -> treated as a late pick (small claim).
    """
    import math

    base = _CLAIM_BASE.get(position, 0.0)
    pick = 220 if overall_pick is None or pd.isna(overall_pick) else float(overall_pick)
    claim = base * math.exp(-(pick - 1) / _CLAIM_SCALE)
    carry_frac = _CARRY_FRACTION.get(position, 0.0)
    return claim * (1 - carry_frac), claim * carry_frac
