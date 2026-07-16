"""Scoring must reproduce the legacy CSVs' fantasy point columns exactly."""

import pandas as pd
import pytest

from ffmodel.config import LEGACY_WEEKLY_DIR
from ffmodel.data import legacy
from ffmodel.simulation.scoring import fantasy_points

CASES = [
    ("standard", "StandardFantasyPoints"),
    ("half_ppr", "HalfPPRFantasyPoints"),
    ("ppr", "PPRFantasyPoints"),
]


@pytest.mark.parametrize("season,week", [(1999, 1), (2010, 8), (2020, 17)])
@pytest.mark.parametrize("fmt,col", CASES)
def test_reproduces_csv_points(season, week, fmt, col):
    raw = pd.read_csv(LEGACY_WEEKLY_DIR / str(season) / f"week{week}.csv", index_col=0)
    canonical = legacy.load_weekly([season])
    subset = canonical[canonical["week"] == week].reset_index(drop=True)
    assert len(subset) == len(raw)
    calc = fantasy_points(subset, fmt)
    assert (calc - raw[col].reset_index(drop=True)).abs().max() < 1e-9


def test_rules_ordering():
    df = legacy.load_weekly([2020])
    wk1 = df[df["week"] == 1]
    std = fantasy_points(wk1, "standard")
    half = fantasy_points(wk1, "half_ppr")
    ppr = fantasy_points(wk1, "ppr")
    assert ((half - std) - wk1["receptions"] * 0.5).abs().max() < 1e-9
    assert ((ppr - std) - wk1["receptions"] * 1.0).abs().max() < 1e-9
