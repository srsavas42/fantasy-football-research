"""Load Wikipedia-derived coaching lineage and manual play-caller overrides.

Wikipedia supplies reproducible HC/OC assignments and career-history priors.
It does not establish who called plays, so confirmed effective-date overrides
remain a separate, higher-authority table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ffmodel.config import COACHING_PERIODS_PATH, WIKIPEDIA_COACHING_DIR

COACHING_COLUMNS = [
    "team",
    "effective_from",
    "effective_to",
    "head_coach",
    "offensive_coordinator",
    "play_caller",
    "source_url",
    "confidence",
]

WIKIPEDIA_TABLE_KEYS = {
    "team_seasons": {"season", "franchise_code", "team_name"},
    "team_season_assignments": {
        "season", "franchise_code", "role", "coach_name", "source_revision_id"
    },
    "coach_history": {
        "coach_page_title", "organization", "start_season", "role"
    },
    "scheme_sources": {
        "season", "franchise_code", "scheme_coach", "scheme_basis"
    },
    "scheme_lineage": {
        "season", "franchise_code", "prior_team_code", "prior_season"
    },
}


def load_coaching_periods(path: str | Path = COACHING_PERIODS_PATH) -> pd.DataFrame:
    """Load and validate the manual play-caller table."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COACHING_COLUMNS)
    frame = pd.read_csv(path)
    missing = [column for column in COACHING_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"coaching table is missing columns: {missing}")
    for column in ("effective_from", "effective_to"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    if (frame["effective_to"] < frame["effective_from"]).any():
        raise ValueError("coaching effective_to must not precede effective_from")
    valid_confidence = {"confirmed", "high", "medium", "low"}
    invalid = set(frame["confidence"].dropna()) - valid_confidence
    if invalid:
        raise ValueError(f"invalid coaching confidence values: {sorted(invalid)}")
    return frame[COACHING_COLUMNS]


def load_wikipedia_coaching_table(
    table: str,
    directory: str | Path = WIKIPEDIA_COACHING_DIR,
) -> pd.DataFrame:
    """Load and minimally validate one generated Wikipedia coaching table."""
    if table not in WIKIPEDIA_TABLE_KEYS:
        raise ValueError(
            f"unknown Wikipedia coaching table {table!r}; "
            f"choose from {sorted(WIKIPEDIA_TABLE_KEYS)}"
        )
    directory = Path(directory)
    parquet_path = directory / f"{table}.parquet"
    csv_path = directory / f"{table}.csv"
    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        frame = pd.read_csv(csv_path)
    else:
        return pd.DataFrame(columns=sorted(WIKIPEDIA_TABLE_KEYS[table]))
    missing = WIKIPEDIA_TABLE_KEYS[table] - set(frame.columns)
    if missing:
        raise ValueError(f"{table} is missing columns: {sorted(missing)}")
    return frame


def load_team_season_assignments(
    directory: str | Path = WIKIPEDIA_COACHING_DIR,
) -> pd.DataFrame:
    return load_wikipedia_coaching_table("team_season_assignments", directory)


def load_scheme_sources(
    directory: str | Path = WIKIPEDIA_COACHING_DIR,
) -> pd.DataFrame:
    """Load one selected offensive scheme carrier per team-season."""
    return load_wikipedia_coaching_table("scheme_sources", directory)


def load_scheme_lineage(
    directory: str | Path = WIKIPEDIA_COACHING_DIR,
) -> pd.DataFrame:
    """Load prior NFL team-seasons for each selected scheme carrier."""
    return load_wikipedia_coaching_table("scheme_lineage", directory)
