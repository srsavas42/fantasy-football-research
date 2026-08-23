"""Preseason team win totals, normalized across four source vintages.

The market's opinion on a team, priced before the season starts. It is the one
team-level input in this package that is not derived from play-by-play, and
unlike the draft board it is an opinion about *teams* rather than players --
which is the layer the ADP work never reached.

Strictly preseason, which is the constraint the whole feature exists under: a
win total is posted before week one and settled after week eighteen, so using
the line is legitimate and using ``Actual Wins`` or ``Result`` would be reading
the answer. Those columns are dropped on load rather than carried and trusted
not to be used.

## Four vintages, three schemas, three team conventions

``2003_2022_win_totals.csv``
    ``season,team,line,over_odds,under_odds``, era-correct abbreviations --
    ``OAK`` through 2019 then ``LV``, ``SD`` through 2016 then ``LAC``,
    ``STL`` through 2015 then ``LA``.
``{2023,2024,2025}_nfl_regular_season_win_total_odds.csv``
    One file per season, no season column (it is in the filename), and teams as
    full names: ``Arizona Cardinals``.
``NFL Win Totals-export-2026-08-23.csv``
    A wide export with coach and hold columns, and *franchise-stable*
    abbreviations that ignore relocations -- the Raiders are ``OAK`` in 2026,
    six years after moving to Las Vegas, and the Rams are ``LAR`` where the
    2003-2022 file says ``LA``.

Every code and name is resolved through :func:`team_identity`, the same
resolver ``_normalize_teams`` uses to build the player frame, so the output
joins to ``player_rows.team`` without a second mapping that could disagree with
the first. That matters more than it looks: ``OAK`` in the 2026 file means Las
Vegas, and a hand-written map that took it at face value would silently drop
the Raiders from every join while reporting nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ffmodel.data.wikipedia_coaching import team_identity

DEFAULT_WIN_TOTAL_DIR = Path(__file__).resolve().parents[3] / "Vegas Win Totals"

# The franchises as the player frame knows them, used to build the reverse
# name lookup for the full-name vintages rather than hard-coding thirty-two
# strings that would then drift from the resolver.
FRANCHISE_CODES = (
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
)

WIN_TOTAL_COLUMNS = ("season", "team", "win_total", "over_odds", "under_odds")

# Columns naming what actually happened. Present in the 2023-2025 and 2026
# files and dropped on load: this is a preseason feature and a realized-outcome
# column sitting in the frame is an invitation to leak.
OUTCOME_COLUMNS = ("Actual Wins", "Result", "Actual", "Week Bet Settled")

_SEASON_IN_NAME = re.compile(r"(19|20)\d{2}")


def _name_lookup(season: int) -> dict[str, str]:
    """Full team name to franchise code, for one season's naming."""
    out: dict[str, str] = {}
    for code in FRANCHISE_CODES:
        identity = team_identity(code, season)
        out[identity.team_name] = identity.franchise_code
    return out


def _american_to_probability(odds: pd.Series) -> pd.Series:
    """Implied probability from American odds, before removing the vig.

    Positive odds pay ``odds/100``; negative odds risk ``|odds|/100`` to win 1.
    The two sides sum to more than one -- that excess is the book's margin and
    :func:`devig` removes it.
    """
    values = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(
            values >= 0,
            100.0 / (values + 100.0),
            -values / (-values + 100.0),
        ),
        index=values.index,
        dtype=float,
    ).where(values.notna())


def devig(over_odds, under_odds) -> pd.Series:
    """Vig-free probability that a team clears its win total.

    Both sides are priced with a margin, so the raw implied probabilities sum to
    about 1.05. Normalising by their sum removes it proportionally, which is the
    standard treatment and is exact when the book prices both sides with the
    same margin.

    Returned rather than folded into the loader because it is the quantity a
    model should read: a line of 9.5 at -200 and a line of 9.5 at +150 are very
    different opinions, and the line alone cannot tell them apart.
    """
    over = _american_to_probability(pd.Series(over_odds).reset_index(drop=True))
    under = _american_to_probability(pd.Series(under_odds).reset_index(drop=True))
    total = over + under
    return (over / total.where(total > 0)).astype(float)


def _load_wide_export(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    # This export repeats 'Over'/'Under' and 'Team' as column names, so pandas
    # disambiguates them and positional access is the only reliable route.
    columns = list(frame.columns)
    out = pd.DataFrame(
        {
            "season": pd.to_numeric(frame["Season"], errors="coerce"),
            "_raw_team": frame["Team"].astype(str) if "Team" in columns else frame.iloc[:, 0],
            "win_total": pd.to_numeric(frame["Vegas Total"], errors="coerce"),
            "over_odds": pd.to_numeric(frame.iloc[:, columns.index("Over")], errors="coerce"),
            "under_odds": pd.to_numeric(frame.iloc[:, columns.index("Under")], errors="coerce"),
        }
    )
    return out


def _load_named_season(path: Path) -> pd.DataFrame:
    season_match = _SEASON_IN_NAME.search(path.name)
    if season_match is None:
        raise ValueError(
            f"{path.name} has no season in its filename and no season column; "
            "these per-season exports carry the year only in the name"
        )
    season = int(season_match.group(0))
    frame = pd.read_csv(path)
    return pd.DataFrame(
        {
            "season": season,
            "_raw_team": frame["Team"].astype(str),
            "win_total": pd.to_numeric(frame["Win Total"], errors="coerce"),
            "over_odds": pd.to_numeric(frame["Over Odds"], errors="coerce"),
            "under_odds": pd.to_numeric(frame["Under Odds"], errors="coerce"),
        }
    )


def _load_long(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return pd.DataFrame(
        {
            "season": pd.to_numeric(frame["season"], errors="coerce"),
            "_raw_team": frame["team"].astype(str),
            "win_total": pd.to_numeric(frame["line"], errors="coerce"),
            "over_odds": pd.to_numeric(frame["over_odds"], errors="coerce"),
            "under_odds": pd.to_numeric(frame["under_odds"], errors="coerce"),
        }
    )


def _resolve_team(raw: str, season: int, names: dict[str, str]) -> str | None:
    text = str(raw).strip()
    if text in names:
        return names[text]
    try:
        return team_identity(text, int(season)).franchise_code
    except Exception:
        return None


def load_win_totals(directory: Path | str = DEFAULT_WIN_TOTAL_DIR) -> pd.DataFrame:
    """Every vintage in one frame, keyed the way the player rows are keyed.

    Raises rather than returning a short frame when a season is missing teams:
    a win-total feature with a silent hole becomes a missing-value pattern that
    the model reads as information, and thirty-one teams looks like thirty-two
    in every summary anyone would glance at.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"no win-total directory at {directory}")

    blocks: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.csv")):
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if "Vegas Total" in header:
            blocks.append(_load_wide_export(path))
        elif "Win Total" in header:
            blocks.append(_load_named_season(path))
        elif {"season", "team", "line"} <= set(header):
            blocks.append(_load_long(path))
        else:
            raise ValueError(
                f"{path.name} matches no known win-total schema; its columns "
                f"are {header}"
            )
    if not blocks:
        raise FileNotFoundError(f"no CSV files in {directory}")

    frame = pd.concat(blocks, ignore_index=True)
    frame = frame[frame["season"].notna()].copy()
    frame["season"] = frame["season"].astype(int)

    resolved = []
    for season, block in frame.groupby("season", sort=True):
        names = _name_lookup(int(season))
        block = block.copy()
        block["team"] = [
            _resolve_team(raw, int(season), names) for raw in block["_raw_team"]
        ]
        resolved.append(block)
    frame = pd.concat(resolved, ignore_index=True)

    unresolved = frame[frame["team"].isna()]
    if len(unresolved):
        offenders = sorted(set(unresolved["_raw_team"].astype(str)))
        raise ValueError(
            f"could not resolve these teams to a franchise code: {offenders}. "
            "Add them to the identity resolver rather than mapping them here, "
            "so the player frame and this table cannot disagree"
        )

    frame["over_probability"] = devig(frame["over_odds"], frame["under_odds"])
    out = frame[list(WIN_TOTAL_COLUMNS) + ["over_probability"]].copy()
    out = out.sort_values(["season", "team"]).reset_index(drop=True)

    duplicated = out.duplicated(["season", "team"], keep=False)
    if duplicated.any():
        pairs = out.loc[duplicated, ["season", "team"]].to_dict("records")
        raise ValueError(f"duplicate season/team win totals: {pairs}")

    counts = out.groupby("season").size()
    short = counts[counts != 32]
    if len(short):
        raise ValueError(
            f"these seasons do not have 32 teams: {short.to_dict()}. A hole in "
            "a win-total column reads as information to a model rather than as "
            "missing data"
        )
    return out
