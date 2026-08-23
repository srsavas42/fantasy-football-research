"""Loading preseason win totals across four source vintages.

The risky parts are not the parsing. They are that three of the files name
teams three different ways -- including one that calls the Raiders ``OAK`` six
years after they left Oakland -- and that two of them carry realized outcomes
next to the preseason line.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ffmodel.features.vegas import devig, load_win_totals


def _long(tmp_path, rows):
    frame = pd.DataFrame(rows, columns=["season", "team", "line", "over_odds", "under_odds"])
    frame.to_csv(tmp_path / "2003_2022_win_totals.csv", index=False)


def _named(tmp_path, season, rows):
    frame = pd.DataFrame(
        rows, columns=["Team", "Win Total", "Over Odds", "Under Odds"]
    )
    frame["Week Bet Settled"] = "Week 18"
    frame["Actual Wins"] = 9
    frame["Result"] = "Over"
    frame.to_csv(
        tmp_path / f"{season}_nfl_regular_season_win_total_odds.csv", index=False
    )


def _wide(tmp_path, season, codes):
    """The 2026-style export: duplicated column names and a Vegas Total.

    ``Over``/``Under``/``Team`` each appear twice in that file, so the loader
    has to read them positionally. Reproduced here rather than simplified,
    because the duplication is the part that can break.
    """
    frame = pd.DataFrame(
        {
            "Team": codes,
            "Season": season,
            "Coach": "",
            "QB": "",
            "Actual": 0,
            "Vegas Total": 8.5,
            "Adj. Total": 8.5,
            "Result": "Active",
            "Over": -110,
            "Under": -110,
            "Hold": 0.045,
        }
    )
    frame["Over "] = 0.5
    frame["Under "] = 0.5
    frame.columns = [c.rstrip() if c in ("Over ", "Under ") else c for c in frame.columns]
    frame.to_csv(tmp_path / f"NFL Win Totals-export-{season}.csv", index=False)


def _thirty_two(season):
    """A full slate, so the completeness guard does not fire in other tests."""
    codes = [
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
        "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA",
        "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
        "TEN", "WAS",
    ]
    return [(season, code, 8.5, -110, -110) for code in codes]


def test_the_three_schemas_load_into_one_frame(tmp_path):
    from ffmodel.data.wikipedia_coaching import team_identity

    _long(tmp_path, _thirty_two(2022))
    codes = [row[1] for row in _thirty_two(2023)]
    # Full names, spelled by the resolver so the test cannot drift from it.
    _named(
        tmp_path,
        2023,
        [(team_identity(c, 2023).team_name, 9.5, -120, 100) for c in codes],
    )
    _wide(tmp_path, 2026, codes)

    out = load_win_totals(tmp_path)
    assert set(out.season) == {2022, 2023, 2026}
    assert (out.groupby("season").size() == 32).all()
    assert list(out.columns) == [
        "season", "team", "win_total", "over_odds", "under_odds", "over_probability"
    ]


def test_realized_outcomes_never_reach_the_frame(tmp_path):
    """The per-season files carry Actual Wins next to the preseason line."""
    from ffmodel.data.wikipedia_coaching import team_identity

    codes = [row[1] for row in _thirty_two(2024)]
    _named(
        tmp_path,
        2024,
        [(team_identity(c, 2024).team_name, 9.5, -120, 100) for c in codes],
    )
    out = load_win_totals(tmp_path)

    for leaked in ("Actual Wins", "Result", "Week Bet Settled", "actual_wins"):
        assert leaked not in out.columns


def test_a_relocated_franchise_resolves_by_era_not_by_label(tmp_path):
    """The Raiders are OAK in 2003 and OAK again in the 2026 export.

    Those two mean different cities. Taking either at face value would split
    one franchise into two, or attach Las Vegas numbers to an Oakland row.
    """
    rows = _thirty_two(2005)
    rows = [(s, "OAK" if t == "LV" else t, l, o, u) for s, t, l, o, u in rows]
    _long(tmp_path, rows)

    out = load_win_totals(tmp_path)
    assert "OAK" not in set(out.team)
    assert (out.team == "LV").sum() == 1


def test_a_season_missing_a_team_is_refused(tmp_path):
    """Thirty-one teams looks like thirty-two in any summary worth glancing at."""
    _long(tmp_path, _thirty_two(2022)[:-1])

    with pytest.raises(ValueError, match="do not have 32 teams"):
        load_win_totals(tmp_path)


def test_an_unknown_team_is_named_rather_than_dropped(tmp_path):
    rows = _thirty_two(2022)
    rows[0] = (2022, "Gotham Rogues", 8.5, -110, -110)
    _long(tmp_path, rows)

    with pytest.raises(ValueError, match="could not resolve"):
        load_win_totals(tmp_path)


def test_an_unrecognised_schema_is_refused_with_its_columns(tmp_path):
    pd.DataFrame({"club": ["ARI"], "wins": [8.5]}).to_csv(
        tmp_path / "odd.csv", index=False
    )
    with pytest.raises(ValueError, match="matches no known win-total schema"):
        load_win_totals(tmp_path)


def test_devig_removes_the_book_margin():
    """Raw implied probabilities sum above one; the excess is the hold."""
    # -110 both sides: the classic 4.5% hold, symmetric, so a fair coin.
    assert devig([-110], [-110]).iloc[0] == pytest.approx(0.5)
    # A heavy favourite on the over.
    assert devig([-200], [170]).iloc[0] > 0.6
    assert devig([170], [-200]).iloc[0] < 0.4


def test_devig_is_symmetric():
    a = devig([-150], [130]).iloc[0]
    b = devig([130], [-150]).iloc[0]
    assert a + b == pytest.approx(1.0)


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_win_totals(tmp_path / "nope")
