import json

import pandas as pd
import pytest

mwparserfromhell = pytest.importorskip("mwparserfromhell")

from ffmodel.data import coaching
from ffmodel.data import wikipedia_coaching as wiki


def page(title: str, text: str) -> wiki.WikipediaPage:
    return wiki.WikipediaPage(
        requested_title=title,
        title=title,
        page_id=1,
        revision_id=123,
        revision_timestamp="2026-07-15T00:00:00Z",
        wikitext=text,
        missing=False,
        fetched_at="2026-07-15T00:00:01Z",
    )


def test_team_identity_uses_era_correct_names():
    assert wiki.team_identity("RAI", 1990).team_name == "Los Angeles Raiders"
    assert wiki.team_identity("OAK", 2019).team_name == "Oakland Raiders"
    assert wiki.team_identity("LVR", 2024).team_name == "Las Vegas Raiders"
    assert wiki.team_identity("STL", 1980).franchise_code == "ARI"
    assert wiki.team_identity("STL", 2000).franchise_code == "LAR"
    assert wiki.team_identity("HOU", 1995).team_name == "Houston Oilers"
    assert wiki.team_identity("HOU", 2024).team_name == "Houston Texans"


def test_team_identity_accepts_historical_nflverse_roster_aliases():
    assert wiki.team_identity("ARZ", 2014).franchise_code == "ARI"
    assert wiki.team_identity("BLT", 2014).franchise_code == "BAL"
    assert wiki.team_identity("CLV", 2014).franchise_code == "CLE"
    assert wiki.team_identity("HST", 2014).franchise_code == "HOU"
    assert wiki.team_identity("SL", 2014).franchise_code == "LAR"


def test_modern_infobox_extracts_midseason_oc_change_and_ignores_citations():
    text = """
{{Infobox NFL team season
| team = Buffalo Bills
| year = 2023
| coach = [[Sean McDermott]]<ref>Profile at [[ESPN]]</ref>
| off_coach = [[Ken Dorsey (American football)|Ken Dorsey]] (fired after Week 10)<br>
  [[Joe Brady (American football coach)|Joe Brady]] (interim)
}}
"""
    frame = wiki.parse_team_season_page(
        page("2023 Buffalo Bills season", text),
        wiki.team_identity("BUF", 2023),
        2023,
    )
    assert frame["coach_name"].tolist() == ["Sean McDermott", "Ken Dorsey", "Joe Brady"]
    oc = frame[frame["role"] == "OC"].reset_index(drop=True)
    assert oc.loc[0, "start_week"] == 1
    assert oc.loc[0, "end_week"] == 10
    assert oc.loc[1, "start_week"] == 11
    assert oc.loc[1, "end_week"] == 18
    assert bool(oc.loc[1, "is_interim"])
    assert not oc["needs_review"].any()


def test_old_staff_template_is_used_when_infobox_has_no_coordinator_fields():
    text = """
{{Infobox NFL team season | team = Indianapolis Colts | year = 1999}}
{{NFL final roster
| head_coach =
* Head coach – [[Jim E. Mora]]
| offensive =
* Offensive coordinator – [[Tom Moore (American football coach)|Tom Moore]]
}}
"""
    frame = wiki.parse_team_season_page(
        page("1999 Indianapolis Colts season", text),
        wiki.team_identity("IND", 1999),
        1999,
    )
    assert set(frame["role"]) == {"HC", "OC"}
    assert set(frame["extraction_method"]) == {"staff_template"}
    assert frame.set_index("role").loc["OC", "coach_name"] == "Tom Moore"


def test_coach_history_parses_nested_nfl_stops_and_present_year():
    text = """
{{Infobox NFL biography
| name = Alex Example
| pastcoaching =
* William & Mary (1998)<br>Graduate assistant
* {{ubl|[[Philadelphia Eagles]] ({{NFL Year|2001|2010}})}}
** {{ubl|({{NFL Year|2001|2003}})<br>Defensive assistant}}
** {{ubl|({{NFL Year|2009|2010}})<br>Defensive coordinator}}
* [[Carolina Panthers]] ({{NFL Year|2011|2016}})<br>Offensive coordinator
* [[Buffalo Bills]] ({{NFL Year|2017|present}})<br>Head coach
}}
"""
    frame = wiki.parse_coach_history(page("Alex Example", text), "Alex Example")
    panthers = frame[frame["organization"] == "Carolina Panthers"].iloc[0]
    bills = frame[frame["organization"] == "Buffalo Bills"].iloc[0]
    eagles = frame[frame["organization"] == "Philadelphia Eagles"]
    assert len(eagles) == 2
    assert {int(value) for value in eagles["start_season"]} == {2001, 2009}
    assert panthers["organization_team_code"] == "CAR"
    assert bool(panthers["is_offensive_coordinator"])
    assert int(bills["start_season"]) == 2017
    assert pd.isna(bills["end_season"])
    assert bool(bills["is_current"])


def test_scheme_source_uses_only_prior_oc_history():
    assignments = pd.DataFrame(
        [
            (2024, "BUF", "Buffalo Bills", "HC", "Defensive HC", "Defensive HC", 1),
            (2024, "BUF", "Buffalo Bills", "OC", "Bills OC", "Bills OC", 1),
            (2024, "TB", "Tampa Bay Buccaneers", "HC", "Future OC HC", "Future OC HC", 1),
            (2024, "TB", "Tampa Bay Buccaneers", "OC", "Bucs OC", "Bucs OC", 1),
            (2024, "LAR", "Los Angeles Rams", "HC", "Former OC HC", "Former OC HC", 1),
            (2024, "LAR", "Los Angeles Rams", "OC", "Rams OC", "Rams OC", 1),
        ],
        columns=[
            "season", "franchise_code", "team_name", "role", "coach_name",
            "coach_page_title", "assignment_order",
        ],
    )
    history = pd.DataFrame(
        [
            ("Defensive HC", 2020, False),
            ("Bills OC", 2020, True),
            ("Future OC HC", 2025, True),
            ("Bucs OC", 2021, True),
            ("Former OC HC", 2020, True),
            ("Rams OC", 2022, True),
        ],
        columns=["coach_page_title", "start_season", "is_offensive_coordinator"],
    )
    out = wiki.build_scheme_sources(assignments, history).set_index("franchise_code")
    assert out.loc["BUF", "scheme_coach"] == "Bills OC"
    assert out.loc["TB", "scheme_coach"] == "Bucs OC"
    assert out.loc["LAR", "scheme_coach"] == "Former OC HC"
    assert out.loc["LAR", "scheme_basis"] == "hc_prior_offensive_coordinator"


def test_lineage_is_strictly_prior_and_attaches_available_mentor():
    sources = pd.DataFrame(
        [
            {
                "season": 2024,
                "franchise_code": "BUF",
                "scheme_coach": "Example Coach",
                "scheme_coach_page_title": "Example Coach",
                "scheme_basis": "offensive_coordinator",
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "coach_page_title": "Example Coach",
                "organization_team_code": "LAR",
                "organization": "Los Angeles Rams",
                "start_season": 2022,
                "end_season": 2025,
                "role": "Quarterbacks coach",
                "source_url": "https://example.test/revision",
                "source_revision_id": 77,
            }
        ]
    )
    assignments = pd.DataFrame(
        [
            (2022, "LAR", "HC", "Mentor One", 1),
            (2023, "LAR", "HC", "Mentor Two", 1),
        ],
        columns=["season", "franchise_code", "role", "coach_name", "assignment_order"],
    )
    out = wiki.build_scheme_lineage(sources, history, assignments)
    assert out["prior_season"].tolist() == [2022, 2023]
    assert out["mentor_head_coach"].tolist() == ["Mentor One", "Mentor Two"]
    assert (out["prior_season"] < out["season"]).all()


def test_wikipedia_client_archives_revision_and_reuses_cache(monkeypatch, tmp_path):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "query": {
                "pages": [
                    {
                        "pageid": 42,
                        "title": "Example Coach",
                        "revisions": [
                            {
                                "revid": 9001,
                                "timestamp": "2026-07-15T00:00:00Z",
                                "slots": {"main": {"content": "example"}},
                            }
                        ],
                    }
                ]
            }
        }

    monkeypatch.setattr(wiki, "get_json", fake_get)
    client = wiki.WikipediaClient(tmp_path, delay_seconds=0)
    first = client.fetch_page("Example Coach")
    second = client.fetch_page("Example Coach")
    assert first.revision_id == second.revision_id == 9001
    assert len(calls) == 1
    cached = json.loads(next((tmp_path / "wikipedia" / "pages").glob("*.json")).read_text())
    assert cached["revision_id"] == 9001


def test_generated_table_loader_prefers_parquet_and_validates(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "season": 2024,
                "franchise_code": "BUF",
                "scheme_coach": "Example Coach",
                "scheme_basis": "offensive_coordinator",
            }
        ]
    )
    frame.to_parquet(tmp_path / "scheme_sources.parquet", index=False)
    loaded = coaching.load_scheme_sources(tmp_path)
    assert loaded.loc[0, "scheme_coach"] == "Example Coach"


def test_parse_season_tokens_supports_ranges_and_lists():
    assert wiki.parse_season_tokens(["2020:2022", "2024,2025"]) == [
        2020, 2021, 2022, 2024, 2025
    ]


def test_resolving_a_coach_page_survives_an_unlinked_name(monkeypatch, tmp_path):
    """``coach_page_title`` is ``pd.NA`` for the "unlinked name" fallback row
    ``_assignment_rows`` emits when a coach's name has no wiki link. Passing
    that straight to ``page_title and not pd.isna(page_title)`` used to raise
    ``TypeError: boolean value of NA is ambiguous`` -- pandas.NA refuses to be
    coerced to bool, unlike None or NaN. Caught only once the scraper ran to
    scale for the first time: every existing fixture here supplies a real
    linked coach name, so this path had never been exercised.
    """
    calls = []

    def fake_get(url, *, params=None, **kwargs):
        calls.append(params)
        if params.get("list") == "search":
            return {"query": {"search": [{"title": "Some Coach (American football)"}]}}
        return {"query": {"pages": [{"missing": True}]}}

    monkeypatch.setattr(wiki, "get_json", fake_get)
    client = wiki.WikipediaClient(tmp_path, delay_seconds=0)

    result = wiki._resolve_coach_page(client, pd.NA, "Some Coach")

    assert isinstance(result, wiki.WikipediaPage)
    # Fell through to the name search rather than crashing on the NA title.
    assert any(call.get("action") == "query" and "list" in call for call in calls)


def test_resolving_a_coach_page_falls_back_to_the_name_when_title_is_na(monkeypatch, tmp_path):
    """The final not-found fallback also read ``page_title`` truthily
    (``page_title or coach_name``) and would have hit the identical crash one
    call later, once the first one was fixed."""
    def fake_get(url, *, params=None, **kwargs):
        return {"query": {"pages": [{"missing": True}], "search": []}}

    monkeypatch.setattr(wiki, "get_json", fake_get)
    client = wiki.WikipediaClient(tmp_path, delay_seconds=0)

    result = wiki._resolve_coach_page(client, pd.NA, "Some Coach")

    assert result.missing
    assert result.title == "Some Coach"
