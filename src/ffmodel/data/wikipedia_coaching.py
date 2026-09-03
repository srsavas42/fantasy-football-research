"""Resumable Wikipedia coaching-history and scheme-lineage pipeline.

The scraper uses the MediaWiki API, archives exact page revisions, and keeps
source facts separate from the modeling rule that chooses a scheme carrier:

* use the head coach when his *prior* history includes offensive coordinator;
* otherwise use the season's offensive coordinator;
* when no OC is documented, fall back to the HC and flag the row for review.

All prior-team lineage is restricted to seasons before the projected season.
Wikipedia is a starting source, not authoritative truth; ambiguous or missing
rows are emitted to ``review_queue.csv`` rather than silently guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ffmodel.config import CACHE_DIR, LEGACY_YEARLY_DIR, WIKIPEDIA_COACHING_DIR
from ffmodel.data import ingest
from ffmodel.data.http import RemoteDataError, get_json

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE = "https://en.wikipedia.org/wiki/"
WIKIPEDIA_LICENSE = "Wikipedia text: CC BY-SA 4.0; page revision attribution retained"

ASSIGNMENT_COLUMNS = [
    "season",
    "team_code",
    "franchise_code",
    "team_name",
    "role",
    "coach_name",
    "coach_page_title",
    "assignment_order",
    "start_week",
    "end_week",
    "assignment_note",
    "is_interim",
    "needs_review",
    "extraction_method",
    "team_season_page",
    "source_url",
    "source_revision_id",
    "source_revision_timestamp",
]

HISTORY_COLUMNS = [
    "coach_name",
    "coach_page_title",
    "organization",
    "organization_page_title",
    "organization_team_code",
    "is_nfl",
    "start_season",
    "end_season",
    "is_current",
    "role",
    "is_offensive_coordinator",
    "raw_entry",
    "source_url",
    "source_revision_id",
    "source_revision_timestamp",
]

TEAM_SEASON_COLUMNS = [
    "season",
    "team_code",
    "franchise_code",
    "team_name",
    "data_source",
]


@dataclass(frozen=True)
class TeamIdentity:
    team_code: str
    franchise_code: str
    team_name: str


_STATIC_TEAMS = {
    "ATL": ("ATL", "Atlanta Falcons"),
    "BUF": ("BUF", "Buffalo Bills"),
    "CAR": ("CAR", "Carolina Panthers"),
    "CHI": ("CHI", "Chicago Bears"),
    "CIN": ("CIN", "Cincinnati Bengals"),
    "CLE": ("CLE", "Cleveland Browns"),
    "CLV": ("CLE", "Cleveland Browns"),
    "DAL": ("DAL", "Dallas Cowboys"),
    "DEN": ("DEN", "Denver Broncos"),
    "DET": ("DET", "Detroit Lions"),
    "GB": ("GB", "Green Bay Packers"),
    "GNB": ("GB", "Green Bay Packers"),
    "JAX": ("JAX", "Jacksonville Jaguars"),
    "KC": ("KC", "Kansas City Chiefs"),
    "KAN": ("KC", "Kansas City Chiefs"),
    "MIA": ("MIA", "Miami Dolphins"),
    "MIN": ("MIN", "Minnesota Vikings"),
    "NO": ("NO", "New Orleans Saints"),
    "NOR": ("NO", "New Orleans Saints"),
    "NYG": ("NYG", "New York Giants"),
    "NYJ": ("NYJ", "New York Jets"),
    "PHI": ("PHI", "Philadelphia Eagles"),
    "PIT": ("PIT", "Pittsburgh Steelers"),
    "SEA": ("SEA", "Seattle Seahawks"),
    "SF": ("SF", "San Francisco 49ers"),
    "SFO": ("SF", "San Francisco 49ers"),
    "TB": ("TB", "Tampa Bay Buccaneers"),
    "TAM": ("TB", "Tampa Bay Buccaneers"),
}


def team_identity(team_code: str, season: int) -> TeamIdentity:
    """Resolve repository/nflverse abbreviations to era-correct page names."""
    code = str(team_code).upper().strip()
    if code in _STATIC_TEAMS:
        franchise, name = _STATIC_TEAMS[code]
        return TeamIdentity(code, franchise, name)
    # "AZ" is what the nflverse season-roster feed calls Arizona, distinct from
    # the "ARI" the weekly feeds use. Added when the 2026 projection build hit
    # it: one unresolved code aborts the whole frame, which is the right
    # behaviour and worth keeping, so the fix belongs in the resolver rather
    # than at the call site where it would drift from the player frame.
    if code in {"ARI", "ARZ", "AZ", "PHO", "CRD"}:
        name = "Phoenix Cardinals" if season <= 1993 else "Arizona Cardinals"
        return TeamIdentity(code, "ARI", name)
    if code == "STL" and season <= 1987:
        return TeamIdentity(code, "ARI", "St. Louis Cardinals")
    if code in {"BAL", "BLT", "RAV"}:
        if season <= 1983:
            return TeamIdentity(code, "IND", "Baltimore Colts")
        return TeamIdentity(code, "BAL", "Baltimore Ravens")
    if code in {"IND", "CLT"}:
        return TeamIdentity(code, "IND", "Indianapolis Colts")
    if code in {"BOS", "NWE", "NE"}:
        name = "Boston Patriots" if season <= 1970 else "New England Patriots"
        return TeamIdentity(code, "NE", name)
    if code in {"HOU", "HST", "HTX"}:
        if season <= 1996:
            return TeamIdentity(code, "TEN", "Houston Oilers")
        return TeamIdentity(code, "HOU", "Houston Texans")
    if code in {"TEN", "OTI"}:
        name = "Tennessee Oilers" if season <= 1998 else "Tennessee Titans"
        return TeamIdentity(code, "TEN", name)
    if code in {"SDG", "SD", "LAC"}:
        name = "San Diego Chargers" if season <= 2016 else "Los Angeles Chargers"
        return TeamIdentity(code, "LAC", name)
    if code in {"RAM", "LAR", "LA"}:
        name = "Los Angeles Rams" if season <= 1994 or season >= 2016 else "St. Louis Rams"
        return TeamIdentity(code, "LAR", name)
    if code in {"SL", "STL"}:
        return TeamIdentity(code, "LAR", "St. Louis Rams")
    if code in {"OAK", "RAI", "LVR", "LV"}:
        if 1982 <= season <= 1994:
            name = "Los Angeles Raiders"
        elif season >= 2020:
            name = "Las Vegas Raiders"
        else:
            name = "Oakland Raiders"
        return TeamIdentity(code, "LV", name)
    if code == "WAS":
        if season <= 2019:
            name = "Washington Redskins"
        elif season <= 2021:
            name = "Washington Football Team"
        else:
            name = "Washington Commanders"
        return TeamIdentity(code, "WAS", name)
    raise KeyError(f"unsupported team code {team_code!r} for {season}")


_NFL_NAME_TO_CODE = {
    name.lower(): franchise
    for _, (franchise, name) in _STATIC_TEAMS.items()
}
_NFL_NAME_TO_CODE.update(
    {
        "arizona cardinals": "ARI",
        "phoenix cardinals": "ARI",
        "st. louis cardinals": "ARI",
        "st louis cardinals": "ARI",
        "baltimore colts": "IND",
        "indianapolis colts": "IND",
        "baltimore ravens": "BAL",
        "boston patriots": "NE",
        "new england patriots": "NE",
        "houston oilers": "TEN",
        "tennessee oilers": "TEN",
        "tennessee titans": "TEN",
        "houston texans": "HOU",
        "san diego chargers": "LAC",
        "los angeles chargers": "LAC",
        "los angeles rams": "LAR",
        "st. louis rams": "LAR",
        "st louis rams": "LAR",
        "oakland raiders": "LV",
        "los angeles raiders": "LV",
        "las vegas raiders": "LV",
        "washington redskins": "WAS",
        "washington football team": "WAS",
        "washington commanders": "WAS",
    }
)


def franchise_code_for_name(name: str | None) -> str | None:
    if not name:
        return None
    clean = re.sub(r"\s+", " ", str(name)).strip().lower()
    clean = re.sub(r"\s+football$", "", clean)
    return _NFL_NAME_TO_CODE.get(clean)


def available_legacy_seasons() -> list[int]:
    return sorted(
        int(path.stem)
        for path in LEGACY_YEARLY_DIR.glob("*.csv")
        if path.stem.isdigit()
    )


def discover_team_seasons(
    seasons: Iterable[int] | None = None,
    *,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Find exact team-seasons represented by committed or requested data."""
    requested = sorted(set(map(int, seasons or available_legacy_seasons())))
    rows: list[dict[str, Any]] = []
    remote_seasons = []
    for season in requested:
        path = LEGACY_YEARLY_DIR / f"{season}.csv"
        if not path.exists():
            remote_seasons.append(season)
            continue
        raw = pd.read_csv(path, usecols=lambda column: column in {"Tm", "Team"})
        if raw.shape[1] == 0:
            continue
        for code in sorted(raw.iloc[:, 0].dropna().astype(str).unique()):
            if re.fullmatch(r"\d+TM", code):
                continue
            identity = team_identity(code, season)
            rows.append(
                {
                    "season": season,
                    **asdict(identity),
                    "data_source": "legacy_yearly",
                }
            )

    if remote_seasons:
        schedules = ingest.load_schedules(remote_seasons, cache_dir=cache_dir)
        for season, games in schedules.groupby("season"):
            codes = pd.concat([games["home_team"], games["away_team"]]).dropna().unique()
            for code in sorted(map(str, codes)):
                identity = team_identity(code, int(season))
                rows.append(
                    {
                        "season": int(season),
                        **asdict(identity),
                        "data_source": "nflverse_schedules",
                    }
                )
    if not rows:
        return pd.DataFrame(columns=TEAM_SEASON_COLUMNS)
    return (
        pd.DataFrame(rows, columns=TEAM_SEASON_COLUMNS)
        .drop_duplicates(["season", "franchise_code"])
        .sort_values(["season", "franchise_code"])
        .reset_index(drop=True)
    )


@dataclass
class WikipediaPage:
    requested_title: str
    title: str
    page_id: int | None
    revision_id: int | None
    revision_timestamp: str | None
    wikitext: str
    missing: bool
    fetched_at: str

    @property
    def url(self) -> str:
        return WIKIPEDIA_PAGE + self.title.replace(" ", "_")


class WikipediaClient:
    """Polite MediaWiki client with revision-preserving disk cache."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        delay_seconds: float = 0.2,
        refresh: bool = False,
        offline: bool = False,
        retries: int = 3,
    ):
        root = Path(cache_dir) if cache_dir is not None else CACHE_DIR
        self.cache_dir = root / "wikipedia" / "pages"
        self.delay_seconds = max(0.0, float(delay_seconds))
        self.refresh = refresh
        self.offline = offline
        self.retries = retries
        self._last_request = 0.0

    def _path(self, title: str) -> Path:
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

    def _request(self, params: dict[str, Any]) -> Any:
        if self.offline:
            raise FileNotFoundError("Wikipedia page is not cached and --offline was set")
        wait = self.delay_seconds - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        for attempt in range(self.retries):
            try:
                payload = get_json(WIKIPEDIA_API, params=params)
                self._last_request = time.monotonic()
                if isinstance(payload, dict) and "error" in payload:
                    raise RemoteDataError(str(payload["error"]))
                return payload
            except RemoteDataError:
                if attempt + 1 >= self.retries:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def fetch_page(self, title: str) -> WikipediaPage:
        path = self._path(title)
        if path.exists() and not self.refresh:
            return WikipediaPage(**json.loads(path.read_text(encoding="utf-8")))
        payload = self._request(
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "ids|timestamp|content",
                "rvslots": "main",
                "titles": title,
                "redirects": 1,
            }
        )
        page = payload["query"]["pages"][0]
        missing = bool(page.get("missing", False))
        revision = (page.get("revisions") or [{}])[0]
        content = revision.get("slots", {}).get("main", {}).get("content", "")
        result = WikipediaPage(
            requested_title=title,
            title=page.get("title", title),
            page_id=page.get("pageid"),
            revision_id=revision.get("revid"),
            revision_timestamp=revision.get("timestamp"),
            wikitext=content,
            missing=missing,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return result

    def search(self, query: str, limit: int = 5) -> list[str]:
        payload = self._request(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
            }
        )
        return [item["title"] for item in payload.get("query", {}).get("search", [])]


def _mw():
    try:
        import mwparserfromhell
    except ImportError as exc:
        raise RuntimeError(
            'Wikipedia coaching scrape requires `pip install -e ".[scrape]"`.'
        ) from exc
    return mwparserfromhell


def _normal_name(value: Any) -> str:
    return re.sub(r"[ _]+", " ", str(value)).strip().lower()


def _find_template(wikitext: str, names: set[str]):
    code = _mw().parse(wikitext)
    wanted = {_normal_name(name) for name in names}
    for template in code.filter_templates(recursive=True):
        if _normal_name(template.name) in wanted:
            return template
    return None


def _param(template, names: Iterable[str]) -> str | None:
    if template is None:
        return None
    wanted = {_normal_name(name) for name in names}
    for parameter in template.params:
        if _normal_name(parameter.name) in wanted:
            return str(parameter.value).strip()
    return None


def _render_wiki_text(value: str | None) -> str:
    if not value:
        return ""
    code = _mw().parse(re.sub(r"<br\s*/?>", "\n", str(value), flags=re.I))
    for template in reversed(code.filter_templates(recursive=True)):
        name = _normal_name(template.name)
        positional = [str(param.value) for param in template.params if not param.showkey]
        if name in {"nfl year", "nfly"}:
            years = [
                item.strip()
                for item in positional
                if re.fullmatch(r"(?:\d{4}|present|current)", item.strip(), re.I)
            ]
            replacement = "–".join(years[:2])
        elif name in {"ubl", "unbulleted list", "plainlist", "hlist", "nowrap", "small"}:
            replacement = " ".join(positional)
        else:
            continue
        code.replace(template, replacement)
    plain = code.strip_code(normalize=True, collapse=True)
    plain = html.unescape(str(plain))
    return re.sub(r"\s+", " ", plain).strip(" *|;,:()")


_LINK_RE = re.compile(r"\[\[(?P<title>[^\]|#]+)(?:\|(?P<label>[^\]]+))?\]\]")


def _strip_references(value: str) -> str:
    """Remove references so linked publishers are never parsed as coaches."""
    cleaned = re.sub(r"<!--.*?-->", "", value, flags=re.S)
    cleaned = re.sub(r"<ref\b[^>]*/\s*>", "", cleaned, flags=re.I)
    cleaned = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", "", cleaned, flags=re.I | re.S)
    code = _mw().parse(cleaned)
    for template in reversed(code.filter_templates(recursive=True)):
        if _normal_name(template.name).startswith("cite "):
            code.remove(template)
    return str(code)


def _assignment_rows(value: str | None, role: str, method: str) -> list[dict[str, Any]]:
    if not value:
        return []
    value = _strip_references(value)
    matches = list(_LINK_RE.finditer(value))
    rows = []
    seen_pages: set[str] = set()
    for index, match in enumerate(matches):
        page_title = match.group("title").strip()
        if page_title in seen_pages:
            continue
        seen_pages.add(page_title)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        note = _render_wiki_text(value[match.end() : end])
        rows.append(
            {
                "role": role,
                "coach_name": _render_wiki_text(match.group("label") or match.group("title")),
                "coach_page_title": page_title,
                "assignment_order": len(rows) + 1,
                "assignment_note": note,
                "is_interim": "interim" in note.lower(),
                "extraction_method": method,
            }
        )
    if not rows:
        name = _render_wiki_text(value)
        if name:
            rows.append(
                {
                    "role": role,
                    "coach_name": name,
                    "coach_page_title": pd.NA,
                    "assignment_order": 1,
                    "assignment_note": "unlinked name",
                    "is_interim": "interim" in name.lower(),
                    "extraction_method": method + "_unlinked",
                }
            )
    return rows


def _staff_fallback(wikitext: str, role: str) -> list[dict[str, Any]]:
    label = "Head coach" if role == "HC" else "Offensive coordinator"
    pattern = re.compile(
        rf"{label}\s*(?:–|—|-|:)\s*(?P<value>\[\[[^\]]+\]\]|[^\n|]+)", re.I
    )
    match = pattern.search(wikitext)
    return _assignment_rows(match.group("value"), role, "staff_template") if match else []


def _apply_week_spans(rows: list[dict[str, Any]], season: int) -> None:
    max_week = 18 if season >= 2021 else 17
    if len(rows) == 1:
        rows[0].update(
            start_week=1,
            end_week=max_week,
            needs_review=bool(pd.isna(rows[0].get("coach_page_title"))),
        )
        return
    next_start: int | None = 1
    for index, row in enumerate(rows):
        row["start_week"] = next_start
        match = re.search(r"(?:after|through)\s+week\s+(\d+)", row["assignment_note"], re.I)
        if match:
            row["end_week"] = int(match.group(1))
            next_start = row["end_week"] + 1
        elif index + 1 == len(rows) and next_start is not None:
            row["end_week"] = max_week
        else:
            row["end_week"] = pd.NA
            next_start = None
    unresolved = any(
        pd.isna(row["start_week"])
        or pd.isna(row["end_week"])
        or pd.isna(row.get("coach_page_title"))
        for row in rows
    )
    for row in rows:
        row["needs_review"] = unresolved


def parse_team_season_page(
    page: WikipediaPage, team: TeamIdentity, season: int
) -> pd.DataFrame:
    """Extract long-form HC/OC assignments from one season page."""
    template = _find_template(page.wikitext, {"Infobox NFL team season"})
    team_name = _render_wiki_text(_param(template, {"team"})) or team.team_name
    hc = _assignment_rows(_param(template, {"coach", "head coach"}), "HC", "infobox")
    oc = _assignment_rows(
        _param(template, {"off coach", "off_coach", "offensive coach"}),
        "OC",
        "infobox",
    )
    if not hc:
        hc = _staff_fallback(page.wikitext, "HC")
    if not oc:
        oc = _staff_fallback(page.wikitext, "OC")
    for group in (hc, oc):
        _apply_week_spans(group, season)
    rows = []
    for row in [*hc, *oc]:
        rows.append(
            {
                "season": season,
                "team_code": team.team_code,
                "franchise_code": team.franchise_code,
                "team_name": team_name,
                **row,
                "team_season_page": page.title,
                "source_url": page.url,
                "source_revision_id": page.revision_id,
                "source_revision_timestamp": page.revision_timestamp,
            }
        )
    return pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)


def _resolve_team_page(client: WikipediaClient, team: TeamIdentity, season: int) -> WikipediaPage:
    expected = f"{season} {team.team_name} season"
    page = client.fetch_page(expected)
    if not page.missing:
        return page
    for title in client.search(f'intitle:{season} "{team.team_name}" season'):
        if str(season) in title and "season" in title.lower():
            candidate = client.fetch_page(title)
            if not candidate.missing:
                return candidate
    return page


def _years(raw: str) -> tuple[int | None, int | None, bool]:
    rendered = _render_wiki_text(raw)
    found = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", rendered)]
    current = "present" in rendered.lower()
    if not found:
        return None, None, current
    return found[0], (None if current else found[-1]), current


def _organization(raw: str) -> tuple[str, str | None]:
    links = list(_LINK_RE.finditer(raw))
    if links:
        link = links[0]
        return (
            _render_wiki_text(link.group("label") or link.group("title")),
            link.group("title").strip(),
        )
    plain = _render_wiki_text(raw)
    return re.split(r"\s*\((?:19|20)\d{2}", plain, maxsplit=1)[0].strip(), None


def _entry_role(raw: str, organization: str) -> str:
    pieces = re.split(r"<br\s*/?>", raw, flags=re.I)
    if len(pieces) > 1:
        return _render_wiki_text(" ".join(pieces[1:]))
    plain = _render_wiki_text(raw)
    if organization and plain.lower().startswith(organization.lower()):
        plain = plain[len(organization) :].strip()
    plain = re.sub(r"\(?\b(?:19|20)\d{2}(?:\s*[–-]\s*(?:(?:19|20)\d{2}|present))?\)?", "", plain)
    return plain.strip(" -*(),")


def _history_row(
    page: WikipediaPage,
    coach_name: str,
    organization: str,
    organization_page: str | None,
    start: int | None,
    end: int | None,
    current: bool,
    role: str,
    raw: str,
) -> dict[str, Any]:
    team_code = franchise_code_for_name(organization) or franchise_code_for_name(organization_page)
    return {
        "coach_name": coach_name,
        "coach_page_title": page.title,
        "organization": organization,
        "organization_page_title": organization_page,
        "organization_team_code": team_code,
        "is_nfl": team_code is not None,
        "start_season": start,
        "end_season": end,
        "is_current": current,
        "role": role or pd.NA,
        "is_offensive_coordinator": bool(
            re.search(r"\b(?:interim\s+)?offensive coordinator\b", role, re.I)
        ),
        "raw_entry": raw.strip(),
        "source_url": page.url,
        "source_revision_id": page.revision_id,
        "source_revision_timestamp": page.revision_timestamp,
    }


def _pastcoaching_history(page: WikipediaPage, template, coach_name: str) -> list[dict[str, Any]]:
    raw = _param(template, {"pastcoaching", "past coaching"})
    if not raw:
        return []
    groups: list[dict[str, Any]] = []
    current_group = None
    for line in raw.splitlines():
        match = re.match(r"^\s*(\*+)\s*(.+)$", line)
        if not match:
            continue
        level, content = len(match.group(1)), match.group(2).strip()
        if level == 1:
            organization, organization_page = _organization(content)
            start, end, current = _years(content)
            current_group = {
                "raw": content,
                "organization": organization,
                "organization_page": organization_page,
                "start": start,
                "end": end,
                "current": current,
                "role": _entry_role(content, organization),
                "children": [],
            }
            groups.append(current_group)
        elif current_group is not None:
            current_group["children"].append(content)

    rows = []
    for group in groups:
        if group["children"]:
            for child in group["children"]:
                start, end, current = _years(child)
                rows.append(
                    _history_row(
                        page,
                        coach_name,
                        group["organization"],
                        group["organization_page"],
                        start or group["start"],
                        end if end is not None else (None if current else group["end"]),
                        current or group["current"],
                        _entry_role(child, ""),
                        child,
                    )
                )
        else:
            rows.append(
                _history_row(
                    page,
                    coach_name,
                    group["organization"],
                    group["organization_page"],
                    group["start"],
                    group["end"],
                    group["current"],
                    group["role"],
                    group["raw"],
                )
            )
    return rows


def _indexed_coaching_history(page: WikipediaPage, template, coach_name: str):
    rows = []
    params = {_normal_name(param.name).replace(" ", ""): str(param.value) for param in template.params}
    indexes = sorted(
        int(match.group(1))
        for key in params
        if (match := re.fullmatch(r"coachyears(\d+)", key))
    )
    for index in indexes:
        years_raw = params.get(f"coachyears{index}", "")
        team_raw = params.get(f"coachteam{index}", "")
        if not team_raw:
            continue
        organization, page_title = _organization(team_raw)
        role_match = re.search(r"\(([^()]*(?:coach|coordinator)[^()]*)\)\s*$", _render_wiki_text(team_raw), re.I)
        start, end, current = _years(years_raw)
        rows.append(
            _history_row(
                page,
                coach_name,
                organization,
                page_title,
                start,
                end,
                current,
                role_match.group(1) if role_match else "",
                f"{years_raw} | {team_raw}",
            )
        )
    return rows


def parse_coach_history(page: WikipediaPage, fallback_name: str) -> pd.DataFrame:
    template = _find_template(page.wikitext, {"Infobox NFL biography", "Infobox college coach"})
    if template is None:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    coach_name = _render_wiki_text(_param(template, {"name"})) or fallback_name
    rows = _pastcoaching_history(page, template, coach_name)
    if not rows:
        rows = _indexed_coaching_history(page, template, coach_name)
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def _resolve_coach_page(
    client: WikipediaClient, page_title: str | None, coach_name: str
) -> WikipediaPage:
    # ``page_title`` is ``pd.NA`` for the "unlinked name" fallback rows
    # ``_assignment_rows`` emits when a coach's name wasn't wiki-linked, and
    # ``pd.NA`` raises on ``bool()`` rather than behaving like a normal falsy
    # value. ``_optional_text`` already exists for exactly this -- it checks
    # ``is None`` and ``pd.isna`` before ever coercing to a plain string, so
    # it never triggers that ambiguous-boolean error. This function's own
    # fallback on the last line had the identical bug (``page_title or
    # coach_name``), unreached only because the first crash always fired
    # first.
    resolved_title = _optional_text(page_title)
    if resolved_title is not None:
        page = client.fetch_page(resolved_title)
        if not page.missing:
            return page
    for title in client.search(f'"{coach_name}" American football coach'):
        candidate = client.fetch_page(title)
        if not candidate.missing:
            return candidate
    fallback_title = resolved_title or coach_name
    return WikipediaPage(
        requested_title=fallback_title,
        title=fallback_title,
        page_id=None,
        revision_id=None,
        revision_timestamp=None,
        wikitext="",
        missing=True,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def _optional_text(value: Any) -> str | None:
    return None if value is None or pd.isna(value) or not str(value).strip() else str(value)


def build_scheme_sources(assignments: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Apply the user-specified HC-prior-OC else OC selection rule."""
    rows = []
    if assignments.empty:
        return pd.DataFrame()
    ending = assignments.sort_values("assignment_order").drop_duplicates(
        ["season", "franchise_code", "role"], keep="last"
    )
    prior_lookup = {
        (int(row.season), row.franchise_code, row.role): row.coach_page_title
        for row in ending.itertuples()
    }
    for (season, team_code), group in assignments.groupby(["season", "franchise_code"]):
        hcs = group[group["role"] == "HC"].sort_values("assignment_order")
        ocs = group[group["role"] == "OC"].sort_values("assignment_order")
        hc = hcs.iloc[0] if not hcs.empty else None
        oc = ocs.iloc[0] if not ocs.empty else None
        hc_page = None if hc is None else _optional_text(hc["coach_page_title"])
        hc_history = history[
            (history["coach_page_title"] == hc_page)
            & (pd.to_numeric(history["start_season"], errors="coerce") < int(season))
        ] if hc_page is not None else history.iloc[0:0]
        hc_was_oc = bool(hc_history["is_offensive_coordinator"].fillna(False).any())
        if hc is not None and hc_was_oc:
            selected, basis = hc, "hc_prior_offensive_coordinator"
        elif oc is not None:
            selected, basis = oc, "offensive_coordinator"
        elif hc is not None:
            selected, basis = hc, "hc_fallback_no_oc_documented"
        else:
            selected, basis = None, "unresolved_no_hc_or_oc"
        selected_page = (
            None if selected is None else _optional_text(selected["coach_page_title"])
        )
        selected_history = (
            history[history["coach_page_title"] == selected_page]
            if selected_page is not None
            else history.iloc[0:0]
        )
        coach_source = selected_history.iloc[0] if not selected_history.empty else None
        previous_hc = prior_lookup.get((int(season) - 1, team_code, "HC"))
        previous_oc = prior_lookup.get((int(season) - 1, team_code, "OC"))
        rows.append(
            {
                "season": int(season),
                "franchise_code": team_code,
                "team_name": group["team_name"].iloc[0],
                "head_coach": None if hc is None else hc["coach_name"],
                "head_coach_page_title": hc_page,
                "offensive_coordinator": None if oc is None else oc["coach_name"],
                "offensive_coordinator_page_title": (
                    None if oc is None else _optional_text(oc["coach_page_title"])
                ),
                "scheme_coach": None if selected is None else selected["coach_name"],
                "scheme_coach_page_title": selected_page,
                "scheme_basis": basis,
                "hc_was_prior_oc": hc_was_oc,
                "new_head_coach": (
                    pd.NA if previous_hc is None else previous_hc != hc_page
                ),
                "new_offensive_coordinator": (
                    pd.NA
                    if previous_oc is None
                    else oc is None
                    or previous_oc != _optional_text(oc["coach_page_title"])
                ),
                "has_midseason_change": len(hcs) > 1 or len(ocs) > 1,
                "team_season_source_url": (
                    None if selected is None else selected.get("source_url")
                ),
                "team_season_source_revision_id": (
                    None if selected is None else selected.get("source_revision_id")
                ),
                "team_season_source_revision_timestamp": (
                    None
                    if selected is None
                    else selected.get("source_revision_timestamp")
                ),
                "scheme_coach_source_url": (
                    None if coach_source is None else coach_source.get("source_url")
                ),
                "scheme_coach_source_revision_id": (
                    None
                    if coach_source is None
                    else coach_source.get("source_revision_id")
                ),
                "scheme_coach_source_revision_timestamp": (
                    None
                    if coach_source is None
                    else coach_source.get("source_revision_timestamp")
                ),
                "needs_review": bool(
                    hc is None
                    or oc is None
                    or len(hcs) > 1
                    or len(ocs) > 1
                    or selected_page is None
                    or selected_history.empty
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "franchise_code"]).reset_index(drop=True)


def build_scheme_lineage(
    scheme_sources: pd.DataFrame,
    history: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Expand the selected coach's prior NFL stops and attach prior-team HCs."""
    primary_hc = (
        assignments[assignments["role"] == "HC"]
        .sort_values("assignment_order")
        .drop_duplicates(["season", "franchise_code"])
    )
    mentors = {
        (int(row.season), row.franchise_code): row.coach_name
        for row in primary_hc.itertuples()
    }
    rows = []
    for source in scheme_sources.itertuples():
        if _optional_text(source.scheme_coach_page_title) is None:
            continue
        stops = history[
            (history["coach_page_title"] == source.scheme_coach_page_title)
            & history["organization_team_code"].notna()
        ]
        for stop in stops.itertuples():
            if pd.isna(stop.start_season) or int(stop.start_season) >= source.season:
                continue
            end = source.season - 1 if pd.isna(stop.end_season) else min(int(stop.end_season), source.season - 1)
            for prior_season in range(int(stop.start_season), end + 1):
                rows.append(
                    {
                        "season": source.season,
                        "franchise_code": source.franchise_code,
                        "scheme_coach": source.scheme_coach,
                        "scheme_coach_page_title": source.scheme_coach_page_title,
                        "scheme_basis": source.scheme_basis,
                        "prior_team_code": stop.organization_team_code,
                        "prior_team_name": stop.organization,
                        "prior_season": prior_season,
                        "prior_role": stop.role,
                        "mentor_head_coach": mentors.get((prior_season, stop.organization_team_code)),
                        "recency_years": source.season - prior_season,
                        "source_url": stop.source_url,
                        "source_revision_id": stop.source_revision_id,
                    }
                )
    columns = [
        "season", "franchise_code", "scheme_coach", "scheme_coach_page_title",
        "scheme_basis", "prior_team_code", "prior_team_name", "prior_season",
        "prior_role", "mentor_head_coach", "recency_years", "source_url",
        "source_revision_id",
    ]
    return pd.DataFrame(rows, columns=columns).drop_duplicates()


def _write_table(frame: pd.DataFrame, name: str, output_dir: Path) -> None:
    frame.to_csv(output_dir / f"{name}.csv", index=False)
    frame.to_parquet(output_dir / f"{name}.parquet", index=False)


def scrape_wikipedia_coaching(
    seasons: Iterable[int] | None = None,
    *,
    teams: Iterable[str] | None = None,
    output_dir: Path | str = WIKIPEDIA_COACHING_DIR,
    cache_dir: Path | str | None = None,
    delay_seconds: float = 0.2,
    refresh: bool = False,
    offline: bool = False,
    limit_team_seasons: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the complete two-stage Wikipedia scrape and lineage build."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    team_seasons = discover_team_seasons(seasons, cache_dir=cache_dir)
    if teams:
        wanted = {str(team).upper() for team in teams}
        team_seasons = team_seasons[
            team_seasons["team_code"].isin(wanted)
            | team_seasons["franchise_code"].isin(wanted)
        ]
    if limit_team_seasons is not None:
        team_seasons = team_seasons.head(limit_team_seasons)

    client = WikipediaClient(
        cache_dir,
        delay_seconds=delay_seconds,
        refresh=refresh,
        offline=offline,
    )
    assignment_frames = []
    reviews: list[dict[str, Any]] = []
    total = len(team_seasons)
    for index, row in enumerate(team_seasons.itertuples(), start=1):
        identity = TeamIdentity(row.team_code, row.franchise_code, row.team_name)
        page = _resolve_team_page(client, identity, int(row.season))
        print(f"team pages {index}/{total}: {page.title}")
        if page.missing:
            reviews.append(
                {"issue": "missing_team_season_page", "season": row.season, "team": row.franchise_code, "subject": page.requested_title}
            )
            continue
        frame = parse_team_season_page(page, identity, int(row.season))
        if frame.empty or not (frame["role"] == "HC").any():
            reviews.append(
                {"issue": "missing_head_coach", "season": row.season, "team": row.franchise_code, "subject": page.title}
            )
        if frame.empty or not (frame["role"] == "OC").any():
            reviews.append(
                {"issue": "missing_offensive_coordinator", "season": row.season, "team": row.franchise_code, "subject": page.title}
            )
        assignment_frames.append(frame)
    assignments = (
        pd.concat(assignment_frames, ignore_index=True)
        if assignment_frames
        else pd.DataFrame(columns=ASSIGNMENT_COLUMNS)
    )

    history_frames = []
    coach_keys = assignments[["coach_name", "coach_page_title"]].drop_duplicates()
    total_coaches = len(coach_keys)
    for index, coach in enumerate(coach_keys.itertuples(index=False), start=1):
        page = _resolve_coach_page(client, coach.coach_page_title, coach.coach_name)
        print(f"coach pages {index}/{total_coaches}: {page.title}")
        if page.missing:
            reviews.append(
                {"issue": "missing_coach_page", "season": pd.NA, "team": pd.NA, "subject": coach.coach_name}
            )
            continue
        frame = parse_coach_history(page, coach.coach_name)
        if frame.empty:
            reviews.append(
                {"issue": "missing_structured_coach_history", "season": pd.NA, "team": pd.NA, "subject": page.title}
            )
        history_frames.append(frame)
    history = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame(columns=HISTORY_COLUMNS)
    )
    scheme_sources = build_scheme_sources(assignments, history)
    lineage = build_scheme_lineage(scheme_sources, history, assignments)
    if not scheme_sources.empty:
        for row in scheme_sources[scheme_sources["needs_review"]].itertuples():
            reviews.append(
                {"issue": "scheme_source_needs_review", "season": row.season, "team": row.franchise_code, "subject": row.scheme_coach}
            )
    review_queue = pd.DataFrame(
        reviews, columns=["issue", "season", "team", "subject"]
    ).drop_duplicates()

    _write_table(team_seasons, "team_seasons", output_dir)
    _write_table(assignments, "team_season_assignments", output_dir)
    _write_table(history, "coach_history", output_dir)
    _write_table(scheme_sources, "scheme_sources", output_dir)
    _write_table(lineage, "scheme_lineage", output_dir)
    review_queue.to_csv(output_dir / "review_queue.csv", index=False)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons": sorted(set(team_seasons["season"].astype(int))) if not team_seasons.empty else [],
        "team_filter": list(teams or []),
        "rows": {
            "team_seasons": len(team_seasons),
            "assignments": len(assignments),
            "coach_history": len(history),
            "scheme_sources": len(scheme_sources),
            "scheme_lineage": len(lineage),
            "review_queue": len(review_queue),
        },
        "source": WIKIPEDIA_API,
        "license": WIKIPEDIA_LICENSE,
        "selection_rule": "HC if prior OC; otherwise OC; HC fallback if no OC documented",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "team_seasons": team_seasons,
        "assignments": assignments,
        "history": history,
        "scheme_sources": scheme_sources,
        "lineage": lineage,
        "review_queue": review_queue,
    }


def parse_season_tokens(tokens: Iterable[str] | None) -> list[int] | None:
    if not tokens:
        return None
    seasons = set()
    for token in tokens:
        for piece in token.split(","):
            if ":" in piece:
                start, end = map(int, piece.split(":", 1))
                seasons.update(range(start, end + 1))
            else:
                seasons.add(int(piece))
    return sorted(seasons)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ffmodel-coaches")
    parser.add_argument(
        "--seasons",
        nargs="*",
        help="years or inclusive ranges, e.g. 1999:2025; default is all committed yearly data",
    )
    parser.add_argument("--teams", nargs="*", help="team/franchise codes for a focused run")
    parser.add_argument("--output-dir", default=str(WIKIPEDIA_COACHING_DIR))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--delay", type=float, default=0.2, help="minimum seconds between API calls")
    parser.add_argument("--refresh", action="store_true", help="re-fetch cached Wikipedia pages")
    parser.add_argument("--offline", action="store_true", help="use only archived page responses")
    parser.add_argument("--limit-team-seasons", type=int, help="small smoke/review run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = scrape_wikipedia_coaching(
        parse_season_tokens(args.seasons),
        teams=args.teams,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        delay_seconds=args.delay,
        refresh=args.refresh,
        offline=args.offline,
        limit_team_seasons=args.limit_team_seasons,
    )
    for name, frame in outputs.items():
        print(f"{name}: {len(frame):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
