"""Probe ESPN and Sleeper for injury information the nflverse feed does not carry.

The nflverse injury table names a body part and nothing else (see
``docs/ros-reconciliation-2026-10.md``). ESPN's injury endpoint and Sleeper's player
records are the two free sources that may carry the *nature* of an injury (tear,
strain, sprain, fracture, surgery, concussion) and a timeline. Neither keeps history,
so this does two things at once:

* prints what each one actually serves -- every field path with how often it is
  filled and an example -- so the question "is there anything valuable" is answered
  from the real payload rather than from memory; and
* saves a raw, dated snapshot under ``data/injury_snapshots/<date>/`` so that every
  run adds a week of history that no endpoint will ever give back.

Neither host is reachable from the research sandbox (egress policy), so this is run
by ``.github/workflows/probe-injury-sources.yml`` and the snapshot committed back.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ffmodel.data.http import RemoteDataError, get_json

ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
SLEEPER_PLAYERS = "https://api.sleeper.app/v1/players/nfl"

#: Words that describe the *nature* of an injury, which the body-part feed lacks.
NATURE_WORDS = {
    "tear": r"\b(?:torn|tear|tore|tears)\b",
    "strain": r"\bstrain(?:ed)?\b",
    "sprain": r"\bsprain(?:ed)?\b",
    "pull": r"\b(?:pull(?:ed)?|tweak(?:ed)?)\b",
    "fracture": r"\b(?:fracture[sd]?|broken|break)\b",
    "concussion": r"\b(?:concussion|head injury)\b",
    "surgery": r"\b(?:surgery|surgical|operation|scope|scoped)\b",
    "soreness": r"\b(?:sore|soreness|bruise[ds]?|contusion)\b",
    "dislocation": r"\b(?:dislocat\w+)\b",
}

#: Words that give a timeline, which would let a model learn return time directly.
TIMELINE_WORDS = {
    "weeks": r"\b\d+\s*(?:-|to)?\s*\d*\s*weeks?\b",
    "week-to-week": r"\bweek[- ]to[- ]week\b",
    "day-to-day": r"\bday[- ]to[- ]day\b",
    "season": r"\b(?:for the season|season-ending|rest of the season)\b",
    "return": r"\b(?:expected to return|return(?:s|ed)? (?:in|on|by)|timetable|expected back)\b",
    "ir": r"\b(?:injured reserve|\bIR\b)\b",
}

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE"}


def flatten(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    """Every leaf of a JSON value as (dotted path, value); list indexes are dropped."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from flatten(inner, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for inner in value:
            yield from flatten(inner, f"{prefix}[]")
    else:
        yield prefix, value


def field_table(records: list[dict]) -> list[dict]:
    """Per field path: how many records fill it, distinct values, and an example."""
    filled: Counter = Counter()
    values: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        seen = set()
        for path, leaf in flatten(record):
            if leaf in (None, "", [], {}):
                continue
            seen.add(path)
            if len(values[path]) < 500:
                values[path][str(leaf)[:80]] += 1
        filled.update(seen)
    total = max(len(records), 1)
    rows = []
    for path, count in filled.most_common():
        top = values[path].most_common(3)
        rows.append(
            {
                "path": path,
                "filled": count,
                "share": count / total,
                "distinct": len(values[path]),
                "examples": [value for value, _ in top],
            }
        )
    return rows


def keyword_hits(texts: Iterable[str], words: dict[str, str]) -> Counter:
    hits: Counter = Counter()
    for text in texts:
        for name, pattern in words.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                hits[name] += 1
    return hits


def espn_records(payload: dict) -> list[dict]:
    """One record per injured player. The team wrapper is kept as ``team``."""
    out = []
    for team in payload.get("injuries", []) or []:
        name = team.get("displayName") or team.get("name")
        for item in team.get("injuries", []) or []:
            out.append({"team": name, **item})
    return out


def sleeper_injury_records(players: dict) -> list[dict]:
    """Fantasy-position players that carry any injury field."""
    keep = []
    for pid, record in players.items():
        if not isinstance(record, dict):
            continue
        positions = set(record.get("fantasy_positions") or []) | {record.get("position")}
        if not positions & FANTASY_POSITIONS:
            continue
        if any(record.get(key) for key in ("injury_status", "injury_body_part", "injury_notes", "injury_start_date")):
            keep.append({"player_id": pid, **record})
    return keep


def _print_table(title: str, rows: list[dict], limit: int = 40) -> None:
    print(f"\n{title}")
    print(f"  {'field':58s}{'filled':>7s}{'share':>7s}{'distinct':>9s}  examples")
    for row in rows[:limit]:
        examples = "; ".join(row["examples"])[:70]
        print(f"  {row['path'][:58]:58s}{row['filled']:7d}{100 * row['share']:6.0f}%{row['distinct']:9d}  {examples}")


def _texts(records: list[dict], key_filter) -> list[str]:
    out = []
    for record in records:
        for path, leaf in flatten(record):
            if isinstance(leaf, str) and key_filter(path):
                out.append(leaf)
    return out


def report(name: str, records: list[dict], text_filter) -> dict:
    print(f"\n=== {name}: {len(records)} injured player records ===")
    table = field_table(records)
    _print_table("fields served", table)
    texts = _texts(records, text_filter)
    nature = keyword_hits(texts, NATURE_WORDS)
    timeline = keyword_hits(texts, TIMELINE_WORDS)
    print(f"\n  free-text strings scanned: {len(texts)}")
    print("  nature words:  " + (", ".join(f"{k} {v}" for k, v in nature.most_common()) or "none"))
    print("  timeline words: " + (", ".join(f"{k} {v}" for k, v in timeline.most_common()) or "none"))
    longest = sorted(texts, key=len, reverse=True)[:3]
    for text in longest:
        print(f"  e.g. {text[:240]!r}")
    return {"records": len(records), "fields": table, "nature": dict(nature), "timeline": dict(timeline)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/injury_snapshots"))
    parser.add_argument("--skip", nargs="*", default=[], choices=["espn", "sleeper"])
    args = parser.parse_args(argv)

    day = datetime.now(timezone.utc).date().isoformat()
    folder = args.out / day
    folder.mkdir(parents=True, exist_ok=True)
    summary: dict = {"date": day}
    failed = False

    if "espn" not in args.skip:
        try:
            payload = get_json(ESPN_INJURIES)
            (folder / "espn_injuries.json").write_text(json.dumps(payload, sort_keys=True), "utf-8")
            summary["espn"] = report(
                "ESPN",
                espn_records(payload),
                lambda path: any(k in path.lower() for k in ("comment", "detail", "type", "status")),
            )
        except RemoteDataError as exc:
            print(f"ESPN failed: {exc}", file=sys.stderr)
            summary["espn"] = {"error": str(exc)}
            failed = True

    if "sleeper" not in args.skip:
        try:
            players = get_json(SLEEPER_PLAYERS)
            injured = sleeper_injury_records(players)
            (folder / "sleeper_injuries.json").write_text(
                json.dumps(injured, sort_keys=True, separators=(",", ":")), "utf-8"
            )
            summary["sleeper"] = report("Sleeper", injured, lambda path: "injur" in path.lower() or "note" in path.lower())
        except RemoteDataError as exc:
            print(f"Sleeper failed: {exc}", file=sys.stderr)
            summary["sleeper"] = {"error": str(exc)}
            failed = True

    (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    print(f"\nwrote {folder}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
