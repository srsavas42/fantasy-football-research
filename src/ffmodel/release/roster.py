"""Fail-closed Sleeper-to-nflverse target-roster reconciliation.

This module deliberately accepts captured source *bytes*, rather than provider
clients or data frames.  It is therefore safe to call only after T02 has
verified a source manifest and replayed its immutable inputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

import pandas as pd

from ffmodel.data.identity import is_gsis_id, normalize_player_name
from ffmodel.data.wikipedia_coaching import team_identity
from ffmodel.features.volume import MODEL_POSITIONS, opportunity_position

from .errors import ReleaseContractError
from .schema import sha256_digest


class RosterReconciliationError(ReleaseContractError):
    """Captured roster identities cannot safely form a release roster."""


class RosterDisposition(str, Enum):
    ELIGIBLE_VETERAN = "eligible_veteran"
    ELIGIBLE_ROOKIE = "eligible_rookie_cold_start"
    EXCLUDED_POSITION = "excluded_non_model_position"
    EXCLUDED_STATUS = "excluded_roster_status"


@dataclass(frozen=True)
class IdentityOverride:
    """A manual mapping usable only for one exact pair of captured inputs."""

    sleeper_id: str
    nflverse_id: str
    sleeper_payload_digest: str
    nflverse_payload_digest: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("sleeper_id", "nflverse_id", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RosterReconciliationError(f"override {name} must be a non-empty string")
        for name in ("sleeper_payload_digest", "nflverse_payload_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise RosterReconciliationError(f"override {name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ReconciledRoster:
    """Projection-ready roster and one auditable disposition per Sleeper row."""

    players: pd.DataFrame
    dispositions: pd.DataFrame
    sleeper_payload_digest: str
    nflverse_payload_digest: str

    @property
    def excluded_counts(self) -> dict[str, int]:
        excluded = self.dispositions[~self.dispositions["eligible"]]
        return {
            str(key): int(value)
            for key, value in excluded.groupby("disposition", sort=True).size().items()
        }


_ELIGIBLE_STATUSES = frozenset({"ACT", "ACTIVE", "INA", "INACTIVE", "RES", "RESERVE", "IR", "INJURED_RESERVE", "INJURED RESERVE", "PUP", "NFI", "EXE", "EXEMPT"})
_EXCLUDED_STATUSES = frozenset({"CUT", "RETIRED", "FREE_AGENT", "FREE AGENT", "SUSPENDED", "WAIVED", "DELETED", "PRACTICE_SQUAD", "PRACTICE SQUAD"})
_OUTPUT_COLUMNS = (
    "season", "team", "player_key", "player_id", "player_name", "position",
    "sleeper_id", "nflverse_id", "roster_status", "roster_active", "roster_reserve",
    "age", "experience", "depth_rank", "qb_depth_rank", "qb_listed_starter",
    "roster_snapshot_week", "depth_snapshot_week", "roster_snapshot_source",
    "observed_roster_games", "cold_start",
    "match_method", "match_confidence", "match_evidence",
)


def reconcile_captured_roster(
    sleeper_players_payload: bytes,
    nflverse_players_payload: bytes,
    *,
    target_season: int,
    overrides: Iterable[IdentityOverride] = (),
) -> ReconciledRoster:
    """Reconcile captured Sleeper players to a deterministic projection roster.

    Precedence is a digest-bound manual override, Sleeper's exact GSIS id,
    nflverse's exact Sleeper crosswalk, then an unambiguous normalized
    name/position/team crosswalk.  An unresolved non-rookie never degrades to
    a cold start; it aborts the release with its Sleeper identity in the error.
    """
    if isinstance(target_season, bool) or not isinstance(target_season, int) or not 2000 <= target_season <= 9999:
        raise RosterReconciliationError("target_season must be an explicit year between 2000 and 9999")
    sleeper = _decode_sleeper(sleeper_players_payload)
    nflverse = _decode_nflverse(nflverse_players_payload)
    sleeper_digest = sha256_digest(sleeper_players_payload)
    nflverse_digest = sha256_digest(nflverse_players_payload)
    candidate = _candidate_dimension(nflverse, target_season)
    override_map = _validated_overrides(overrides, sleeper_digest, nflverse_digest, candidate)
    unknown_override_ids = sorted(set(override_map) - set(sleeper))
    if unknown_override_ids:
        raise RosterReconciliationError(f"overrides reference Sleeper identities absent from captured payload: {unknown_override_ids}")
    candidate_by_id = candidate.set_index("nflverse_id", drop=False)

    audit: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for sleeper_id in sorted(sleeper):
        raw = sleeper[sleeper_id]
        if not isinstance(raw, Mapping):
            raise RosterReconciliationError(f"Sleeper player {sleeper_id!r} is not an object")
        if sleeper_id in seen_ids:
            raise RosterReconciliationError(f"duplicate Sleeper identity {sleeper_id!r}")
        seen_ids.add(sleeper_id)
        row = dict(raw)
        raw_position = _text(row.get("position"))
        position = _canonical_position(raw_position)
        name = _first_text(row, "full_name", "player_name", "display_name", "first_name")
        status = _status(row)
        base = {"sleeper_id": sleeper_id, "player_name": name, "source_position": raw_position, "position": position, "roster_status": status}
        if position not in MODEL_POSITIONS:
            audit.append({**base, "eligible": False, "disposition": RosterDisposition.EXCLUDED_POSITION.value, "nflverse_id": None, "match_method": None, "match_confidence": None, "match_evidence": "position outside QB/RB/WR/TE"})
            continue
        if status in _EXCLUDED_STATUSES:
            audit.append({**base, "eligible": False, "disposition": RosterDisposition.EXCLUDED_STATUS.value, "nflverse_id": None, "match_method": None, "match_confidence": None, "match_evidence": f"recognized excluded status {status}"})
            continue
        if status not in _ELIGIBLE_STATUSES:
            raise RosterReconciliationError(f"Sleeper player {sleeper_id!r} has unknown or missing roster status {status!r}")
        if not name:
            raise RosterReconciliationError(f"Sleeper player {sleeper_id!r} is missing a player name")
        team = _canonical_team(_first_text(row, "team", "team_abbr"), target_season, sleeper_id)
        match = _resolve(row, sleeper_id, position, team, candidate, candidate_by_id, override_map)
        rookie = _is_explicit_rookie(row, target_season)
        if match is None:
            if rookie:
                nflverse_id = None
                method, confidence, evidence = "eligible_rookie_cold_start", "explicit", "years_exp=0 and rookie_year equals target season"
            else:
                raise RosterReconciliationError(f"unresolved veteran Sleeper player {sleeper_id!r} ({name}, {position}, {team}); add an exact digest-bound override or correct captured identities")
        else:
            nflverse_id, method, confidence, evidence = match
        disposition = RosterDisposition.ELIGIBLE_ROOKIE if rookie else RosterDisposition.ELIGIBLE_VETERAN
        cold_start = rookie
        player_key = nflverse_id if nflverse_id else f"sleeper:{sleeper_id}"
        roster_status = _projection_roster_status(status)
        depth_rank = _number(row.get("depth_chart_order"))
        output = {"season": target_season, "team": team, "player_key": player_key, "player_id": nflverse_id, "player_name": name, "position": position, "sleeper_id": sleeper_id, "nflverse_id": nflverse_id, "roster_status": roster_status, "roster_active": int(roster_status == "ACT"), "roster_reserve": int(roster_status in {"RES", "INA", "EXE"}), "age": _age_on_september_first(row.get("birth_date"), target_season), "experience": _experience(row), "depth_rank": depth_rank, "qb_depth_rank": depth_rank if position == "QB" else None, "qb_listed_starter": int(position == "QB" and depth_rank == 1), "roster_snapshot_week": None, "depth_snapshot_week": None, "roster_snapshot_source": "sleeper_capture", "observed_roster_games": None, "cold_start": cold_start, "match_method": method, "match_confidence": confidence, "match_evidence": evidence}
        eligible.append(output)
        audit.append({**base, "team": team, "eligible": True, "disposition": disposition.value, "nflverse_id": nflverse_id, "match_method": method, "match_confidence": confidence, "match_evidence": evidence})
    _validate_eligible(eligible)
    players = pd.DataFrame(eligible, columns=_OUTPUT_COLUMNS).sort_values(["team", "position", "player_key"], kind="stable").reset_index(drop=True)
    dispositions = pd.DataFrame(audit).sort_values("sleeper_id", kind="stable").reset_index(drop=True)
    return ReconciledRoster(players, dispositions, sleeper_digest, nflverse_digest)


def _decode_sleeper(payload: bytes) -> dict[str, Mapping[str, Any]]:
    if not isinstance(payload, bytes) or not payload:
        raise RosterReconciliationError("Sleeper captured payload must be non-empty bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RosterReconciliationError("Sleeper captured payload is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or not value:
        raise RosterReconciliationError("Sleeper captured payload must be a non-empty object keyed by Sleeper IDs")
    if any(not isinstance(key, str) or not key for key in value):
        raise RosterReconciliationError("Sleeper captured payload contains an invalid Sleeper identity")
    return value


def _decode_nflverse(payload: bytes) -> pd.DataFrame:
    if not isinstance(payload, bytes) or not payload:
        raise RosterReconciliationError("nflverse captured payload must be non-empty bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RosterReconciliationError("nflverse captured payload is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("data"), list) or not isinstance(value.get("schema"), dict):
        raise RosterReconciliationError("nflverse captured payload must be the captured JSON-table form")
    frame = pd.DataFrame(value["data"])
    if frame.empty:
        raise RosterReconciliationError("nflverse player dimension is empty")
    return frame


def _candidate_dimension(frame: pd.DataFrame, target_season: int) -> pd.DataFrame:
    out = frame.copy()
    id_column = _first_column(out, "gsis_id", "player_id")
    name_column = _first_column(out, "display_name", "full_name", "player_name", "football_name")
    if id_column is None or name_column is None:
        raise RosterReconciliationError("nflverse player dimension must include a GSIS/player id and player name")
    out["nflverse_id"] = out[id_column].map(_text)
    out["nflverse_name"] = out[name_column].map(_text)
    if out["nflverse_id"].isna().any() or out["nflverse_name"].isna().any():
        raise RosterReconciliationError("nflverse player dimension contains a missing identity or player name")
    # nflverse's global player dimension retains historical/provider-native
    # rows alongside current canonical GSIS records.  They are not valid join
    # targets, but their routine presence must not reject the captured source.
    out = out[is_gsis_id(out["nflverse_id"])].copy()
    if out.empty:
        raise RosterReconciliationError("nflverse player dimension contains no canonical GSIS identities")
    if out["nflverse_id"].duplicated().any():
        bad = sorted(out.loc[out["nflverse_id"].duplicated(keep=False), "nflverse_id"].unique())
        raise RosterReconciliationError(f"duplicate nflverse identities: {bad}")
    out["nflverse_position"] = _series_position(out, "position")
    team_column = _first_column(out, "team", "team_abbr", "club_code", "latest_team")
    out["nflverse_team"] = out[team_column].map(lambda value: _canonical_team_optional(value, target_season)) if team_column else None
    sleeper_column = _first_column(out, "sleeper_id", "sleeper_player_id")
    out["nflverse_sleeper_id"] = out[sleeper_column].map(_text) if sleeper_column else None
    if sleeper_column and out.loc[out["nflverse_sleeper_id"].notna(), "nflverse_sleeper_id"].duplicated().any():
        raise RosterReconciliationError("nflverse player dimension maps a Sleeper identity more than once")
    out["name_key"] = normalize_player_name(out["nflverse_name"])
    return out


def _resolve(row: Mapping[str, Any], sleeper_id: str, position: str, team: str, candidate: pd.DataFrame, candidate_by_id: pd.DataFrame, overrides: Mapping[str, IdentityOverride]) -> tuple[str, str, str, str] | None:
    override = overrides.get(sleeper_id)
    if override:
        _require_position_match(candidate_by_id.loc[override.nflverse_id], sleeper_id, position, "digest-bound override")
        return override.nflverse_id, "digest_bound_override", "manual_exact", override.reason
    gsis = _first_text(row, "gsis_id")
    if gsis and gsis in candidate_by_id.index:
        _require_position_match(candidate_by_id.loc[gsis], sleeper_id, position, "Sleeper GSIS")
        return gsis, "sleeper_exact_gsis", "exact", "Sleeper gsis_id exactly present in captured nflverse dimension"
    direct = candidate[candidate["nflverse_sleeper_id"].eq(sleeper_id)]
    if len(direct) == 1:
        _require_position_match(direct.iloc[0], sleeper_id, position, "nflverse Sleeper crosswalk")
        return str(direct.iloc[0]["nflverse_id"]), "nflverse_sleeper_crosswalk", "exact", "captured nflverse Sleeper identifier"
    if len(direct) > 1:
        raise RosterReconciliationError(f"ambiguous nflverse Sleeper crosswalk for {sleeper_id!r}")
    name = _first_text(row, "full_name", "player_name", "display_name", "first_name")
    key = normalize_player_name(pd.Series([name])).iloc[0]
    named = candidate[candidate["name_key"].eq(key) & candidate["nflverse_position"].eq(position)]
    by_team = named[named["nflverse_team"].eq(team)]
    if len(by_team) == 1:
        return str(by_team.iloc[0]["nflverse_id"]), "name_position_team_crosswalk", "deterministic", "unique normalized name, position, and team"
    if len(by_team) > 1 or len(named) > 1:
        raise RosterReconciliationError(f"ambiguous nflverse name crosswalk for Sleeper player {sleeper_id!r}")
    if len(named) == 1:
        return str(named.iloc[0]["nflverse_id"]), "name_position_crosswalk", "deterministic", "unique normalized name and position; Sleeper roster team retained"
    return None


def _require_position_match(candidate: pd.Series, sleeper_id: str, position: str, method: str) -> None:
    if str(candidate["nflverse_position"]) != position:
        raise RosterReconciliationError(
            f"{method} resolves Sleeper player {sleeper_id!r} to a conflicting nflverse position"
        )


def _validated_overrides(overrides: Iterable[IdentityOverride], sleeper_digest: str, nflverse_digest: str, dimension: pd.DataFrame) -> dict[str, IdentityOverride]:
    result: dict[str, IdentityOverride] = {}
    known_ids = set(dimension["nflverse_id"])
    for override in overrides:
        if not isinstance(override, IdentityOverride):
            raise RosterReconciliationError("roster overrides must use IdentityOverride")
        if override.sleeper_payload_digest != sleeper_digest or override.nflverse_payload_digest != nflverse_digest:
            raise RosterReconciliationError(f"override for Sleeper player {override.sleeper_id!r} is not bound to these exact captured payloads")
        if override.sleeper_id in result:
            raise RosterReconciliationError(f"duplicate override for Sleeper player {override.sleeper_id!r}")
        if override.nflverse_id not in known_ids:
            raise RosterReconciliationError(f"override nflverse identity {override.nflverse_id!r} is absent from captured player dimension")
        result[override.sleeper_id] = override
    return result


def _validate_eligible(players: list[dict[str, Any]]) -> None:
    if not players:
        raise RosterReconciliationError("captured Sleeper source produced no eligible QB/RB/WR/TE roster entries")
    roster = pd.DataFrame(players)
    if roster["sleeper_id"].duplicated().any():
        raise RosterReconciliationError("duplicate eligible Sleeper identity")
    duplicate_keys = roster[roster["player_key"].duplicated(keep=False)]
    if not duplicate_keys.empty:
        if duplicate_keys.groupby("player_key")["team"].nunique().gt(1).any():
            raise RosterReconciliationError("player appears on multiple eligible teams")
        raise RosterReconciliationError("duplicate eligible projection player identity")
    missing_qb = sorted(set(roster["team"]) - set(roster.loc[roster["position"].eq("QB"), "team"]))
    if missing_qb:
        raise RosterReconciliationError(f"eligible teams are missing a quarterback: {missing_qb}")


def _is_explicit_rookie(row: Mapping[str, Any], target_season: int) -> bool:
    return _integer(row.get("years_exp")) == 0 and _integer(row.get("rookie_year")) == target_season


def _experience(row: Mapping[str, Any]) -> int | None:
    value = _integer(row.get("years_exp"))
    return value if value is not None and value >= 0 else None


def _status(row: Mapping[str, Any]) -> str:
    value = _first_text(row, "status", "roster_status")
    return value.upper().replace("-", "_") if value else ""


def _projection_roster_status(status: str) -> str:
    if status in {"ACT", "ACTIVE"}:
        return "ACT"
    if status in {"INA", "INACTIVE"}:
        return "INA"
    if status in {"EXE", "EXEMPT"}:
        return "EXE"
    return "RES"


def _canonical_position(value: str | None) -> str:
    if not value:
        return "OTHER"
    return str(opportunity_position(pd.Series([value])).iloc[0])


def _series_position(frame: pd.DataFrame, column: str) -> pd.Series:
    return opportunity_position(frame[column]) if column in frame else pd.Series("OTHER", index=frame.index, dtype="string")


def _canonical_team(value: str | None, season: int, sleeper_id: str) -> str:
    if not value:
        raise RosterReconciliationError(f"Sleeper player {sleeper_id!r} is missing a roster team")
    try:
        return team_identity(value, season).franchise_code
    except (KeyError, TypeError, ValueError) as exc:
        raise RosterReconciliationError(f"Sleeper player {sleeper_id!r} has unknown roster team {value!r}") from exc


def _canonical_team_optional(value: Any, season: int) -> str | None:
    text = _text(value)
    if not text:
        return None
    try:
        return team_identity(text, season).franchise_code
    except (KeyError, TypeError, ValueError) as exc:
        raise RosterReconciliationError(f"nflverse player dimension has unknown team {text!r}") from exc


def _first_column(frame: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _first_text(row: Mapping[str, Any], *names: str) -> str | None:
    return next((text for name in names if (text := _text(row.get(name)))), None)


def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if str(value).strip() == str(integer) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _age_on_september_first(value: Any, target_season: int) -> float | None:
    text = _text(value)
    if not text:
        return None
    birth_date = pd.to_datetime(text, errors="coerce")
    if pd.isna(birth_date):
        return None
    return round((pd.Timestamp(target_season, 9, 1) - birth_date).days / 365.25, 6)
