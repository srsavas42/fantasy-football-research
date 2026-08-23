from __future__ import annotations

import json

import pytest

from ffmodel.release import IdentityOverride, RosterReconciliationError, reconcile_captured_roster, sha256_digest
from ffmodel.features.season_average import PRESEASON_FEATURES, build_projection_data


def _table(rows):
    return json.dumps({"schema": {"fields": []}, "data": rows}, separators=(",", ":")).encode()


def _inputs(*, sleeper_changes=None, nflverse_changes=None):
    sleeper = {
        "s-qb": {"full_name": "Quarter Back", "position": "QB", "team": "SFO", "status": "Active", "gsis_id": "00-0001", "years_exp": 4},
        "s-wr": {"full_name": "Trade Receiver", "position": "WR", "team": "SFO", "status": "ACT", "years_exp": 3},
        "s-rookie": {"full_name": "New Rookie", "position": "WR", "team": "SFO", "status": "Reserve", "years_exp": 0, "rookie_year": 2026},
        "s-k": {"full_name": "Kicker Person", "position": "K", "team": "SFO", "status": "Active"},
        "s-fa": {"full_name": "Free Agent", "position": "RB", "status": "Free Agent", "years_exp": 4},
    }
    nflverse = [
        {"gsis_id": "00-0001", "display_name": "Quarter Back", "position": "QB", "team": "SF", "sleeper_id": "different-qb"},
        {"gsis_id": "00-0002", "display_name": "Trade Receiver", "position": "WR", "team": "SEA", "sleeper_id": "s-wr"},
    ]
    for key, value in (sleeper_changes or {}).items():
        if value is None:
            sleeper.pop(key)
        else:
            sleeper[key] = value
    nflverse = nflverse_changes if nflverse_changes is not None else nflverse
    return json.dumps(sleeper, separators=(",", ":"), sort_keys=True).encode(), _table(nflverse)


def test_reconciliation_precedence_dispositions_and_projection_contract():
    sleeper, nflverse = _inputs()

    result = reconcile_captured_roster(sleeper, nflverse, target_season=2026)

    assert result.players.columns.tolist() == [
        "season", "team", "player_key", "player_id", "player_name", "position",
        "sleeper_id", "nflverse_id", "roster_status", "roster_active", "roster_reserve",
        "age", "experience", "depth_rank", "qb_depth_rank", "qb_listed_starter",
        "roster_snapshot_week", "depth_snapshot_week", "roster_snapshot_source",
        "observed_roster_games", "cold_start",
        "match_method", "match_confidence", "match_evidence",
    ]
    assert result.players.set_index("sleeper_id").loc["s-qb", "match_method"] == "sleeper_exact_gsis"
    assert result.players.set_index("sleeper_id").loc["s-wr", "match_method"] == "nflverse_sleeper_crosswalk"
    rookie = result.players.set_index("sleeper_id").loc["s-rookie"]
    assert rookie["cold_start"] and rookie["player_key"] == "sleeper:s-rookie"
    assert len(result.dispositions) == 5
    assert result.excluded_counts == {"excluded_non_model_position": 1, "excluded_roster_status": 1}
    assert set(result.players["team"]) == {"SF"}
    assert {"season", "team", "player_key", "player_id", "player_name", "position", "age", "experience", "roster_status", "roster_active", "roster_reserve", "depth_rank", "qb_depth_rank", "qb_listed_starter", "roster_snapshot_week", "depth_snapshot_week", "roster_snapshot_source", "observed_roster_games"} <= set(result.players.columns)
    assert result.players["roster_snapshot_source"].eq("sleeper_capture").all()


def test_unique_name_position_crosswalk_allows_a_traded_player_and_team_alias():
    sleeper, nflverse = _inputs()
    raw = json.loads(sleeper)
    raw["s-wr"].pop("full_name")
    raw["s-wr"]["player_name"] = "Trade Receiver"
    raw["s-wr"]["team"] = "SFO"
    raw["s-wr"]["status"] = "Active"
    raw["s-wr"].pop("years_exp")
    raw["s-wr"]["years_exp"] = 2
    table = json.loads(nflverse)
    table["data"][1].pop("sleeper_id")
    out = reconcile_captured_roster(json.dumps(raw, sort_keys=True).encode(), json.dumps(table, separators=(",", ":")).encode(), target_season=2026)
    row = out.players.set_index("sleeper_id").loc["s-wr"]
    assert row["match_method"] == "name_position_crosswalk"
    assert row["team"] == "SF"


@pytest.mark.parametrize(
    "defect, message",
    [
        ({"s-qb": {"full_name": "Quarter Back", "position": "QB", "team": "ZZZ", "status": "Active", "gsis_id": "00-0001", "years_exp": 4}}, "unknown roster team"),
        ({"s-qb": {"full_name": "Quarter Back", "position": "QB", "team": "SFO", "status": "Mystery", "gsis_id": "00-0001", "years_exp": 4}}, "unknown or missing roster status"),
    ],
)
def test_unknown_model_roster_status_or_team_fails_closed(defect, message):
    sleeper, nflverse = _inputs(sleeper_changes=defect)
    with pytest.raises(RosterReconciliationError, match=message):
        reconcile_captured_roster(sleeper, nflverse, target_season=2026)


def test_unresolved_veteran_never_becomes_a_rookie():
    sleeper, nflverse = _inputs()
    raw = json.loads(sleeper)
    raw["s-wr"].pop("years_exp")
    raw["s-wr"]["years_exp"] = 0
    raw["s-wr"].pop("rookie_year", None)
    table = json.loads(nflverse)
    table["data"] = table["data"][:1]
    with pytest.raises(RosterReconciliationError, match="unresolved veteran Sleeper player 's-wr'"):
        reconcile_captured_roster(json.dumps(raw, sort_keys=True).encode(), json.dumps(table).encode(), target_season=2026)


def test_exact_identity_or_override_cannot_cross_positions():
    sleeper, nflverse = _inputs()
    raw = json.loads(sleeper)
    raw["s-qb"]["position"] = "WR"
    with pytest.raises(RosterReconciliationError, match="conflicting nflverse position"):
        reconcile_captured_roster(json.dumps(raw, sort_keys=True).encode(), nflverse, target_season=2026)


def test_global_player_dimension_ignores_noncanonical_history_and_keeps_matched_rookie_cold_start():
    sleeper, nflverse = _inputs()
    raw = json.loads(sleeper)
    raw["s-qb"].update({"years_exp": 0, "rookie_year": 2026})
    table = json.loads(nflverse)
    table["data"].append({"gsis_id": "OLD123", "display_name": "Historical Row", "position": "WR", "team": "SF"})
    result = reconcile_captured_roster(json.dumps(raw, sort_keys=True).encode(), json.dumps(table).encode(), target_season=2026)
    qb = result.players.set_index("sleeper_id").loc["s-qb"]
    assert qb["player_id"] == "00-0001"
    assert bool(qb["cold_start"])
    assert result.dispositions.set_index("sleeper_id").loc["s-qb", "disposition"] == "eligible_rookie_cold_start"


def test_duplicate_or_ambiguous_identity_fails_closed():
    sleeper, nflverse = _inputs(
        sleeper_changes={"s-wr-2": {"full_name": "Trade Receiver", "position": "WR", "team": "SFO", "status": "ACT", "gsis_id": "00-0002", "years_exp": 5}},
    )
    with pytest.raises(RosterReconciliationError, match="duplicate eligible projection player identity"):
        reconcile_captured_roster(sleeper, nflverse, target_season=2026)
    sleeper, nflverse = _inputs(
        sleeper_changes={"s-wr-2": {"full_name": "Trade Receiver", "position": "WR", "team": "SEA", "status": "ACT", "gsis_id": "00-0002", "years_exp": 5}},
    )
    with pytest.raises(RosterReconciliationError, match="multiple eligible teams"):
        reconcile_captured_roster(sleeper, nflverse, target_season=2026)
    ambiguous = [
        {"gsis_id": "00-0001", "display_name": "Quarter Back", "position": "QB", "team": "SF"},
        {"gsis_id": "00-0002", "display_name": "Trade Receiver", "position": "WR", "team": "SEA"},
        {"gsis_id": "00-0003", "display_name": "Trade Receiver", "position": "WR", "team": "LAR"},
    ]
    raw = json.loads(sleeper)
    raw.pop("s-wr-2")
    raw["s-wr"].pop("full_name")
    raw["s-wr"]["player_name"] = "Trade Receiver"
    with pytest.raises(RosterReconciliationError, match="ambiguous nflverse name crosswalk"):
        reconcile_captured_roster(json.dumps(raw, sort_keys=True).encode(), _table(ambiguous), target_season=2026)


def test_missing_qb_is_a_release_blocker():
    sleeper, nflverse = _inputs(sleeper_changes={"s-qb": None})
    with pytest.raises(RosterReconciliationError, match="missing a quarterback"):
        reconcile_captured_roster(sleeper, nflverse, target_season=2026)


def test_digest_bound_override_applies_only_to_exact_payloads():
    sleeper, nflverse = _inputs()
    raw = json.loads(sleeper)
    raw["s-wr"].pop("full_name")
    raw["s-wr"]["player_name"] = "No Match"
    table = json.loads(nflverse)
    table["data"][1].pop("sleeper_id")
    sleeper = json.dumps(raw, sort_keys=True).encode()
    nflverse = json.dumps(table, separators=(",", ":")).encode()
    override = IdentityOverride("s-wr", "00-0002", sha256_digest(sleeper), sha256_digest(nflverse), "reviewed collision")
    result = reconcile_captured_roster(sleeper, nflverse, target_season=2026, overrides=[override])
    assert result.players.set_index("sleeper_id").loc["s-wr", "match_method"] == "digest_bound_override"
    changed = sleeper.replace(b"No Match", b"Different")
    with pytest.raises(RosterReconciliationError, match="not bound to these exact captured payloads"):
        reconcile_captured_roster(changed, nflverse, target_season=2026, overrides=[override])


def test_reconciliation_replay_is_deterministic():
    sleeper, nflverse = _inputs()
    first = reconcile_captured_roster(sleeper, nflverse, target_season=2026)
    raw = json.loads(sleeper)
    reversed_payload = json.dumps(dict(reversed(list(raw.items()))), separators=(",", ":")).encode()
    second = reconcile_captured_roster(reversed_payload, nflverse, target_season=2026)
    assert first.players.to_dict("records") == second.players.to_dict("records")
    assert first.dispositions.to_dict("records") == second.dispositions.to_dict("records")


def test_reconciled_roster_enters_existing_projection_builder():
    sleeper, nflverse = _inputs()
    raw = json.loads(sleeper)
    for row in raw.values():
        if row.get("rookie_year") == 2026:
            row["rookie_year"] = 2021
    roster = reconcile_captured_roster(json.dumps(raw, sort_keys=True).encode(), nflverse, target_season=2021).players
    data = build_projection_data(2021, roster_snapshot=roster, history_seasons=[2018, 2019, 2020], source="legacy")
    assert set(PRESEASON_FEATURES) <= set(data.player_rows.columns)
