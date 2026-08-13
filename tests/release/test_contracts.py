from datetime import datetime, timedelta, timezone

import pytest

from ffmodel.release import (
    APPROVAL_VALIDITY,
    ApprovalDecision,
    ApprovalRecord,
    PlayerPredictionRecord,
    Position,
    PredictionQuantiles,
    ReleaseContract,
    ReleaseContractError,
    ReleaseIdentity,
    ReleaseManifest,
    ReleaseState,
    RankingRecord,
    SamplerMinimum,
    ScoringFormat,
    SchemaValidationError,
    canonical_json,
    target_season_cutoff,
)
from ffmodel.release.schemas import EvidenceRecord

DIGEST = "a" * 64


@pytest.mark.parametrize("year", [2000, 2026, 9999])
def test_cutoff_is_exact_utc_boundary(year):
    assert target_season_cutoff(year) == datetime(year, 8, 31, tzinfo=timezone.utc)


@pytest.mark.parametrize("value", [None, True, "2026", 1999, 10_000])
def test_target_season_is_explicit_and_valid(value):
    with pytest.raises(ReleaseContractError):
        target_season_cutoff(value)


def test_contract_enforces_production_refit_and_sampler_floor():
    identity = ReleaseIdentity(2026, "attempt-01")
    contract = ReleaseContract(identity, (ScoringFormat.STANDARD, ScoringFormat.PPR))
    assert contract.scoring_formats == (ScoringFormat.PPR, ScoringFormat.STANDARD)
    for kwargs in ({"draws": 1999}, {"tune": 1999}, {"chains": 3}):
        with pytest.raises(ReleaseContractError):
            SamplerMinimum(**kwargs)
    with pytest.raises(ReleaseContractError, match="refitting"):
        ReleaseContract(identity, (ScoringFormat.PPR,), artifact_reuse_allowed=True)
    with pytest.raises(ReleaseContractError, match="refitting"):
        ReleaseContract(identity, (ScoringFormat.PPR,), refit_required=False)


def test_states_declare_approval_pause_without_transition_api():
    assert {ReleaseState.AWAITING_APPROVAL, ReleaseState.PUBLISHABLE, ReleaseState.REJECTED} <= set(ReleaseState)
    assert not hasattr(ReleaseState, "transition")


def test_prediction_requires_all_explicit_scoring_formats():
    source_scoring = {ScoringFormat.PPR: PredictionQuantiles(300, 200, 295, 400)}
    prediction = PlayerPredictionRecord("00-003", "Example Player", Position.QB, source_scoring)
    prediction.validate_scoring_formats((ScoringFormat.PPR,))
    assert PlayerPredictionRecord.from_dict(prediction.to_dict()) == prediction
    source_scoring[ScoringFormat.STANDARD] = PredictionQuantiles(250, 150, 245, 350)
    assert set(prediction.scoring) == {ScoringFormat.PPR}
    with pytest.raises(TypeError):
        prediction.scoring[ScoringFormat.STANDARD] = PredictionQuantiles(250, 150, 245, 350)
    with pytest.raises(ReleaseContractError, match="exactly match"):
        prediction.validate_scoring_formats((ScoringFormat.PPR, ScoringFormat.STANDARD))


def test_canonical_json_golden_is_stable():
    assert canonical_json({"z": [3, 2], "a": {"b": 1}}) == '{"a":{"b":1},"z":[3,2]}'


def _approval(**overrides):
    values = dict(release_root=r"C:\Users\example\Documents\releases", target_season=2026, attempt="attempt-01", approver="reviewer@example.test", decision=ApprovalDecision.APPROVE, staged_release_digest=DIGEST, reason="validated against the release checklist", decided_at=datetime(2026, 8, 31, tzinfo=timezone.utc))
    values.update(overrides)
    return ApprovalRecord.create(**values)


def test_approval_binds_utc_digest_and_exact_24_hour_validity():
    approval = _approval()
    assert approval.expires_at == approval.decided_at + APPROVAL_VALIDITY
    assert approval.is_valid_at(approval.decided_at + timedelta(hours=23, minutes=59), staged_release_digest=DIGEST)
    assert not approval.is_valid_at(approval.expires_at, staged_release_digest=DIGEST)
    assert not approval.is_valid_at(approval.decided_at, staged_release_digest="b" * 64)
    assert ApprovalRecord.from_dict(approval.to_dict()) == approval


def test_complete_versioned_approval_artifact_has_stable_canonical_golden():
    assert canonical_json(_approval().to_dict()) == (
        '{"approver":"reviewer@example.test","attempt":"attempt-01",'
        '"decided_at":"2026-08-31T00:00:00Z","decision":"approve",'
        '"expires_at":"2026-09-01T00:00:00Z",'
        '"reason":"validated against the release checklist",'
        '"release_root":"C:\\\\Users\\\\example\\\\Documents\\\\releases",'
        '"schema_version":"release-approval.v1",'
        '"staged_release_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"target_season":2026}'
    )


@pytest.mark.parametrize("reason", ["x" * 1001, r"see C:\\secrets\\approval.txt", r"note:C:\\secrets\\approval.txt", r"see \\server\\share\\approval.txt", "see /tmp/approval.txt", "record, /tmp/approval.txt", "record;/tmp/approval.txt", "record[/tmp/approval.txt]", "record-/tmp/approval.txt", "token=not-a-real-token", "api key=not-a-real-key", "AWS AKIA0123456789ABCDEF key", "github_pat_abcdefghijklmnopqrstuvwxyz1234567890"])
def test_approval_reason_rejects_length_paths_and_secret_looking_text(reason):
    with pytest.raises(ReleaseContractError):
        _approval(reason=reason)


def test_approval_rejects_non_utc_and_non_exact_expiry():
    with pytest.raises(ReleaseContractError, match="UTC"):
        _approval(decided_at=datetime(2026, 8, 31, tzinfo=timezone(timedelta(hours=-4))))
    approval = _approval()
    with pytest.raises(ReleaseContractError, match="exactly 24"):
        ApprovalRecord(**{**approval.__dict__, "expires_at": approval.expires_at + timedelta(seconds=1)})


def test_schema_unknown_fields_fail_closed():
    data = _approval().to_dict(); data["unexpected"] = "no"
    with pytest.raises(SchemaValidationError, match="unknown"):
        ApprovalRecord.from_dict(data)


def test_ranking_schema_round_trip_and_negative_cases_are_strict():
    ranking = RankingRecord("00-003", Position.QB, ScoringFormat.PPR, 1)
    assert RankingRecord.from_dict(ranking.to_dict()) == ranking
    with pytest.raises(ReleaseContractError):
        RankingRecord("00-003", Position.QB, ScoringFormat.PPR, 0)
    data = ranking.to_dict(); data["unexpected"] = "no"
    with pytest.raises(SchemaValidationError, match="unknown"):
        RankingRecord.from_dict(data)
    data = ranking.to_dict(); data["position"] = "K"
    with pytest.raises(SchemaValidationError):
        RankingRecord.from_dict(data)


@pytest.mark.parametrize("values", [(1, 2, 1, 3), (float("nan"), 1, 2, 3)])
def test_quantiles_reject_unordered_or_nonfinite_values(values):
    with pytest.raises(ReleaseContractError):
        PredictionQuantiles(*values)


def test_evidence_schema_round_trip_unknown_fields_and_version_are_strict():
    record = EvidenceRecord("runtime-input", DIGEST, "sleeper", datetime(2026, 8, 30, tzinfo=timezone.utc))
    assert EvidenceRecord.from_dict(record.to_dict()) == record
    data = record.to_dict(); data["unexpected"] = "no"
    with pytest.raises(SchemaValidationError, match="unknown"):
        EvidenceRecord.from_dict(data)
    with pytest.raises(ReleaseContractError, match="schema_version"):
        EvidenceRecord("runtime-input", DIGEST, "sleeper", datetime(2026, 8, 30, tzinfo=timezone.utc), "release-evidence.v2")


def test_manifest_round_trip_and_unknown_fields_fail_closed():
    identity = ReleaseIdentity(2026, "attempt-01")
    source_evidence = [EvidenceRecord("runtime-input", DIGEST, "sleeper", datetime(2026, 8, 30, tzinfo=timezone.utc))]
    manifest = ReleaseManifest(identity, ReleaseContract(identity, (ScoringFormat.PPR,)), source_evidence)
    source_evidence.clear()
    assert len(manifest.evidence) == 1
    assert ReleaseManifest.from_dict(manifest.to_dict()) == manifest
    data = manifest.to_dict(); data["unknown"] = True
    with pytest.raises(SchemaValidationError, match="unknown"):
        ReleaseManifest.from_dict(data)


def test_package_import_smoke():
    import ffmodel.release
    assert ffmodel.release.APPROVAL_SCHEMA_VERSION == "release-approval.v1"
