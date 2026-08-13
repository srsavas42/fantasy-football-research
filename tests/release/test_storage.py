from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from ffmodel.release import (
    ApprovalDecision,
    ApprovalRecord,
    LockConflictError,
    ReleaseIdentity,
    ReleaseState,
    ReleaseStore,
    StaleLockError,
    StorageError,
)


DIGEST = "a" * 64


def _store(tmp_path, attempt="attempt-01"):
    store = ReleaseStore(tmp_path / "Documents" / "releases", ReleaseIdentity(2026, attempt), stale_lock_age=timedelta(seconds=1))
    assert store.initialize() is ReleaseState.CAPTURED
    return store


def _awaiting_approval(store):
    for state in (ReleaseState.BOUND, ReleaseState.FITTED, ReleaseState.VALIDATED):
        store.transition(state)
    (store.stage_dir / "outputs").mkdir(parents=True)
    (store.stage_dir / "outputs" / "ranking.json").write_text('{"ok":true}\n', encoding="utf-8", newline="\n")
    sealed = store.seal_staging()
    store.transition(ReleaseState.AWAITING_APPROVAL)
    return sealed


def _approval(store, digest, *, decision=ApprovalDecision.APPROVE, decided_at=None):
    return ApprovalRecord.create(
        release_root=str(store.root), target_season=2026, attempt=store.identity.attempt,
        approver="reviewer@example.test", decision=decision, staged_release_digest=digest,
        reason="reviewed release checklist", decided_at=decided_at or datetime.now(timezone.utc),
    )


def test_lifecycle_is_durable_monotonic_and_requires_approval_pause(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(StorageError, match="intent-specific"):
        store.transition(ReleaseState.PUBLISHABLE)
    sealed = _awaiting_approval(store)
    assert store.state() is ReleaseState.AWAITING_APPROVAL
    assert sealed.digest == store.sealed_package().digest
    with pytest.raises(StorageError, match="only a publishable"):
        store.publish()
    saved = ReleaseStore(store.root, store.identity)
    assert saved.state() is ReleaseState.AWAITING_APPROVAL
    journal = [json.loads(line) for line in store._journal_path.read_text(encoding="utf-8").splitlines()]
    assert any(item["kind"] == "transition" and item["to_state"] == "AWAITING_APPROVAL" for item in journal)


def test_generic_transition_cannot_bypass_approval_or_publish_gates(tmp_path):
    store = _store(tmp_path)
    for state in (ReleaseState.BOUND, ReleaseState.FITTED, ReleaseState.VALIDATED):
        store.transition(state)
    with pytest.raises(StorageError, match="intent-specific"):
        store.transition(ReleaseState.PUBLISHABLE)
    assert store.state() is ReleaseState.VALIDATED


def test_transition_and_approval_partial_commits_fail_closed_then_recover(tmp_path, monkeypatch):
    store = _store(tmp_path)
    original = store._write_state
    def fail_bound(state):
        if state is ReleaseState.BOUND:
            raise OSError("simulated state failure")
        original(state)
    monkeypatch.setattr(store, "_write_state", fail_bound)
    with pytest.raises(OSError, match="simulated state"):
        store.transition(ReleaseState.BOUND)
    with pytest.raises(StorageError, match="transaction recovery"):
        store.state()
    monkeypatch.setattr(store, "_write_state", original)
    assert store.recover() is None
    assert store.state() is ReleaseState.BOUND

    store.transition(ReleaseState.FITTED); store.transition(ReleaseState.VALIDATED)
    (store.stage_dir / "out").mkdir(parents=True); (store.stage_dir / "out" / "x.json").write_text("{}", encoding="utf-8")
    sealed = store.seal_staging(); store.transition(ReleaseState.AWAITING_APPROVAL)
    def fail_publishable(state):
        if state is ReleaseState.PUBLISHABLE:
            raise OSError("simulated approval state failure")
        original(state)
    monkeypatch.setattr(store, "_write_state", fail_publishable)
    with pytest.raises(OSError, match="simulated approval"):
        store.record_approval(_approval(store, sealed.digest))
    with pytest.raises(StorageError, match="transaction recovery"):
        store.state()
    monkeypatch.setattr(store, "_write_state", original)
    store.recover()
    assert store.state() is ReleaseState.PUBLISHABLE


def test_initialize_and_publication_event_failures_remain_recoverable(tmp_path, monkeypatch):
    store = ReleaseStore(tmp_path / "root", ReleaseIdentity(2026, "attempt-01"))
    store.attempt_dir.mkdir(parents=True)
    original_event = store._event
    monkeypatch.setattr(store, "_event", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("event failure")))
    with pytest.raises(OSError, match="event failure"):
        store._commit_mutation(ReleaseState.CAPTURED, "initialized", {"state": "CAPTURED"})
    with pytest.raises(StorageError, match="transaction recovery"):
        store.state()
    monkeypatch.setattr(store, "_event", original_event)
    store.recover()
    assert store.state() is ReleaseState.CAPTURED


def test_approve_is_exact_digest_and_24_hour_bound(tmp_path):
    store = _store(tmp_path)
    sealed = _awaiting_approval(store)
    with pytest.raises(StorageError, match="digest"):
        store.record_approval(_approval(store, DIGEST))
    expired = _approval(store, sealed.digest, decided_at=datetime.now(timezone.utc) - timedelta(hours=24))
    with pytest.raises(StorageError, match="expired"):
        store.record_approval(expired)
    assert store.record_approval(_approval(store, sealed.digest)) is ReleaseState.PUBLISHABLE
    assert store.state() is ReleaseState.PUBLISHABLE


def test_rejection_is_terminal_and_never_mutates_current_pointer(tmp_path):
    store = _store(tmp_path)
    sealed = _awaiting_approval(store)
    prior = store.root / "current-release.json"
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text('{"prior":true}', encoding="utf-8")
    assert store.record_approval(_approval(store, sealed.digest, decision=ApprovalDecision.REJECT)) is ReleaseState.REJECTED
    assert prior.read_text(encoding="utf-8") == '{"prior":true}'
    with pytest.raises(StorageError, match="only a publishable"):
        store.publish()


def test_seal_detects_mutation_and_publication_atomically_replaces_pointer(tmp_path):
    store = _store(tmp_path)
    sealed = _awaiting_approval(store)
    (store.stage_dir / "outputs" / "ranking.json").write_text('{"changed":true}\n', encoding="utf-8")
    assert store.record_approval(_approval(store, sealed.digest)) is ReleaseState.PUBLISHABLE
    previous = store.root / "current-release.json"
    previous.write_text('{"previous":true}', encoding="utf-8")
    with pytest.raises(StorageError, match="changed after sealing"):
        store.publish()
    assert previous.read_text(encoding="utf-8") == '{"previous":true}'


def test_resealing_after_approval_is_rejected_and_cannot_change_approved_bytes(tmp_path):
    store = _store(tmp_path)
    sealed = _awaiting_approval(store)
    store.record_approval(_approval(store, sealed.digest))
    (store.stage_dir / "outputs" / "ranking.json").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(StorageError, match="only be sealed"):
        store.seal_staging()
    with pytest.raises(StorageError, match="changed after sealing"):
        store.publish()


def test_publish_revalidates_approval_and_recovers_after_pointer_replace_failure(tmp_path, monkeypatch):
    store = _store(tmp_path)
    sealed = _awaiting_approval(store)
    decided_at = datetime.now(timezone.utc)
    store.record_approval(_approval(store, sealed.digest, decided_at=decided_at))
    import ffmodel.release.storage as storage
    original = storage._atomic_write
    def fail_pointer(path, data):
        if path.name == "current-release.json":
            raise OSError("simulated replacement failure")
        original(path, data)
    monkeypatch.setattr(storage, "_atomic_write", fail_pointer)
    with pytest.raises(OSError, match="simulated"):
        store.publish()
    assert not (store.root / "current-release.json").exists()
    monkeypatch.setattr(storage, "_atomic_write", original)
    assert store.publish().is_dir()

    expired = _store(tmp_path / "expired")
    expired_seal = _awaiting_approval(expired)
    old = datetime.now(timezone.utc) - timedelta(hours=23, minutes=59)
    expired.record_approval(_approval(expired, expired_seal.digest, decided_at=old))
    with pytest.raises(StorageError, match="stored approval is expired"):
        expired.publish(now=old + timedelta(hours=24))


def test_recover_publication_commits_pointer_swap_interrupted_before_state(tmp_path, monkeypatch):
    store = _store(tmp_path)
    sealed = _awaiting_approval(store)
    store.record_approval(_approval(store, sealed.digest))
    original = store._write_state
    def fail_published(state):
        if state is ReleaseState.PUBLISHED:
            raise OSError("simulated state commit failure")
        original(state)
    monkeypatch.setattr(store, "_write_state", fail_published)
    with pytest.raises(OSError, match="simulated state"):
        store.publish()
    with pytest.raises(StorageError, match="recovery is required"):
        store.state()
    assert (store.root / "current-release.json").exists()
    monkeypatch.setattr(store, "_write_state", original)
    assert store.recover_publication().is_dir()
    assert store.state() is ReleaseState.PUBLISHED
    assert not store._publication_path.exists()


def test_publish_event_failure_drains_both_markers_before_retry_success(tmp_path, monkeypatch):
    store = _store(tmp_path)
    sealed = _awaiting_approval(store)
    store.record_approval(_approval(store, sealed.digest))
    original_event = store._event
    def fail_published(kind, **facts):
        if kind == "published":
            raise OSError("simulated published event failure")
        original_event(kind, **facts)
    monkeypatch.setattr(store, "_event", fail_published)
    with pytest.raises(OSError, match="simulated published"):
        store.publish()
    assert store._publication_path.exists() and store._mutation_path.exists()
    monkeypatch.setattr(store, "_event", original_event)
    assert store.publish().is_dir()
    assert store.state() is ReleaseState.PUBLISHED
    assert not store._publication_path.exists() and not store._mutation_path.exists()


def test_public_recover_publication_drains_final_event_transaction(tmp_path, monkeypatch):
    store = _store(tmp_path)
    sealed = _awaiting_approval(store)
    store.record_approval(_approval(store, sealed.digest))
    original_event = store._event
    monkeypatch.setattr(store, "_event", lambda kind, **facts: (_ for _ in ()).throw(OSError("published event failure")) if kind == "published" else original_event(kind, **facts))
    with pytest.raises(OSError, match="published event"):
        store.publish()
    monkeypatch.setattr(store, "_event", original_event)
    assert store.recover_publication().is_dir()
    assert store.state() is ReleaseState.PUBLISHED
    assert not store._publication_path.exists() and not store._mutation_path.exists()


def test_publish_uses_sealed_copy_and_stable_current_pointer(tmp_path):
    store = _store(tmp_path)
    sealed = _awaiting_approval(store)
    store.record_approval(_approval(store, sealed.digest))
    destination = store.publish()
    pointer = json.loads((store.root / "current-release.json").read_text(encoding="utf-8"))
    assert destination.is_dir()
    assert pointer["digest"] == sealed.digest
    assert pointer["published_path"] == destination.name
    assert store.state() is ReleaseState.PUBLISHED


def test_lock_conflict_and_stale_lock_fail_closed_with_evidence(tmp_path):
    store = _store(tmp_path)
    now = datetime.now(timezone.utc)
    store._lock_path.write_text(json.dumps({"pid": 123, "created_at": now.isoformat().replace("+00:00", "Z")}), encoding="utf-8")
    with pytest.raises(LockConflictError):
        with store.mutation_lock():
            pass
    store._lock_path.write_text(json.dumps({"pid": 123, "created_at": (now - timedelta(seconds=2)).isoformat().replace("+00:00", "Z")}), encoding="utf-8")
    with pytest.raises(StaleLockError):
        with store.mutation_lock():
            pass
    events = [json.loads(line)["kind"] for line in store._journal_path.read_text(encoding="utf-8").splitlines()]
    assert "lock_conflict" in events and "stale_lock_detected" in events


def test_unsafe_attempt_and_mismatched_approval_root_fail_closed(tmp_path):
    with pytest.raises(StorageError, match="safe path"):
        ReleaseStore(tmp_path, ReleaseIdentity(2026, "../escape"))
    store = _store(tmp_path)
    sealed = _awaiting_approval(store)
    approval = ApprovalRecord.create(release_root=str(tmp_path / "different"), target_season=2026, attempt=store.identity.attempt, approver="reviewer", decision=ApprovalDecision.APPROVE, staged_release_digest=sealed.digest, reason="safe reason", decided_at=datetime.now(timezone.utc))
    with pytest.raises(StorageError, match="release root"):
        store.record_approval(approval)
