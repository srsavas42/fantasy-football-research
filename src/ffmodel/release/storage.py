"""Durable local storage for sealed annual prediction releases.

This module is deliberately the only owner of release-state mutation and of
the current-release pointer.  It performs no network, model, or CLI work.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePath
from typing import Any, Iterator
from uuid import uuid4

from .contract import ReleaseIdentity, ReleaseState
from .lifecycle import LifecycleError, validate_transition
from .schema import canonical_json_bytes, datetime_from_wire, datetime_to_wire, require_exact_fields, require_mapping, sha256_digest
from .schemas import ApprovalDecision, ApprovalRecord


STATE_SCHEMA_VERSION = "release-state.v1"
SEAL_SCHEMA_VERSION = "release-seal.v1"
POINTER_SCHEMA_VERSION = "release-current-pointer.v1"
DEFAULT_STALE_LOCK_AGE = timedelta(hours=2)


class StorageError(LifecycleError):
    """Raised when durable storage cannot prove a safe operation."""


class LockConflictError(StorageError):
    """Raised while an active attempt mutation lock exists."""


class StaleLockError(StorageError):
    """Raised for a stale lock; operators must resolve it explicitly."""


@dataclass(frozen=True)
class SealedPackage:
    digest: str
    files: tuple[tuple[str, str], ...]
    sealed_at: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _append_json_line(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _safe_component(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or any(part in {".", ".."} for part in PurePath(value).parts) or any(char in value for char in "\\/\x00"):
        raise StorageError(f"{name} is not a safe path component")
    return value


def _tree_digest(directory: Path) -> tuple[str, tuple[tuple[str, str], ...]]:
    if not directory.is_dir() or directory.is_symlink():
        raise StorageError("staged package must be a real directory")
    files: list[tuple[str, str]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink() or not path.is_file():
            if path.is_dir():
                continue
            raise StorageError(f"staged package contains an unsafe path: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append((relative, digest.hexdigest()))
    if not files:
        raise StorageError("staged package must not be empty")
    return sha256_digest(canonical_json_bytes({"files": files})), tuple(files)


class ReleaseStore:
    """Attempt-scoped storage authority below one local release root."""

    def __init__(self, release_root: str | Path, identity: ReleaseIdentity, *, stale_lock_age: timedelta = DEFAULT_STALE_LOCK_AGE) -> None:
        if not isinstance(identity, ReleaseIdentity):
            raise StorageError("identity must be a ReleaseIdentity")
        if not isinstance(stale_lock_age, timedelta) or stale_lock_age <= timedelta(0):
            raise StorageError("stale_lock_age must be positive")
        self.root = Path(release_root).expanduser().resolve()
        self.identity = identity
        self.stale_lock_age = stale_lock_age
        attempt = _safe_component(identity.attempt, "attempt")
        self.attempt_dir = self.root / "attempts" / str(identity.target_season) / attempt
        self.stage_dir = self.attempt_dir / "staging"
        self._state_path = self.attempt_dir / "state.json"
        self._seal_path = self.attempt_dir / "sealed-package.json"
        self._approval_path = self.attempt_dir / "approval.json"
        self._mutation_path = self.attempt_dir / "mutation-pending.json"
        self._publication_path = self.attempt_dir / "publication-prepared.json"
        self._journal_path = self.attempt_dir / "evidence.jsonl"
        self._lock_path = self.attempt_dir / ".mutation.lock"

    def initialize(self) -> ReleaseState:
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
        with self.mutation_lock():
            if self._state_path.exists():
                return self.state()
            self._commit_mutation(ReleaseState.CAPTURED, "initialized", {"state": ReleaseState.CAPTURED.value})
            return ReleaseState.CAPTURED

    def state(self) -> ReleaseState:
        if self._publication_path.exists() or self._mutation_path.exists():
            raise StorageError("transaction recovery is required before release state may be read")
        return self._read_state()

    def _read_state(self) -> ReleaseState:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            data = require_mapping(data, field_name="release state")
            require_exact_fields(data, {"schema_version", "state", "updated_at"}, schema=STATE_SCHEMA_VERSION)
            if data["schema_version"] != STATE_SCHEMA_VERSION:
                raise StorageError("unsupported persisted state schema")
            datetime_from_wire(data["updated_at"], field_name="updated_at")
            return ReleaseState(data["state"])
        except (OSError, json.JSONDecodeError, ValueError, LifecycleError) as exc:
            raise StorageError("release state is missing or invalid") from exc

    def transition(self, target: ReleaseState) -> ReleaseState:
        with self.mutation_lock():
            current = self.state()
            if target in {ReleaseState.PUBLISHABLE, ReleaseState.REJECTED, ReleaseState.PUBLISHED}:
                raise StorageError("approval-gated and publication states require their intent-specific storage operation")
            try:
                validate_transition(current, target)
            except LifecycleError as exc:
                raise StorageError(str(exc)) from exc
            if target is ReleaseState.AWAITING_APPROVAL:
                sealed = self.sealed_package()
                digest, _ = _tree_digest(self.stage_dir)
                if digest != sealed.digest:
                    raise StorageError("staging package changed after sealing")
            self._commit_mutation(target, "transition", {"from_state": current.value, "to_state": target.value})
            return target

    def seal_staging(self) -> SealedPackage:
        with self.mutation_lock():
            if self.state() is not ReleaseState.VALIDATED:
                raise StorageError("staging may only be sealed from the validated state")
            digest, files = _tree_digest(self.stage_dir)
            sealed_at = _utc_now()
            payload = {"schema_version": SEAL_SCHEMA_VERSION, "digest": digest, "files": [{"path": path, "digest": item_digest} for path, item_digest in files], "sealed_at": datetime_to_wire(sealed_at)}
            _atomic_write(self._seal_path, canonical_json_bytes(payload))
            self._event("sealed", digest=digest, file_count=len(files))
            return SealedPackage(digest=digest, files=files, sealed_at=sealed_at)

    def sealed_package(self) -> SealedPackage:
        try:
            data = require_mapping(json.loads(self._seal_path.read_text(encoding="utf-8")), field_name="seal")
            require_exact_fields(data, {"schema_version", "digest", "files", "sealed_at"}, schema=SEAL_SCHEMA_VERSION)
            if data["schema_version"] != SEAL_SCHEMA_VERSION or not isinstance(data["digest"], str) or not isinstance(data["files"], list):
                raise StorageError("sealed package is invalid")
            files = tuple((item["path"], item["digest"]) for item in data["files"] if isinstance(item, dict) and set(item) == {"path", "digest"})
            if not files or len(files) != len(data["files"]):
                raise StorageError("sealed package files are invalid")
            return SealedPackage(data["digest"], files, datetime_from_wire(data["sealed_at"], field_name="sealed_at"))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, LifecycleError) as exc:
            raise StorageError("sealed package is missing or invalid") from exc

    def record_approval(self, approval: ApprovalRecord, *, now: datetime | None = None) -> ReleaseState:
        with self.mutation_lock():
            if not isinstance(approval, ApprovalRecord):
                raise StorageError("approval must be an ApprovalRecord")
            if approval.target_season != self.identity.target_season or approval.attempt != self.identity.attempt:
                raise StorageError("approval does not bind this release attempt")
            if Path(approval.release_root).expanduser().resolve() != self.root:
                raise StorageError("approval does not bind this release root")
            current = self.state()
            if current is not ReleaseState.AWAITING_APPROVAL:
                raise StorageError("approval can only be recorded while awaiting approval")
            sealed = self.sealed_package()
            if approval.staged_release_digest != sealed.digest:
                raise StorageError("approval digest does not match sealed staging package")
            if approval.decision is ApprovalDecision.REJECT:
                self._commit_mutation(ReleaseState.REJECTED, "rejected", {"digest": sealed.digest, "approver": approval.approver}, approval=approval)
                return ReleaseState.REJECTED
            current_time = _utc_now() if now is None else now
            if not approval.is_valid_at(current_time, staged_release_digest=sealed.digest):
                raise StorageError("approval is expired or invalid")
            self._commit_mutation(ReleaseState.PUBLISHABLE, "approved", {"digest": sealed.digest, "approver": approval.approver}, approval=approval)
            return ReleaseState.PUBLISHABLE

    def publish(self, *, now: datetime | None = None) -> Path:
        with self.mutation_lock():
            self._recover_pending_mutation()
            recovered = self._recover_prepared_publication()
            if recovered is not None:
                return recovered
            if self.state() is not ReleaseState.PUBLISHABLE:
                raise StorageError("only a publishable release may be published")
            sealed = self.sealed_package()
            digest, _ = _tree_digest(self.stage_dir)
            if digest != sealed.digest:
                raise StorageError("staging package changed after sealing")
            approval = self._load_approval()
            current_time = _utc_now() if now is None else now
            if (approval.target_season != self.identity.target_season or approval.attempt != self.identity.attempt
                    or Path(approval.release_root).expanduser().resolve() != self.root
                    or not approval.is_valid_at(current_time, staged_release_digest=sealed.digest)):
                raise StorageError("stored approval is expired or does not bind this sealed release")
            publication_root = self.root / "published"
            publication_root.mkdir(parents=True, exist_ok=True)
            # Keep the consumer-facing path safely below Windows' legacy path
            # length limit even when the configured Documents root is deep.
            # A full-digest verification below makes a truncated-name collision
            # an explicit failure rather than an ambiguous publication.
            destination = publication_root / f"p-{sealed.digest[:16]}"
            temporary: Path | None = None
            try:
                if destination.exists():
                    existing_digest, _ = _tree_digest(destination)
                    if existing_digest != sealed.digest:
                        raise StorageError("existing published destination is corrupt or conflicting")
                else:
                    temporary = publication_root / f".pending-{uuid4().hex}"
                    shutil.copytree(self.stage_dir, temporary)
                    copied_digest, _ = _tree_digest(temporary)
                    if copied_digest != sealed.digest:
                        raise StorageError("copied publication digest mismatch")
                    os.replace(temporary, destination)
                pointer = {"schema_version": POINTER_SCHEMA_VERSION, "target_season": self.identity.target_season, "attempt": self.identity.attempt, "digest": sealed.digest, "published_path": destination.name, "updated_at": datetime_to_wire(_utc_now())}
                _atomic_write(self._publication_path, canonical_json_bytes({"pointer": pointer, "destination": destination.name}))
                _atomic_write(self.root / "current-release.json", canonical_json_bytes(pointer))
            except BaseException:
                if temporary is not None and temporary.exists():
                    shutil.rmtree(temporary)
                raise
            self._commit_mutation(ReleaseState.PUBLISHED, "published", {"digest": sealed.digest, "destination": destination.name})
            self._publication_path.unlink(missing_ok=True)
            return destination

    def recover_publication(self) -> Path | None:
        """Complete only a verified pointer swap interrupted before state commit."""
        with self.mutation_lock():
            self._recover_pending_mutation()
            return self._recover_prepared_publication()

    def recover(self) -> Path | None:
        """Deterministically finish any durable lifecycle or publication commit."""
        with self.mutation_lock():
            self._recover_pending_mutation()
            return self._recover_prepared_publication()

    @contextmanager
    def mutation_lock(self) -> Iterator[None]:
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "created_at": datetime_to_wire(_utc_now())}
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            try:
                age = self._lock_age()
            except StaleLockError:
                self._event("stale_lock_detected", reason="malformed")
                raise
            self._event("stale_lock_detected" if age >= self.stale_lock_age else "lock_conflict", age_seconds=age.total_seconds())
            if age >= self.stale_lock_age:
                raise StaleLockError("stale mutation lock requires explicit operator recovery") from exc
            raise LockConflictError("attempt is already being mutated") from exc
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(canonical_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
            self._event("lock_acquired", pid=os.getpid())
            yield
        finally:
            try:
                self._lock_path.unlink()
                self._event("lock_released", pid=os.getpid())
            except FileNotFoundError:
                pass

    def _lock_age(self) -> timedelta:
        try:
            data = require_mapping(json.loads(self._lock_path.read_text(encoding="utf-8")), field_name="lock")
            created = datetime_from_wire(data["created_at"], field_name="lock created_at")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, LifecycleError) as exc:
            raise StaleLockError("lock is malformed and requires explicit operator recovery") from exc
        return _utc_now() - created

    def _write_state(self, state: ReleaseState) -> None:
        _atomic_write(self._state_path, canonical_json_bytes({"schema_version": STATE_SCHEMA_VERSION, "state": state.value, "updated_at": datetime_to_wire(_utc_now())}))

    def _load_approval(self) -> ApprovalRecord:
        try:
            return ApprovalRecord.from_dict(json.loads(self._approval_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, LifecycleError) as exc:
            raise StorageError("stored approval is missing or invalid") from exc

    def _commit_mutation(self, target: ReleaseState, kind: str, facts: dict[str, Any], *, approval: ApprovalRecord | None = None) -> None:
        transaction_id = uuid4().hex
        record: dict[str, Any] = {"transaction_id": transaction_id, "target": target.value, "kind": kind, "facts": facts}
        if approval is not None:
            record["approval"] = approval.to_dict()
        _atomic_write(self._mutation_path, canonical_json_bytes(record))
        if approval is not None:
            _atomic_write(self._approval_path, canonical_json_bytes(approval.to_dict()))
        self._write_state(target)
        self._event(kind, transaction_id=transaction_id, **facts)
        self._mutation_path.unlink(missing_ok=True)

    def _recover_pending_mutation(self) -> None:
        if not self._mutation_path.exists():
            return
        try:
            record = require_mapping(json.loads(self._mutation_path.read_text(encoding="utf-8")), field_name="mutation transaction")
            expected = {"transaction_id", "target", "kind", "facts"}
            if "approval" in record:
                expected.add("approval")
            require_exact_fields(record, expected, schema="release-mutation-pending.v1")
            target = ReleaseState(record["target"])
            if not isinstance(record["transaction_id"], str) or not isinstance(record["kind"], str) or not isinstance(record["facts"], dict):
                raise StorageError("mutation transaction fields are invalid")
            if "approval" in record:
                approval = ApprovalRecord.from_dict(record["approval"])
                _atomic_write(self._approval_path, canonical_json_bytes(approval.to_dict()))
            self._write_state(target)
            transaction_id = record["transaction_id"]
            journal = self._journal_path.read_text(encoding="utf-8") if self._journal_path.exists() else ""
            if not any(json.loads(line).get("transaction_id") == transaction_id for line in journal.splitlines() if line):
                self._event(record["kind"], transaction_id=transaction_id, **record["facts"])
            self._mutation_path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, LifecycleError) as exc:
            raise StorageError("mutation transaction is invalid; explicit operator recovery required") from exc

    def _recover_prepared_publication(self) -> Path | None:
        if not self._publication_path.exists():
            return None
        try:
            record = require_mapping(json.loads(self._publication_path.read_text(encoding="utf-8")), field_name="publication transaction")
            require_exact_fields(record, {"pointer", "destination"}, schema="release-publication-prepared.v1")
            pointer = require_mapping(record["pointer"], field_name="publication pointer")
            current_path = self.root / "current-release.json"
            if not current_path.exists():
                # Pointer replacement never happened; the sealed destination is
                # safe to reuse on an explicit retry and no consumer saw it.
                self._publication_path.unlink(missing_ok=True)
                return None
            if current_path.read_bytes() != canonical_json_bytes(pointer):
                raise StorageError("prepared publication pointer conflicts with current release")
            destination = self.root / "published" / record["destination"]
            digest, _ = _tree_digest(destination)
            if digest != pointer["digest"]:
                raise StorageError("prepared publication destination digest mismatch")
            if self._read_state() is ReleaseState.PUBLISHABLE:
                self._commit_mutation(ReleaseState.PUBLISHED, "published_recovered", {"digest": digest, "destination": destination.name})
            elif self._read_state() is not ReleaseState.PUBLISHED:
                raise StorageError("prepared publication conflicts with lifecycle state")
            self._publication_path.unlink(missing_ok=True)
            return destination
        except (OSError, json.JSONDecodeError, KeyError, TypeError, LifecycleError) as exc:
            raise StorageError("prepared publication is invalid; explicit operator recovery required") from exc

    def _event(self, kind: str, **facts: Any) -> None:
        _append_json_line(self._journal_path, {"kind": kind, "at": datetime_to_wire(_utc_now()), **facts})
