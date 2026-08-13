"""Immutable capture and offline replay tests for release runtime inputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from ffmodel.release import ReleaseIdentity
from ffmodel.release.sources import ImmutableSourceCapture, SourceCaptureError, SourceManifest, SourceSpec, json_schema_fingerprint


NOW = datetime(2026, 8, 30, 12, tzinfo=timezone.utc)


def _source(source_id: str = "sleeper.players", payload: bytes = b'{"p1":{"position":"QB"}}', *, loader=None) -> SourceSpec:
    return SourceSpec(source_id, f"https://example.test/{source_id}", loader or (lambda: payload), json_schema_fingerprint(payload or b"{}"))


def _capture(tmp_path, attempt: str = "attempt-01") -> ImmutableSourceCapture:
    return ImmutableSourceCapture(tmp_path / "Documents" / "releases", ReleaseIdentity(2026, attempt))


def _run(capture: ImmutableSourceCapture, specs: list[SourceSpec]):
    return capture.capture(specs, code_identity="c558559ba86c5236c4126dabb78aac373f94d9c4", runtime_identity={"python": "3.13", "ffmodel": "0.1.0"}, configuration={"target_positions": ["QB", "RB", "WR", "TE"]}, now=lambda: NOW)


def test_capture_writes_attempt_scoped_immutable_manifest_and_deterministic_names(tmp_path):
    capture = _capture(tmp_path)
    manifest = _run(capture, [_source("nflverse.players", b'{"data":[{"id":"p1"}]}'), _source("sleeper.players")])

    assert manifest.cutoff == datetime(2026, 8, 31, tzinfo=timezone.utc)
    assert [item.filename for item in manifest.sources] == ["nflverse_players.json", "sleeper_players.json"]
    assert capture.manifest_path.is_file()
    assert (capture.attempt_directory / "inputs" / "sleeper_players.json").read_bytes() == b'{"p1":{"position":"QB"}}'
    assert SourceManifest.from_dict(json.loads(capture.manifest_path.read_text(encoding="utf-8"))) == manifest
    with pytest.raises(SourceCaptureError, match="cannot be recaptured"):
        _run(capture, [_source()])


def test_replay_uses_only_captured_bytes_and_never_invokes_loader(tmp_path):
    capture = _capture(tmp_path)
    source = _source()
    _run(capture, [source])
    offline = _source(loader=lambda: (_ for _ in ()).throw(AssertionError("network must not run during replay")))

    manifest, replayed = capture.replay([offline])

    assert manifest.identity == capture.identity
    assert replayed == {"sleeper.players": b'{"p1":{"position":"QB"}}'}


def test_replay_fails_closed_for_missing_or_digest_mutated_input(tmp_path):
    capture = _capture(tmp_path)
    _run(capture, [_source()])
    payload_path = capture.attempt_directory / "inputs" / "sleeper_players.json"
    payload_path.write_bytes(b'{"p1":{"position":"RB"}}')

    with pytest.raises(SourceCaptureError, match="digest"):
        capture.replay([_source()])

    payload_path.unlink()
    with pytest.raises(SourceCaptureError, match="missing"):
        capture.replay([_source()])


def test_interrupted_or_invalid_download_never_creates_a_replayable_manifest(tmp_path):
    capture = _capture(tmp_path)
    with pytest.raises(SourceCaptureError, match="incomplete"):
        _run(capture, [_source(payload=b"")])
    assert not capture.manifest_path.exists()
    with pytest.raises(SourceCaptureError, match="manifest is missing"):
        capture.replay([_source()])


def test_schema_drift_fails_before_any_downstream_frame_construction(tmp_path):
    capture = _capture(tmp_path)
    payload = b'{"players":[{"id":"p1","position":"QB"}]}'
    _run(capture, [_source(payload=payload)])
    original_fingerprint = json_schema_fingerprint(payload)
    assert original_fingerprint != json_schema_fingerprint(b'{"players":[{"id":"p1","role":"QB"}]}')
    drifted_spec = SourceSpec("sleeper.players", "https://example.test/sleeper.players", lambda: payload, "b" * 64)
    with pytest.raises(SourceCaptureError, match="schema fingerprint"):
        capture.replay([drifted_spec])


def test_manifest_is_deterministic_when_clock_and_runtime_identity_are_fixed(tmp_path):
    first = _capture(tmp_path / "one", "attempt-01")
    second = _capture(tmp_path / "two", "attempt-01")
    first_manifest = _run(first, [_source("nflverse.players"), _source("sleeper.players")])
    second_manifest = _run(second, [_source("sleeper.players"), _source("nflverse.players")])
    assert first_manifest.to_dict() == second_manifest.to_dict()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()


@pytest.mark.parametrize("source_id", ["Sleeper.players", "sleeper players", "sleeper/players", "x" * 129])
def test_source_ids_and_filenames_are_windows_safe(source_id):
    with pytest.raises(SourceCaptureError):
        _source(source_id)


@pytest.mark.parametrize("source_id", ["con", "prn", "aux", "nul", "com1", "lpt9"])
def test_windows_reserved_source_filenames_are_rejected(source_id):
    with pytest.raises(SourceCaptureError, match="reserved filename"):
        _source(source_id)


@pytest.mark.parametrize("attempt", ["../escape", r"..\escape", r"C:\escape", "/escape", "con"])
def test_attempt_path_traversal_and_windows_device_names_are_rejected(tmp_path, attempt):
    with pytest.raises(SourceCaptureError):
        _capture(tmp_path, attempt)
    assert not (tmp_path / "escape").exists()


def test_capture_rejects_secret_bearing_urls_and_metadata(tmp_path):
    with pytest.raises(SourceCaptureError, match="credentials"):
        SourceSpec("sleeper.players", "https://example.test/data?api_key=secret", lambda: b"{}", json_schema_fingerprint(b"{}"))
    capture = _capture(tmp_path)
    with pytest.raises(SourceCaptureError, match="secret-bearing key"):
        capture.capture([_source()], code_identity="revision", runtime_identity={"token": "bad"}, configuration={}, now=lambda: NOW)
    with pytest.raises(SourceCaptureError, match="secret-bearing text"):
        _capture(tmp_path / "text").capture([_source()], code_identity="revision", runtime_identity={"revision_note": "sk-123456789012"}, configuration={}, now=lambda: NOW)


def test_capture_rejects_windows_filename_collisions_before_any_download(tmp_path):
    capture = _capture(tmp_path)
    with pytest.raises(SourceCaptureError, match="filenames collide"):
        _run(capture, [_source("sleeper.players"), _source("sleeper-players")])
    assert not capture.attempt_directory.exists()


def test_capture_rejects_valid_json_when_its_shape_differs_from_configured_schema(tmp_path):
    capture = _capture(tmp_path)
    expected = json_schema_fingerprint(b'{"players":[{"id":"p1","position":"QB"}]}')
    drifted_payload = b'{"players":[{"id":"p1","role":"QB"}]}'
    source = SourceSpec("sleeper.players", "https://example.test/sleeper.players", lambda: drifted_payload, expected)
    with pytest.raises(SourceCaptureError, match="configured expected schema"):
        _run(capture, [source])
    assert not capture.manifest_path.exists()
