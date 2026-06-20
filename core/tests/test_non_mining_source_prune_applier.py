from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "build" / "apply_non_mining_source_prune.py"
SPEC = importlib.util.spec_from_file_location("source_prune_applier", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
applier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(applier)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _base_manifest(entries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "workspace_root_name": "workspace",
        "installer_profile_source": "build/installer_profiles.json",
        "guardrails": ["explicit leaf file only"],
        "entries": entries,
    }


def _delete_entry(path: str, payload: bytes, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": path,
        "path_type": "file",
        "disposition": "delete",
        "deletion_allowed": True,
        "rationale": "obsolete source leaf file with explicit proof",
        "protected": False,
        "entrypoint_or_launcher_dependency": False,
        "test_or_fixture_dependency": False,
        "profile_payload_requirement": False,
        "delete_proof": ["obsolete source proven unused by S01/S02 prune research"],
        "sha256": _digest(payload),
        "byte_count": len(payload),
    }
    entry.update(overrides)
    return entry


def test_check_mode_audits_approved_delete_without_mutating(tmp_path: Path) -> None:
    payload = b"obsolete helper\n"
    candidate = tmp_path / "obsolete_module" / "old_unused.py"
    candidate.parent.mkdir()
    candidate.write_bytes(payload)

    audit = applier.evaluate_manifest(
        tmp_path,
        _base_manifest([_delete_entry("obsolete_module/old_unused.py", payload)]),
        "check",
        generated_paths=set(),
    )

    assert candidate.exists()
    assert audit["summary"] == {
        "entry_count": 1,
        "attempted_count": 1,
        "deleted_count": 0,
        "skipped_count": 1,
        "rejected_count": 0,
    }
    assert audit["skipped"][0]["reason"] == "check mode would delete manifest-approved source leaf file"


def test_apply_mode_deletes_only_hash_matching_leaf_file(tmp_path: Path) -> None:
    payload = b"obsolete helper\n"
    candidate = tmp_path / "obsolete_module" / "old_unused.py"
    candidate.parent.mkdir()
    candidate.write_bytes(payload)

    audit = applier.evaluate_manifest(
        tmp_path,
        _base_manifest([_delete_entry("obsolete_module/old_unused.py", payload)]),
        "apply",
        generated_paths=set(),
    )

    assert not candidate.exists()
    assert audit["summary"]["deleted_count"] == 1
    assert audit["deleted"][0]["reason"] == "deleted manifest-approved source leaf file"


def test_rejects_path_traversal_without_mutating(tmp_path: Path) -> None:
    payload = b"obsolete helper\n"
    audit = applier.evaluate_manifest(
        tmp_path,
        _base_manifest([_delete_entry("../outside.py", payload)]),
        "apply",
        generated_paths=set(),
    )

    assert audit["summary"]["rejected_count"] >= 1
    assert any("unsafe path syntax" in item["reason"] for item in audit["rejected"])


def test_rejects_missing_manifest_metadata(tmp_path: Path) -> None:
    payload = b"obsolete helper\n"
    manifest = _base_manifest([_delete_entry("obsolete_module/old_unused.py", payload)])
    manifest.pop("guardrails")

    audit = applier.evaluate_manifest(tmp_path, manifest, "check", generated_paths=set())

    assert any("manifest missing metadata field: guardrails" == item["reason"] for item in audit["rejected"])


def test_rejects_directory_level_delete(tmp_path: Path) -> None:
    payload = b"obsolete helper\n"
    directory = tmp_path / "obsolete_module"
    directory.mkdir()
    entry = _delete_entry("obsolete_module", payload, path_type="directory")

    audit = applier.evaluate_manifest(tmp_path, _base_manifest([entry]), "apply", generated_paths=set())

    assert directory.exists()
    assert any("not an explicit leaf file" in item["reason"] for item in audit["rejected"])


def test_rejects_protected_runtime_path(tmp_path: Path) -> None:
    payload = b"obsolete helper\n"
    candidate = tmp_path / "core" / "old_unused.py"
    candidate.parent.mkdir()
    candidate.write_bytes(payload)
    entry = _delete_entry("core/old_unused.py", payload)

    audit = applier.evaluate_manifest(tmp_path, _base_manifest([entry]), "apply", generated_paths=set())

    assert candidate.exists()
    assert any("protected path regression guard" in item["reason"] for item in audit["rejected"])


def test_rejects_s02_generated_artifact_cleanup_path(tmp_path: Path) -> None:
    payload = b"obsolete helper\n"
    candidate = tmp_path / "obsolete_module" / "old_unused.py"
    candidate.parent.mkdir()
    candidate.write_bytes(payload)

    audit = applier.evaluate_manifest(
        tmp_path,
        _base_manifest([_delete_entry("obsolete_module/old_unused.py", payload)]),
        "apply",
        generated_paths={"obsolete_module/old_unused.py"},
    )

    assert candidate.exists()
    assert any("S02 generated-artifact cleanup" in item["reason"] for item in audit["rejected"])


def test_rejects_hash_mismatch_without_mutating(tmp_path: Path) -> None:
    original = b"obsolete helper\n"
    candidate = tmp_path / "obsolete_module" / "old_unused.py"
    candidate.parent.mkdir()
    candidate.write_bytes(b"changed helper\n")

    audit = applier.evaluate_manifest(
        tmp_path,
        _base_manifest([_delete_entry("obsolete_module/old_unused.py", original)]),
        "apply",
        generated_paths=set(),
    )

    assert candidate.exists()
    assert any("hash no longer matches" in item["reason"] for item in audit["rejected"])
