"""Apply the S03 non-Mining source prune manifest through hard guards.

The default project manifest is intentionally conservative and currently contains
no delete candidates.  This applier still exists so any future delete candidate
must pass the same deterministic guard sequence before a source file can be
removed.  It writes an audit JSON for both check and apply modes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from non_mining_source_prune_manifest import (  # type: ignore
        PROTECTED_FILES,
        PROTECTED_ROOTS,
        validate_delete_entry,
    )
except Exception:  # pragma: no cover - import fallback for unusual launchers
    PROTECTED_ROOTS = (
        "tools/Mining_Signals",
        "skills/Mining_Loadout",
        "core",
        "shared",
        "ui",
        "build",
    )
    PROTECTED_FILES = {
        "redux_mining_launcher.py",
        "redux_mining_launcher_settings.json",
        "skill_launcher.py",
        "skill_launcher_settings.json",
        "build/generated_artifact_cleanup_manifest.py",
        "build/generated_artifact_cleanup_manifest.json",
        "build/apply_generated_artifact_cleanup.py",
        "build/generated_artifact_cleanup_audit.json",
        "build/verify_generated_artifact_cleanup.py",
        "build/installer_profiles.json",
        "build/installer_profiles.py",
        "build/installer_entrypoint.py",
        "build/redux_mining_build.py",
    }

    def validate_delete_entry(entry: dict[str, Any]) -> None:
        rel = entry.get("path")
        if not isinstance(rel, str) or not rel:
            raise ValueError("delete entry missing path")
        if rel.startswith("/") or "\\" in rel or ".." in Path(rel).parts:
            raise ValueError(f"delete entry has unsafe path syntax: {rel}")
        if entry.get("path_type") != "file":
            raise ValueError(f"delete entry is not an explicit leaf file: {rel}")
        if entry.get("protected"):
            raise ValueError(f"delete entry violates protected root guard: {rel}")
        if entry.get("entrypoint_or_launcher_dependency"):
            raise ValueError(f"delete entry violates entrypoint guard: {rel}")
        if entry.get("test_or_fixture_dependency"):
            raise ValueError(f"delete entry violates test/fixture guard: {rel}")
        if entry.get("profile_payload_requirement"):
            raise ValueError(f"delete entry violates installer profile payload guard: {rel}")
        proof = entry.get("delete_proof")
        if not isinstance(proof, list) or not proof:
            raise ValueError(f"delete entry has no explicit proof: {rel}")


MANIFEST_PATH = Path("build/non_mining_source_prune_manifest.json")
AUDIT_PATH = Path("build/non_mining_source_prune_audit.json")
GENERATED_CLEANUP_MANIFEST_PATH = Path("build/generated_artifact_cleanup_manifest.json")
MANIFEST_VERSION = 1
HASH_FIELDS = ("sha256", "current_sha256", "file_sha256")
SIZE_FIELDS = ("size", "byte_count", "current_size", "file_size")
REQUIRED_MANIFEST_FIELDS = (
    "manifest_version",
    "workspace_root_name",
    "installer_profile_source",
    "guardrails",
    "entries",
)
REQUIRED_ENTRY_FIELDS = (
    "path",
    "path_type",
    "disposition",
    "deletion_allowed",
    "rationale",
    "protected",
    "delete_proof",
)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def as_posix(path: Path) -> str:
    return path.as_posix().rstrip("/")


def is_under(rel_path: str, root: str) -> bool:
    return rel_path == root or rel_path.startswith(root.rstrip("/") + "/")


def is_protected_path(rel_path: str) -> tuple[bool, str | None]:
    if rel_path in PROTECTED_FILES:
        return True, rel_path
    for root in PROTECTED_ROOTS:
        if is_under(rel_path, root):
            return True, root
    return False, None


def resolve_inside(root: Path, rel_path: str) -> Path:
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError("entry path is missing")
    if rel_path.startswith("/") or "\\" in rel_path or ".." in Path(rel_path).parts:
        raise ValueError(f"unsafe path syntax: {rel_path}")
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {rel_path}") from exc
    return candidate


def current_path_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "missing"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_hash(entry: dict[str, Any]) -> str | None:
    for field in HASH_FIELDS:
        value = entry.get(field)
        if isinstance(value, str) and value:
            return value.lower()
    return None


def expected_size(entry: dict[str, Any]) -> int | None:
    for field in SIZE_FIELDS:
        value = entry.get(field)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path.as_posix()}")
    return data


def generated_cleanup_paths(root: Path, manifest_path: Path) -> set[str]:
    path = root / manifest_path
    if not path.is_file():
        return set()
    data = load_json(path)
    paths: set[str] = set()
    for entry in data.get("entries", []):
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            paths.add(entry["path"].replace("\\", "/").rstrip("/"))
    return paths


def reject(record: dict[str, Any], reason: str, rejected: list[dict[str, Any]]) -> None:
    record["reason"] = reason
    rejected.append(record)


def validate_manifest_metadata(manifest: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            reasons.append(f"manifest missing metadata field: {field}")
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        reasons.append(f"manifest_version must be {MANIFEST_VERSION}")
    if not isinstance(manifest.get("guardrails"), list) or not manifest.get("guardrails"):
        reasons.append("manifest guardrails must be a non-empty list")
    if not isinstance(manifest.get("entries"), list) or not manifest.get("entries"):
        reasons.append("manifest entries must be a non-empty list")
    return reasons


def audit_record(entry: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "path": entry.get("path"),
        "disposition": entry.get("disposition"),
        "deletion_allowed": entry.get("deletion_allowed"),
    }


def evaluate_manifest(root: Path, manifest: dict[str, Any], mode: str, generated_paths: set[str]) -> dict[str, Any]:
    attempted: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    metadata_rejections = validate_manifest_metadata(manifest)
    for reason in metadata_rejections:
        rejected.append({"path": None, "reason": reason})

    entries = manifest.get("entries") if isinstance(manifest.get("entries"), list) else []
    seen: set[str] = set()
    paths: list[str] = []

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            rejected.append({"index": index, "path": None, "reason": "manifest entry is not an object"})
            continue

        record = audit_record(entry, index)
        rel = entry.get("path")
        if not isinstance(rel, str) or not rel:
            reject(record, "manifest entry missing path", rejected)
            continue
        paths.append(rel)
        if rel in seen:
            reject(record, "manifest contains duplicate path", rejected)
            continue
        seen.add(rel)

        missing_fields = [field for field in REQUIRED_ENTRY_FIELDS if field not in entry]
        if missing_fields:
            reject(record, f"manifest entry missing fields: {', '.join(missing_fields)}", rejected)
            continue

        try:
            abs_path = resolve_inside(root, rel)
        except ValueError as exc:
            reject(record, str(exc), rejected)
            continue

        delete_requested = entry.get("disposition") == "delete" or entry.get("deletion_allowed") is True
        if not delete_requested:
            skipped.append({**record, "reason": "entry is not marked delete-safe"})
            continue

        attempted.append(record)
        try:
            validate_delete_entry(entry)
        except ValueError as exc:
            reject(record, str(exc), rejected)
            continue

        protected, protected_by = is_protected_path(rel)
        if protected:
            reject(record, f"protected path regression guard matched: {protected_by}", rejected)
            continue

        if any(rel == generated or is_under(rel, generated) for generated in generated_paths):
            reject(record, "path belongs to S02 generated-artifact cleanup manifest", rejected)
            continue

        actual_type = current_path_type(abs_path)
        if actual_type != entry.get("path_type"):
            reject(record, f"current path type mismatch: expected {entry.get('path_type')} actual {actual_type}", rejected)
            continue
        if actual_type != "file":
            reject(record, "delete candidate is not a current leaf file", rejected)
            continue

        expected_digest = expected_hash(entry)
        if not expected_digest:
            reject(record, "delete candidate missing current file hash", rejected)
            continue
        actual_digest = sha256_file(abs_path)
        if actual_digest.lower() != expected_digest:
            reject(record, "current file hash no longer matches manifest", rejected)
            continue

        expected_bytes = expected_size(entry)
        actual_bytes = abs_path.stat().st_size
        if expected_bytes is not None and actual_bytes != expected_bytes:
            reject(record, f"current file size mismatch: expected {expected_bytes} actual {actual_bytes}", rejected)
            continue

        if mode == "apply":
            abs_path.unlink()
            deleted.append({**record, "reason": "deleted manifest-approved source leaf file", "sha256": actual_digest, "byte_count": actual_bytes})
        else:
            skipped.append({**record, "reason": "check mode would delete manifest-approved source leaf file", "sha256": actual_digest, "byte_count": actual_bytes})

    if paths != sorted(paths, key=str.lower):
        rejected.append({"path": None, "reason": "manifest entries are not deterministic path order"})

    return {
        "audit_version": 1,
        "mode": mode,
        "manifest_path": MANIFEST_PATH.as_posix(),
        "summary": {
            "entry_count": len(entries),
            "attempted_count": len(attempted),
            "deleted_count": len(deleted),
            "skipped_count": len(skipped),
            "rejected_count": len(rejected),
        },
        "attempted": attempted,
        "deleted": deleted,
        "skipped": skipped,
        "rejected": rejected,
    }


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def run(root: Path, manifest_path: Path, audit_path: Path, mode: str, generated_manifest_path: Path) -> int:
    manifest_abs = root / manifest_path
    audit_abs = root / audit_path
    if not manifest_abs.is_file():
        print(f"manifest missing: {manifest_path.as_posix()}", file=sys.stderr)
        return 1
    try:
        manifest = load_json(manifest_abs)
        generated_paths = generated_cleanup_paths(root, generated_manifest_path)
        audit = evaluate_manifest(root, manifest, mode, generated_paths)
    except Exception as exc:
        print(f"failed to evaluate source prune manifest: {exc}", file=sys.stderr)
        return 1

    audit["manifest_path"] = manifest_path.as_posix()
    audit["audit_path"] = audit_path.as_posix()
    audit_abs.parent.mkdir(parents=True, exist_ok=True)
    audit_abs.write_text(canonical_json(audit), encoding="utf-8")

    summary = audit["summary"]
    print(
        f"source prune {mode} entries={summary['entry_count']} attempted={summary['attempted_count']} "
        f"deleted={summary['deleted_count']} skipped={summary['skipped_count']} rejected={summary['rejected_count']} "
        f"audit={audit_path.as_posix()}"
    )
    return 1 if summary["rejected_count"] else 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="validate manifest and write a non-mutating audit")
    modes.add_argument("--apply", action="store_true", help="delete only manifest-approved files after all guards pass")
    parser.add_argument("--manifest", default=MANIFEST_PATH.as_posix(), help="manifest path relative to workspace root")
    parser.add_argument("--audit", default=AUDIT_PATH.as_posix(), help="audit output path relative to workspace root")
    parser.add_argument(
        "--generated-cleanup-manifest",
        default=GENERATED_CLEANUP_MANIFEST_PATH.as_posix(),
        help="S02 generated-artifact cleanup manifest path relative to workspace root",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    mode = "apply" if args.apply else "check"
    root = workspace_root()
    return run(
        root,
        Path(args.manifest),
        Path(args.audit),
        mode,
        Path(args.generated_cleanup_manifest),
    )


if __name__ == "__main__":
    raise SystemExit(main())
