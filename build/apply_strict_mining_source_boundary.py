"""Apply the strict M007/S05 Mining Redux source-boundary cleanup.

The applier is intentionally conservative.  It only archives/removes paths that
are explicitly classified by ``strict_mining_source_boundary_manifest.json`` as
non-Mining boundaries, records every attempted/skipped/rejected operation, and
fails closed when a manifest entry tries to touch protected shared or Mining
runtime paths.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build import strict_mining_source_boundary_manifest as strict_manifest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "build" / "strict_mining_source_boundary_manifest.json"
AUDIT_PATH = ROOT / "build" / "strict_mining_source_boundary_audit.json"
ARCHIVE_ROOT = ROOT / "archived_source" / "M007_S05_strict_mining_boundary"
INSTALLER_PROFILES_PATH = ROOT / "build" / "installer_profiles.json"
SKILL_REGISTRY_PATH = ROOT / "core" / "skill_registry.py"

MINING_MODULES = {"mining_loadout", "mining_signals"}
MINING_PROFILE_NAMES = {"mining"}
MINING_BUILTIN_IDS = {"mining", "mining_signals"}
PROTECTED_RELATIVE_ROOTS = {
    "build",
    "build/installer_profiles.json",
    "core",
    "core/skill_registry.py",
    "redux_mining_launcher.py",
    "shared",
    "shared/qt",
    "skills/Mining_Loadout",
    "tools/Mining_Signals",
    "ui",
}


class ApplyBoundaryError(RuntimeError):
    """Raised when the strict source-boundary cleanup cannot proceed safely."""


@dataclass(frozen=True)
class PathSnapshot:
    path: str
    exists: bool
    kind: str
    sha256: str | None = None
    file_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"path": self.path, "exists": self.exists, "kind": self.kind}
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        if self.file_count is not None:
            payload["file_count"] = self.file_count
        return payload


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_rel_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    if "#" in normalized:
        normalized = normalized.split("#", 1)[0]
    candidate = (ROOT / normalized).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ApplyBoundaryError(f"rejected path escapes repository root: {value!r}") from exc
    if not normalized or normalized in {".", "./"} or ".." in Path(normalized).parts:
        raise ApplyBoundaryError(f"rejected unsafe relative path: {value!r}")
    return normalized


def path_overlaps(left: str, right: str) -> bool:
    left_parts = Path(left).parts
    right_parts = Path(right).parts
    return left_parts[: len(right_parts)] == right_parts or right_parts[: len(left_parts)] == left_parts


def is_protected_path(relative_path: str) -> bool:
    return any(path_overlaps(relative_path, protected) for protected in PROTECTED_RELATIVE_ROOTS)


def snapshot_path(relative_path: str) -> PathSnapshot:
    absolute = ROOT / relative_path
    if not absolute.exists():
        return PathSnapshot(path=relative_path, exists=False, kind="missing")
    if absolute.is_file():
        digest = hashlib.sha256(absolute.read_bytes()).hexdigest()
        return PathSnapshot(path=relative_path, exists=True, kind="file", sha256=digest, file_count=1)
    if absolute.is_dir():
        hasher = hashlib.sha256()
        count = 0
        for child in sorted(path for path in absolute.rglob("*") if path.is_file()):
            child_rel = child.relative_to(absolute).as_posix()
            hasher.update(child_rel.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(hashlib.sha256(child.read_bytes()).hexdigest().encode("ascii"))
            hasher.update(b"\0")
            count += 1
        return PathSnapshot(path=relative_path, exists=True, kind="directory", sha256=hasher.hexdigest(), file_count=count)
    return PathSnapshot(path=relative_path, exists=True, kind="other")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApplyBoundaryError(f"malformed JSON: {rel(path)}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ApplyBoundaryError(f"JSON root must be object: {rel(path)}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def manifest_archive_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    strict_manifest.verify_manifest(manifest)
    entries = strict_manifest.manifest_entries(manifest)
    archive_entries = [
        dict(entry)
        for entry in entries
        if entry.get("proposed_action") in {"archive", "delete", "archived", "removed"}
        and (
            str(entry.get("classification", "")).startswith("non_mining")
            or entry.get("classification") in {"archived_non_mining_app_source", "removed_non_mining_app_source"}
        )
    ]
    if not archive_entries:
        raise ApplyBoundaryError("manifest contains no non-Mining cleanup entries")
    for entry in archive_entries:
        path = str(entry["path"])
        base_path = normalize_rel_path(path)
        is_app_root_boundary = entry.get("boundary_type") in {"app_source_root", "archived_app_source_root", "removed_app_source_root"}
        if is_app_root_boundary and is_protected_path(base_path):
            raise ApplyBoundaryError(f"protected-boundary mismatch for archive candidate: {path}")
        if is_app_root_boundary and not path.startswith(("skills/", "tools/")):
            raise ApplyBoundaryError(f"filesystem boundary app source root outside skills/tools: {path}")
        if not is_app_root_boundary and not path.startswith(("build/installer_profiles.json#", "core/skill_registry.py#")):
            raise ApplyBoundaryError(f"metadata boundary outside approved files: {path}")
    return archive_entries


def manifest_pruned_metadata_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    strict_manifest.verify_manifest(manifest)
    entries = strict_manifest.manifest_entries(manifest)
    metadata_entries = [
        dict(entry)
        for entry in entries
        if entry.get("proposed_action") == "pruned"
        and entry.get("boundary_type") in {"installer_profile_metadata", "launcher_metadata"}
    ]
    for entry in metadata_entries:
        path = str(entry["path"])
        if not path.startswith(("build/installer_profiles.json#", "core/skill_registry.py#")):
            raise ApplyBoundaryError(f"metadata boundary outside approved files: {path}")
    return metadata_entries


def metadata_pruning_approvals(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    approvals: list[dict[str, Any]] = []
    for entry in manifest_pruned_metadata_entries(manifest):
        approvals.append(
            {
                "path": str(entry["path"]),
                "action": "pruned",
                "boundary_type": entry.get("boundary_type"),
                "classification": entry.get("classification"),
                "ownership": {
                    "owning_profile": entry.get("owning_profile"),
                    "owning_modules": entry.get("owning_modules", []),
                    "launcher_entry_ids": entry.get("launcher_entry_ids", []),
                },
                "dependency_proof": entry.get("installer_dependency_status"),
                "rationale": entry.get("rationale"),
                "tests_docs_impact": entry.get("tests_docs_impact"),
            }
        )
    return approvals


def archive_app_source_root(entry: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    relative_path = normalize_rel_path(str(entry["path"]))
    source = ROOT / relative_path
    destination = ARCHIVE_ROOT / relative_path
    before = snapshot_path(relative_path)
    operation: dict[str, Any] = {
        "path": relative_path,
        "action": "archive",
        "reason": entry.get("rationale", "manifest-approved non-Mining app source root"),
        "before": before.as_dict(),
    }
    if is_protected_path(relative_path):
        operation.update({"status": "rejected", "rejected_reason": "protected-boundary mismatch"})
        raise ApplyBoundaryError(f"protected-boundary mismatch for {relative_path}")
    if not source.exists():
        operation.update({"status": "skipped", "skipped_reason": "already archived or absent"})
        return operation
    if not source.is_dir():
        operation.update({"status": "rejected", "rejected_reason": "expected directory"})
        raise ApplyBoundaryError(f"type mismatch for archive candidate {relative_path}: expected directory")
    if destination.exists():
        destination_snapshot = snapshot_path(rel(destination))
        if destination_snapshot.sha256 != before.sha256 or destination_snapshot.kind != before.kind:
            operation.update({"status": "rejected", "rejected_reason": "archive destination hash/type mismatch"})
            raise ApplyBoundaryError(f"hash/type mismatch for existing archive destination: {rel(destination)}")
        if not dry_run:
            shutil.rmtree(source)
        operation.update(
            {
                "status": "archived" if not dry_run else "would_archive",
                "destination": rel(destination),
                "skipped_reason": "matching archive already exists" if dry_run else None,
            }
        )
        return operation
    if dry_run:
        operation.update({"status": "would_archive", "destination": rel(destination)})
        return operation
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    after_archive = snapshot_path(rel(destination))
    if after_archive.sha256 != before.sha256 or after_archive.kind != before.kind:
        raise ApplyBoundaryError(f"hash/type mismatch after archiving {relative_path}")
    operation.update({"status": "archived", "destination": rel(destination), "after_archive": after_archive.as_dict()})
    return operation


def prune_installer_profiles(*, dry_run: bool) -> dict[str, Any]:
    data = load_json(INSTALLER_PROFILES_PATH)
    profiles = data.get("profiles")
    modules = data.get("modules")
    if not isinstance(profiles, dict) or not isinstance(modules, dict):
        raise ApplyBoundaryError("installer_profiles.json missing profiles/modules objects")
    removed_profiles = [name for name in sorted(profiles) if name not in MINING_PROFILE_NAMES]
    removed_modules = [name for name in sorted(modules) if name not in MINING_MODULES]
    if "mining" not in profiles:
        raise ApplyBoundaryError("stale profile missing required mining profile")
    mining_modules = profiles["mining"].get("modules") if isinstance(profiles["mining"], Mapping) else None
    if mining_modules != ["mining_signals", "mining_loadout"]:
        raise ApplyBoundaryError("stale profile mining module list changed unexpectedly")
    pruned = {
        "profiles": {"mining": profiles["mining"]},
        "modules": {name: modules[name] for name in modules if name in MINING_MODULES},
        "shared_dependencies": data.get("shared_dependencies", {}),
    }
    operation = {
        "path": "build/installer_profiles.json",
        "action": "prune_metadata",
        "status": "would_prune" if dry_run else "pruned",
        "removed_profiles": removed_profiles,
        "removed_modules": removed_modules,
        "before": snapshot_path("build/installer_profiles.json").as_dict(),
    }
    if not dry_run and (removed_profiles or removed_modules):
        write_json(INSTALLER_PROFILES_PATH, pruned)
        operation["after"] = snapshot_path("build/installer_profiles.json").as_dict()
    elif not removed_profiles and not removed_modules:
        operation["status"] = "skipped"
        operation["skipped_reason"] = "installer metadata already Mining-only"
    return operation


def _dict_assignment_text(name: str, values: Mapping[str, str]) -> str:
    lines = [f"{name}: dict[str, str] = {{"]
    for key, value in values.items():
        lines.append(f'    {key!r}: {value!r},')
    lines.append("}")
    return "\n".join(lines)


def _list_assignment_text(name: str, values: list[dict[str, Any]]) -> str:
    lines = [f"{name}: list[dict] = ["]
    for item in values:
        lines.append("    {")
        for key, value in item.items():
            lines.append(f"        {key!r}: {value!r},")
        lines.append("    },")
    lines.append("]")
    return "\n".join(lines)


def _literal_for_registry(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_for_registry(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_for_registry(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _literal_for_registry(key): _literal_for_registry(value)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "N_" and len(node.args) == 1:
        return _literal_for_registry(node.args[0])
    raise ApplyBoundaryError(f"stale launcher metadata contains unsupported literal: {ast.dump(node, include_attributes=False)}")


def _replace_assignment(text: str, assignment_name: str, replacement: str) -> str:
    tree = ast.parse(text, filename=str(SKILL_REGISTRY_PATH))
    target_node: ast.AST | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == assignment_name for target in node.targets):
                target_node = node
                break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == assignment_name:
            target_node = node
            break
    if target_node is None or not hasattr(target_node, "lineno") or not hasattr(target_node, "end_lineno"):
        raise ApplyBoundaryError(f"stale launcher metadata missing assignment {assignment_name}")
    lines = text.splitlines()
    start = int(target_node.lineno) - 1
    end = int(target_node.end_lineno)
    return "\n".join([*lines[:start], replacement, *lines[end:]]) + ("\n" if text.endswith("\n") else "")


def prune_skill_registry(*, dry_run: bool) -> dict[str, Any]:
    text = SKILL_REGISTRY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(SKILL_REGISTRY_PATH))
    current_defaults: dict[str, str] = {}
    current_builtins: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names: list[str] = []
            value_node: ast.AST | None = None
            if isinstance(node, ast.Assign):
                names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                value_node = node.value
            elif isinstance(node.target, ast.Name):
                names = [node.target.id]
                value_node = node.value
            if value_node is not None and "_DEFAULT_LAUNCHER_ENTRY_IDS" in names:
                value = _literal_for_registry(value_node)
                if isinstance(value, dict):
                    current_defaults = {str(key): str(item) for key, item in value.items()}
            if value_node is not None and "_BUILTIN_SKILLS" in names:
                value = _literal_for_registry(value_node)
                if isinstance(value, list):
                    current_builtins = [dict(item) for item in value if isinstance(item, dict)]
    if not current_defaults or not current_builtins:
        raise ApplyBoundaryError("stale launcher metadata missing built-in defaults")
    pruned_defaults = {key: value for key, value in current_defaults.items() if key in MINING_BUILTIN_IDS}
    pruned_builtins = [item for item in current_builtins if item.get("id") in MINING_BUILTIN_IDS]
    removed_default_ids = sorted(set(current_defaults) - set(pruned_defaults))
    removed_builtin_ids = sorted(str(item.get("id")) for item in current_builtins if item.get("id") not in MINING_BUILTIN_IDS)
    operation = {
        "path": "core/skill_registry.py",
        "action": "prune_launcher_metadata",
        "status": "would_prune" if dry_run else "pruned",
        "removed_default_ids": removed_default_ids,
        "removed_builtin_ids": removed_builtin_ids,
        "before": snapshot_path("core/skill_registry.py").as_dict(),
    }
    if not removed_default_ids and not removed_builtin_ids:
        operation["status"] = "skipped"
        operation["skipped_reason"] = "launcher metadata already Mining-only"
        return operation
    if dry_run:
        return operation
    new_text = _replace_assignment(
        text,
        "_DEFAULT_LAUNCHER_ENTRY_IDS",
        _dict_assignment_text("_DEFAULT_LAUNCHER_ENTRY_IDS", pruned_defaults),
    )
    new_text = _replace_assignment(
        new_text,
        "_BUILTIN_SKILLS",
        _list_assignment_text("_BUILTIN_SKILLS", pruned_builtins),
    )
    SKILL_REGISTRY_PATH.write_text(new_text, encoding="utf-8")
    operation["after"] = snapshot_path("core/skill_registry.py").as_dict()
    return operation


def protected_runtime_audit() -> list[dict[str, Any]]:
    results = []
    for relative_path in sorted(PROTECTED_RELATIVE_ROOTS):
        results.append(snapshot_path(relative_path).as_dict())
    return results


def apply_cleanup(*, dry_run: bool, manifest_path: Path = MANIFEST_PATH, audit_path: Path = AUDIT_PATH) -> dict[str, Any]:
    manifest = strict_manifest.load_manifest(manifest_path)
    archive_entries = manifest_archive_entries(manifest)
    metadata_approvals = metadata_pruning_approvals(manifest)
    operations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    try:
        for entry in archive_entries:
            if entry.get("boundary_type") in {"app_source_root", "archived_app_source_root"}:
                operations.append(archive_app_source_root(entry, dry_run=dry_run))
        operations.append(prune_installer_profiles(dry_run=dry_run))
        operations.append(prune_skill_registry(dry_run=dry_run))
    except ApplyBoundaryError as exc:
        rejected.append({"status": "rejected", "reason": str(exc)})
        raise
    finally:
        audit: dict[str, Any] = {
            "audit_version": 1,
            "generated_by": "build/apply_strict_mining_source_boundary.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "manifest_path": rel(manifest_path),
            "archive_root": rel(ARCHIVE_ROOT),
            "attempted": len(operations),
            "archived": sum(1 for op in operations if op.get("status") == "archived"),
            "removed": sum(1 for op in operations if op.get("status") in {"pruned", "removed"}),
            "skipped": sum(1 for op in operations if op.get("status") == "skipped"),
            "rejected": rejected,
            "operations": operations,
            "protected_runtime_snapshots": protected_runtime_audit(),
            "metadata_pruning_approvals": metadata_approvals,
            "approved_manifest_paths": [
                *[str(entry.get("path")) for entry in archive_entries],
                *[str(approval.get("path")) for approval in metadata_approvals],
            ],
        }
        write_json(audit_path, audit)
    return audit


def verify_audit(audit_path: Path = AUDIT_PATH, manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    audit = load_json(audit_path)
    manifest = strict_manifest.load_manifest(manifest_path)
    archive_entries = manifest_archive_entries(manifest)
    metadata_entries = manifest_pruned_metadata_entries(manifest)
    audit_approved_paths = audit.get("approved_manifest_paths")
    if not isinstance(audit_approved_paths, list) or not all(isinstance(item, str) for item in audit_approved_paths):
        raise ApplyBoundaryError("audit missing approved manifest path list")
    approved_paths = {str(entry.get("path")) for entry in archive_entries} | {str(entry.get("path")) for entry in metadata_entries} | set(audit_approved_paths)
    missing_metadata = sorted(str(entry.get("path")) for entry in metadata_entries if str(entry.get("path")) not in set(audit_approved_paths))
    if missing_metadata:
        raise ApplyBoundaryError(f"audit missing metadata pruning approvals: {missing_metadata}")
    metadata_approvals = audit.get("metadata_pruning_approvals")
    if not isinstance(metadata_approvals, list) or len(metadata_approvals) != len(metadata_entries):
        raise ApplyBoundaryError("audit missing metadata pruning approval records")
    approvals_by_path = {str(item.get("path")): item for item in metadata_approvals if isinstance(item, Mapping)}
    for entry in metadata_entries:
        path = str(entry.get("path"))
        approval = approvals_by_path.get(path)
        if approval is None:
            raise ApplyBoundaryError(f"audit missing metadata pruning approval: {path}")
        if approval.get("action") != "pruned":
            raise ApplyBoundaryError(f"audit metadata pruning approval has wrong action: {path}")
        if approval.get("dependency_proof") != entry.get("installer_dependency_status"):
            raise ApplyBoundaryError(f"audit metadata pruning approval dependency proof mismatch: {path}")
        ownership = approval.get("ownership")
        if not isinstance(ownership, Mapping):
            raise ApplyBoundaryError(f"audit metadata pruning approval lacks ownership: {path}")
    operations = audit.get("operations")
    if not isinstance(operations, list):
        raise ApplyBoundaryError("audit missing operations list")
    if audit.get("rejected") not in ([], None):
        raise ApplyBoundaryError("audit records rejected operations")
    attempted_paths = set()
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ApplyBoundaryError("audit operation must be object")
        path = str(operation.get("path", ""))
        status = operation.get("status")
        action = operation.get("action")
        if status not in {"archived", "would_archive", "removed", "would_remove", "pruned", "would_prune", "skipped"}:
            raise ApplyBoundaryError(f"audit operation has invalid status for {path}: {status}")
        if action in {"archive", "remove", "delete"}:
            if path not in approved_paths:
                raise ApplyBoundaryError(f"audit archived unapproved path: {path}")
            if is_protected_path(path):
                raise ApplyBoundaryError(f"audit touched protected shared-runtime path: {path}")
            attempted_paths.add(path)
    expected_app_roots = {
        str(entry.get("path"))
        for entry in archive_entries
        if entry.get("boundary_type") in {"app_source_root", "archived_app_source_root", "removed_app_source_root"}
        and entry.get("proposed_action") in {"archive", "delete"}
    }
    missing = sorted(expected_app_roots - attempted_paths)
    if missing:
        raise ApplyBoundaryError(f"audit missing app source cleanup attempts: {missing}")
    protected_snapshots = audit.get("protected_runtime_snapshots")
    if not isinstance(protected_snapshots, list) or not protected_snapshots:
        raise ApplyBoundaryError("audit missing protected shared-runtime snapshots")
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply strict Mining Redux source-boundary cleanup.")
    parser.add_argument("--dry-run", action="store_true", help="plan cleanup and write audit without mutating source")
    parser.add_argument("--apply", action="store_true", help="archive/prune manifest-approved non-Mining source boundaries")
    parser.add_argument("--verify-audit", action="store_true", help="verify the audit only contains manifest-approved operations")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="strict boundary manifest path")
    parser.add_argument("--audit", default=str(AUDIT_PATH), help="audit JSON path")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    audit_path = Path(args.audit)
    if not audit_path.is_absolute():
        audit_path = ROOT / audit_path

    modes = [args.dry_run, args.apply, args.verify_audit]
    if sum(1 for mode in modes if mode) != 1:
        print("FAIL strict source boundary apply: choose exactly one of --dry-run, --apply, or --verify-audit", file=sys.stderr)
        return 2

    try:
        if args.verify_audit:
            audit = verify_audit(audit_path=audit_path, manifest_path=manifest_path)
            print(
                "PASS strict source boundary audit: "
                f"operations={len(audit['operations'])} archived={audit.get('archived')} removed={audit.get('removed')} skipped={audit.get('skipped')}"
            )
            return 0
        audit = apply_cleanup(dry_run=args.dry_run, manifest_path=manifest_path, audit_path=audit_path)
        mode = "dry-run" if args.dry_run else "apply"
        print(
            f"PASS strict source boundary {mode}: "
            f"operations={len(audit['operations'])} archived={audit.get('archived')} removed={audit.get('removed')} skipped={audit.get('skipped')}"
        )
        return 0
    except (ApplyBoundaryError, strict_manifest.BoundaryError, OSError) as exc:
        print(f"FAIL strict source boundary apply: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
