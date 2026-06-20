"""Build and verify the strict M007 Mining Redux source-boundary inventory.

The inventory is intentionally applied-state: it records non-Mining app roots
that were removed by the strict cleanup, verifies that live source now contains
only Mining Redux app roots plus retained shared runtime, and fails with the
first stale profile, launcher, import, or filesystem boundary that prevents
source-level Mining Redux closure.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "build" / "strict_mining_source_boundary_manifest.json"
AUDIT_PATH = ROOT / "build" / "strict_mining_source_boundary_audit.json"
INSTALLER_PROFILES_PATH = ROOT / "build" / "installer_profiles.json"
SKILL_REGISTRY_PATH = ROOT / "core" / "skill_registry.py"
REDUX_LAUNCHER_PATH = ROOT / "redux_mining_launcher.py"

MINING_MODULES = {"mining_loadout", "mining_signals"}
MINING_LAUNCHER_ENTRY_IDS = {"mining-loadout", "mining-signals"}
NON_MINING_MODULES = {
    "cargo_loader",
    "craft_database",
    "dps_calculator",
    "market_finder",
    "mission_database",
    "trade_hub",
    "battle_buddy",
}

MINING_REFERENCE_PATHS = (
    "redux_mining_launcher.py",
    "build/redux_mining_build.py",
    "build/installer_profile_dry_run.py",
    "build/installer_release_smoke.py",
    "build/verify_non_mining_source_prune.py",
    "tools/Mining_Signals",
    "skills/Mining_Loadout",
)

REQUIRED_NON_MINING_ROOTS = (
    "skills/Cargo_loader",
    "skills/Craft_Database",
    "skills/DPS_Calculator",
    "skills/Market_Finder",
    "skills/Mission_Database",
    "skills/Trade_Hub",
    "tools/Battle_Buddy",
)

REQUIRED_METADATA_BOUNDARIES = (
    "build/installer_profiles.json#profiles.trading",
    "build/installer_profiles.json#profiles.combat",
    "build/installer_profiles.json#profiles.reference",
    "build/installer_profiles.json#profiles.full.non_mining_modules",
    "core/skill_registry.py#non_mining_launcher_defaults",
)

REQUIRED_SHARED_BOUNDARIES = (
    "shared/qt",
    "shared",
    "core",
    "ui",
    "build/installer_profiles.json",
    "redux_mining_launcher.py",
)

VERIFICATION_COMMANDS = (
    "python -B build/strict_mining_source_boundary_manifest.py --check",
    "python -B build/apply_strict_mining_source_boundary.py --verify-audit",
    "python -B -m unittest core.tests.test_strict_mining_source_boundary_manifest core.tests.test_apply_strict_mining_source_boundary",
)


class BoundaryError(RuntimeError):
    """Raised when the strict source-boundary inventory is stale or unsafe."""


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_rel_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    candidate = (ROOT / normalized.split("#", 1)[0]).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise BoundaryError(f"filesystem boundary escapes repository root: {value!r}") from exc
    return normalized


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BoundaryError(f"filesystem boundary missing: {rel(path)}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BoundaryError(f"filesystem boundary malformed JSON: {rel(path)}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BoundaryError(f"filesystem boundary JSON root must be object: {rel(path)}")
    return payload


def profile_modules(profiles: Mapping[str, Any], profile_name: str) -> list[str]:
    profile = profiles.get(profile_name)
    if not isinstance(profile, Mapping):
        raise BoundaryError(f"stale profile missing installer profile: {profile_name}")
    modules = profile.get("modules")
    if not isinstance(modules, list) or not all(isinstance(item, str) for item in modules):
        raise BoundaryError(f"stale profile has invalid modules list: {profile_name}")
    return list(modules)


def module_profile_index(profiles: Mapping[str, Any]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for profile_name in sorted(profiles):
        for module_name in profile_modules(profiles, profile_name):
            index.setdefault(module_name, []).append(profile_name)
    return {module_name: sorted(names) for module_name, names in index.items()}


def source_files_under(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    ignored_dirs = {".git", ".gsd", "__pycache__", ".pytest_cache", "dist", "build"}
    for child in path.rglob("*"):
        if any(part in ignored_dirs for part in child.parts):
            continue
        if child.is_file() and child.suffix.lower() in {".py", ".json", ".md", ".txt", ".toml", ".yaml", ".yml"}:
            yield child


def _path_tokens(candidate_path: str) -> set[str]:
    path = candidate_path.split("#", 1)[0]
    stem = Path(path).name
    tokens = {stem, stem.lower(), stem.replace("_", "-"), stem.lower().replace("_", "-")}
    tokens.add(path)
    tokens.add(path.replace("/", ".").replace(".py", ""))
    return {token for token in tokens if token}


def _is_import_like_reference(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("import ")
        or stripped.startswith("from ")
        or "__import__(" in stripped
        or "importlib.import_module" in stripped
    )


def find_text_references(candidate_path: str, extra_tokens: Iterable[str]) -> list[dict[str, Any]]:
    tokens = sorted(_path_tokens(candidate_path).union(extra_tokens), key=len, reverse=True)
    references: list[dict[str, Any]] = []
    for ref_path in MINING_REFERENCE_PATHS:
        absolute = ROOT / ref_path
        for source_file in source_files_under(absolute):
            try:
                text = source_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not _is_import_like_reference(line):
                    continue
                match = next((token for token in tokens if token and token in line), None)
                if match:
                    references.append({"path": rel(source_file), "line": line_number, "token": match})
                    break
    return references


def skill_registry_launcher_ids() -> dict[str, str]:
    text = SKILL_REGISTRY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(SKILL_REGISTRY_PATH))
    launcher_ids: dict[str, str] = {}
    for node in tree.body:
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "_DEFAULT_LAUNCHER_ENTRY_IDS" in names:
                value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_DEFAULT_LAUNCHER_ENTRY_IDS":
                value_node = node.value
        if value_node is not None:
            value = ast.literal_eval(value_node)
            if isinstance(value, dict):
                launcher_ids = {str(key): str(item) for key, item in value.items()}
    if not launcher_ids:
        raise BoundaryError("stale launcher metadata missing _DEFAULT_LAUNCHER_ENTRY_IDS")
    return launcher_ids


def launcher_ids_for_module(module_name: str, module_config: Mapping[str, Any], registry_ids: Mapping[str, str]) -> list[str]:
    ids: set[str] = set()
    for value in module_config.get("launcher_entry_ids", []) if isinstance(module_config.get("launcher_entry_ids"), list) else []:
        if isinstance(value, str):
            ids.add(value)
    for shortcut in module_config.get("shortcuts", []) if isinstance(module_config.get("shortcuts"), list) else []:
        if isinstance(shortcut, Mapping) and isinstance(shortcut.get("id"), str):
            ids.add(str(shortcut["id"]))
    ids.update(value for key, value in registry_ids.items() if key == module_name or value in ids)
    return sorted(ids)


def entry_for_module(
    module_name: str,
    module_config: Mapping[str, Any],
    profiles_by_module: Mapping[str, list[str]],
    registry_ids: Mapping[str, str],
) -> list[dict[str, Any]]:
    owned_paths = module_config.get("owned_paths")
    if not isinstance(owned_paths, list) or not all(isinstance(item, str) for item in owned_paths):
        raise BoundaryError(f"stale profile module missing owned_paths: {module_name}")
    launcher_ids = launcher_ids_for_module(module_name, module_config, registry_ids)
    entries: list[dict[str, Any]] = []
    for owned_path in sorted(owned_paths):
        if module_name not in MINING_MODULES:
            raise BoundaryError(f"stale profile live installer module is non-Mining: {module_name}")
        references = find_text_references(owned_path, {module_name, *launcher_ids})
        entries.append(
            {
                "path": normalize_rel_path(owned_path),
                "boundary_type": "app_source_root",
                "classification": "mining",
                "proposed_action": "keep",
                "exists": (ROOT / owned_path).exists(),
                "owning_module": module_name,
                "owning_profiles": profiles_by_module.get(module_name, []),
                "launcher_entry_ids": launcher_ids,
                "mining_redux_import_references": references,
                "installer_dependency_status": "required_by_mining_profile",
                "tests_docs_impact": "Retain and verify as active Mining Redux source.",
                "rationale": "Mining Redux visible app root.",
            }
        )
    return entries


def archived_non_mining_entries() -> list[dict[str, Any]]:
    """Return final-state entries for non-Mining app roots removed from Redux source."""
    entries: list[dict[str, Any]] = []
    audit = load_json(AUDIT_PATH) if AUDIT_PATH.exists() else {"operations": []}
    operations = audit.get("operations")
    if not isinstance(operations, list):
        raise BoundaryError("filesystem boundary audit missing operations list")
    by_path = {
        str(op.get("path")): op
        for op in operations
        if isinstance(op, Mapping) and op.get("action") in {"archive", "delete", "remove"}
    }
    for path in REQUIRED_NON_MINING_ROOTS:
        operation = by_path.get(path, {})
        live_exists = (ROOT / path).exists()
        removal_status = "removed" if not live_exists else operation.get("status") if isinstance(operation, Mapping) else "missing_audit_operation"
        module_name = path.split("/", 1)[0]
        entries.append(
            {
                "path": normalize_rel_path(path),
                "boundary_type": "removed_app_source_root",
                "classification": "removed_non_mining_app_source",
                "proposed_action": "removed",
                "exists": live_exists,
                "removal_status": removal_status,
                "owning_module": module_name,
                "owning_profiles": [],
                "launcher_entry_ids": [],
                "mining_redux_import_references": find_text_references(path, []),
                "tests_docs_impact": "Non-Mining app source is intentionally absent from the Redux mining source tree.",
                "rationale": "Strict Redux mining boundary keeps only Mining Redux app roots plus shared runtime.",
            }
        )
    return entries


def metadata_entries(profiles: Mapping[str, Any], registry_ids: Mapping[str, str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for profile_name in ("trading", "combat", "reference"):
        exists = profile_name in profiles
        modules = profile_modules(profiles, profile_name) if exists else []
        entries.append(
            {
                "path": f"build/installer_profiles.json#profiles.{profile_name}",
                "boundary_type": "installer_profile_metadata",
                "classification": "non_mining_profile_metadata" if exists else "archived_non_mining_profile_metadata",
                "proposed_action": "archive" if exists else "pruned",
                "exists": exists,
                "owning_profile": profile_name,
                "owning_modules": modules,
                "launcher_entry_ids": [],
                "mining_redux_import_references": find_text_references("build/installer_profiles.json", modules),
                "installer_dependency_status": "stale non-Mining profile remains" if exists else "pruned from Mining-only installer profile manifest",
                "tests_docs_impact": "Mining-only installer/profile tests must prove Mining Loadout and Mining Signals remain emitted.",
                "rationale": "Explicit non-Mining installer profile boundary was pruned for source-level cleanup.",
            }
        )
    full_exists = "full" in profiles
    full_modules = profile_modules(profiles, "full") if full_exists else []
    non_mining_full = sorted(module for module in full_modules if module not in MINING_MODULES)
    entries.append(
        {
            "path": "build/installer_profiles.json#profiles.full.non_mining_modules",
            "boundary_type": "installer_profile_metadata",
            "classification": "non_mining_profile_metadata" if non_mining_full else "archived_non_mining_profile_metadata",
            "proposed_action": "archive" if non_mining_full else "pruned",
            "exists": bool(non_mining_full),
            "owning_profile": "full",
            "owning_modules": non_mining_full,
            "launcher_entry_ids": [],
            "mining_redux_import_references": find_text_references("build/installer_profiles.json", non_mining_full),
            "installer_dependency_status": "stale full profile names non-Mining modules" if non_mining_full else "full profile non-Mining metadata pruned from Mining-only manifest",
            "tests_docs_impact": "Mining-only installer/profile tests must reject broad full-profile closure.",
            "rationale": "Full profile non-Mining metadata was pruned for strict source-level closure.",
        }
    )
    non_mining_launcher = {
        key: value for key, value in sorted(registry_ids.items()) if value not in MINING_LAUNCHER_ENTRY_IDS
    }
    entries.append(
        {
            "path": "core/skill_registry.py#non_mining_launcher_defaults",
            "boundary_type": "launcher_metadata",
            "classification": "non_mining_launcher_metadata" if non_mining_launcher else "archived_non_mining_launcher_metadata",
            "proposed_action": "archive" if non_mining_launcher else "pruned",
            "exists": bool(non_mining_launcher),
            "owning_modules": sorted(non_mining_launcher),
            "launcher_entry_ids": sorted(non_mining_launcher.values()),
            "mining_redux_import_references": find_text_references("core/skill_registry.py", non_mining_launcher.values()),
            "installer_dependency_status": "stale launcher defaults remain" if non_mining_launcher else "non-Mining launcher defaults pruned; Mining defaults retained",
            "tests_docs_impact": "Launcher visibility tests assert only mining and mining_signals are discoverable for Redux Mining.",
            "rationale": "Built-in launcher defaults were pruned to Mining Redux launch targets.",
        }
    )
    return entries

def shared_entries() -> list[dict[str, Any]]:
    descriptions = {
        "shared/qt": "qt_runtime shared dependency declared by every installer module, including Mining Redux apps.",
        "shared": "Shared configuration, i18n, IPC, logging, and utility runtime imported by Mining Redux paths.",
        "core": "Launcher discovery and compatibility runtime used by Mining Redux entrypoints and tests.",
        "ui": "Shared UI assets/runtime retained unless later proof shows no Mining import dependency.",
        "build/installer_profiles.json": "Installer profile source remains the audit surface until non-Mining metadata is archived.",
        "redux_mining_launcher.py": "Mining Redux launcher entrypoint and hard allowlist.",
    }
    entries: list[dict[str, Any]] = []
    for path in REQUIRED_SHARED_BOUNDARIES:
        entries.append(
            {
                "path": normalize_rel_path(path),
                "boundary_type": "shared_or_mining_runtime",
                "classification": "shared" if path != "redux_mining_launcher.py" else "mining_launcher",
                "proposed_action": "shared" if path != "redux_mining_launcher.py" else "keep",
                "exists": (ROOT / path).exists(),
                "owning_module": None,
                "owning_profiles": ["mining"] if path == "redux_mining_launcher.py" else [],
                "launcher_entry_ids": sorted(MINING_LAUNCHER_ENTRY_IDS) if path == "redux_mining_launcher.py" else [],
                "mining_redux_import_references": find_text_references(path, []),
                "installer_dependency_status": "retained Mining Redux/shared-runtime dependency",
                "tests_docs_impact": "Must remain covered by Mining Redux runtime, launcher, installer, or shared-runtime smoke tests.",
                "rationale": descriptions[path],
            }
        )
    return entries


def build_manifest() -> dict[str, Any]:
    profile_data = load_json(INSTALLER_PROFILES_PATH)
    profiles = profile_data.get("profiles")
    modules = profile_data.get("modules")
    if not isinstance(profiles, Mapping) or not isinstance(modules, Mapping):
        raise BoundaryError("stale profile installer_profiles.json missing profiles/modules objects")
    registry_ids = skill_registry_launcher_ids()
    profiles_by_module = module_profile_index(profiles)

    entries: list[dict[str, Any]] = []
    for module_name in sorted(modules):
        module_config = modules[module_name]
        if not isinstance(module_config, Mapping):
            raise BoundaryError(f"stale profile module config is not an object: {module_name}")
        entries.extend(entry_for_module(module_name, module_config, profiles_by_module, registry_ids))
    entries.extend(archived_non_mining_entries())
    entries.extend(metadata_entries(profiles, registry_ids))
    entries.extend(shared_entries())
    entries = sorted(entries, key=lambda item: (str(item.get("boundary_type")), str(item.get("path"))))

    return {
        "manifest_version": 1,
        "description": "M007/S05 strict Mining Redux source-boundary manifest after applying the approved source cleanup.",
        "workspace_root_name": ROOT.name,
        "generated_by": "build/strict_mining_source_boundary_manifest.py",
        "source_inputs": [
            "build/non_mining_source_prune_manifest.json",
            "build/strict_mining_source_boundary_audit.json",
            "core/skill_registry.py",
            "redux_mining_launcher.py",
            "skills/",
            "tools/",
        ],
        "verification_commands": list(VERIFICATION_COMMANDS),
        "summary": {
            "entry_count": len(entries),
            "removed_non_mining_app_source_roots": sum(1 for item in entries if item.get("classification") == "removed_non_mining_app_source"),
            "live_non_mining_app_source_roots": sum(1 for item in entries if item.get("classification") == "non_mining_app_source"),
            "non_mining_metadata_boundaries": sum(1 for item in entries if str(item.get("classification", "")).startswith("non_mining_") and item.get("classification") != "non_mining_app_source"),
            "shared_or_kept_boundaries": sum(1 for item in entries if item.get("proposed_action") in {"keep", "shared"}),
            "archive_candidates": sum(1 for item in entries if item.get("proposed_action") == "archive"),
            "pruned_metadata_boundaries": sum(1 for item in entries if item.get("proposed_action") == "pruned"),
        },
        "entries": entries,
    }


def manifest_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise BoundaryError("manifest missing entries list")
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise BoundaryError(f"manifest entry {index} is not an object")
        path = entry.get("path")
        if not isinstance(path, str) or not path.strip():
            raise BoundaryError(f"filesystem boundary entry {index} missing non-empty path")
        copied = dict(entry)
        copied["path"] = normalize_rel_path(path)
        normalized.append(copied)
    summary = manifest.get("summary")
    declared_count = summary.get("entry_count") if isinstance(summary, Mapping) else manifest.get("entry_count")
    if declared_count not in {None, len(normalized)}:
        raise BoundaryError(f"filesystem boundary entry_count mismatch: declared={declared_count} actual={len(normalized)}")
    return normalized


def _by_path(entries: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_path: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        path = str(entry.get("path", ""))
        if path in by_path:
            raise BoundaryError(f"filesystem boundary duplicate manifest path: {path}")
        by_path[path] = entry
    return by_path


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    entries = manifest_entries(manifest)
    by_path = _by_path(entries)
    for path in REQUIRED_NON_MINING_ROOTS:
        entry = by_path.get(path)
        if entry is None:
            raise BoundaryError(f"filesystem boundary missing removed non-Mining root classification: {path}")
        if entry.get("classification") == "non_mining_app_source" or entry.get("proposed_action") in {"archive", "delete"}:
            raise BoundaryError(f"filesystem boundary live non-Mining root still requires cleanup: {path}")
        if entry.get("classification") != "removed_non_mining_app_source" or entry.get("proposed_action") != "removed":
            raise BoundaryError(f"filesystem boundary stale removed non-Mining root action: {path}")
        if entry.get("exists") is True:
            raise BoundaryError(f"filesystem boundary non-Mining source root still exists on disk: {path}")
        if entry.get("mining_redux_import_references"):
            raise BoundaryError(f"filesystem boundary Mining Redux still references removed non-Mining root: {path}")

    for path in REQUIRED_METADATA_BOUNDARIES:
        entry = by_path.get(path)
        if entry is None:
            if "profiles" in path:
                raise BoundaryError(f"stale profile missing metadata boundary: {path}")
            raise BoundaryError(f"stale launcher missing metadata boundary: {path}")
        if entry.get("exists") is True or entry.get("proposed_action") in {"archive", "delete"}:
            if "profiles" in path:
                raise BoundaryError(f"stale profile metadata remains live after cleanup: {path}")
            raise BoundaryError(f"stale launcher metadata remains live after cleanup: {path}")
        if entry.get("proposed_action") != "pruned":
            raise BoundaryError(f"stale profile/launcher metadata not marked pruned: {path}")
    for path in REQUIRED_SHARED_BOUNDARIES:
        entry = by_path.get(path)
        if entry is None:
            raise BoundaryError(f"filesystem boundary missing retained shared/runtime classification: {path}")
        if entry.get("proposed_action") not in {"keep", "shared"}:
            raise BoundaryError(f"filesystem boundary shared/runtime path is not retained: {path}")
        if entry.get("exists") is not True:
            raise BoundaryError(f"filesystem boundary retained shared/runtime path missing on disk: {path}")
        if not entry.get("rationale"):
            raise BoundaryError(f"filesystem boundary retained shared/runtime path lacks rationale: {path}")
    for entry in entries:
        refs = entry.get("mining_redux_import_references")
        if refs is None or not isinstance(refs, list):
            raise BoundaryError(f"stale import missing Mining Redux reference audit: {entry.get('path')}")
        if entry.get("classification") in {"non_mining_app_source", "archived_non_mining_app_source"} and refs:
            first = refs[0]
            raise BoundaryError(
                "stale import non-Mining source referenced from Mining Redux path: "
                f"{entry.get('path')} via {first.get('path')}:{first.get('line')} token={first.get('token')!r}"
            )

def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    return load_json(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build/check the strict Mining Redux source-boundary inventory.")
    parser.add_argument("--write", action="store_true", help="write the generated JSON manifest")
    parser.add_argument("--check", action="store_true", help="verify the generated JSON matches disk and is strict")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="manifest path to write/check")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    try:
        generated = build_manifest()
        verify_manifest(generated)
        if args.write:
            write_manifest(manifest_path, generated)
            print(f"PASS strict source boundary manifest written: {rel(manifest_path)} entries={generated['summary']['entry_count']}")
            return 0
        if args.check:
            existing = load_manifest(manifest_path)
            verify_manifest(existing)
            generated_text = json.dumps(generated, indent=2, sort_keys=True) + "\n"
            existing_text = json.dumps(existing, indent=2, sort_keys=True) + "\n"
            if existing_text != generated_text:
                raise BoundaryError(f"filesystem boundary manifest is stale: {rel(manifest_path)}")
            print(
                "PASS strict source boundary manifest check: "
                f"entries={generated['summary']['entry_count']} archive_candidates={generated['summary']['archive_candidates']}"
            )
            return 0
        print(json.dumps(generated, indent=2, sort_keys=True))
        return 0
    except BoundaryError as exc:
        print(f"FAIL strict source boundary manifest: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
