"""Closeout verifier for the strict M007/S05 Mining Redux source boundary.

This command is read/check only.  It verifies the applied strict boundary
manifest, live filesystem/archive state, stale metadata pruning, Redux launcher
visibility, Mining installer profile resolution, and the prerequisite cleanup
verifiers that this stricter contract builds on.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build import strict_mining_source_boundary_manifest as strict_manifest
from build.installer_profiles import ManifestValidationError, load_manifest as load_installer_manifest, resolve_install_plan
from core.launcher_visibility import discover_launcher_skills
import redux_mining_launcher

ROOT = Path(__file__).resolve().parent.parent
STRICT_MANIFEST_PATH = ROOT / "build" / "strict_mining_source_boundary_manifest.json"
STRICT_AUDIT_PATH = ROOT / "build" / "strict_mining_source_boundary_audit.json"
INSTALLER_MANIFEST_PATH = ROOT / "build" / "installer_profiles.json"

MINING_MODULE_IDS = {"mining_loadout", "mining_signals"}
MINING_SKILL_IDS = set(redux_mining_launcher.REDUX_MINING_SKILL_IDS)
MINING_LAUNCHER_ENTRY_IDS = {"mining-loadout", "mining-signals"}
NON_MINING_PROFILE_IDS = {"combat", "full", "reference", "trading"}
NON_MINING_LAUNCHER_ENTRY_IDS = {
    "battle-buddy",
    "cargo-loader",
    "craft-database",
    "dps-calculator",
    "market-finder",
    "mission-database",
    "trade-hub",
}
PRIOR_CLEANUP_COMMANDS = (
    (sys.executable, "-B", "build/verify_generated_artifact_cleanup.py"),
    (sys.executable, "-B", "build/verify_non_mining_source_prune.py"),
)


class VerificationError(RuntimeError):
    """Raised when the strict source-boundary closeout proof fails."""


@dataclass(frozen=True)
class CommandResult:
    args: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def command(self) -> str:
        return " ".join(str(arg) for arg in self.args)

    @property
    def combined_tail(self) -> str:
        return f"{self.stdout}\n{self.stderr}"[-3000:]


Runner = Callable[[Sequence[str], int], CommandResult]


def rel(path: Path, *, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def pass_line(label: str, detail: str) -> None:
    print(f"PASS {label}: {detail}")


def fail_line(label: str, detail: str) -> None:
    print(f"FAIL {label}: {detail}")


def load_json_file(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"{label} missing: {rel(path)}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerificationError(f"{label} malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise VerificationError(f"{label} root must be a JSON object")
    return data


def _normalize_rel_path(value: str, *, root: Path) -> str:
    normalized = value.replace("\\", "/").strip("/")
    if not normalized:
        raise VerificationError("manifest entry path must not be empty")
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise VerificationError(f"path escapes repository root: {value!r}") from exc
    return normalized


def _manifest_entries(data: Mapping[str, Any], *, root: Path = ROOT, label: str = "manifest") -> list[dict[str, Any]]:
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise VerificationError(f"{label} missing entries list")
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise VerificationError(f"{label} entry {index} is not an object")
        path = entry.get("path")
        if not isinstance(path, str) or not path.strip():
            raise VerificationError(f"{label} entry {index} missing non-empty path")
        copied = dict(entry)
        copied["path"] = _normalize_rel_path(path, root=root)
        normalized.append(copied)
    summary = data.get("summary")
    declared = summary.get("entry_count") if isinstance(summary, Mapping) else data.get("entry_count")
    if declared not in {None, len(normalized)}:
        raise VerificationError(f"{label} entry_count mismatch: declared={declared} actual={len(normalized)}")
    return normalized


def _rooted(root: Path, relative_path: str) -> Path:
    normalized = _normalize_rel_path(relative_path, root=root)
    return root / Path(*normalized.split("/"))


def assert_filesystem_boundary(manifest: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, int]:
    entries = _manifest_entries(manifest, root=root, label="strict manifest")
    archived_count = 0
    retained_count = 0
    for entry in entries:
        path = str(entry["path"])
        classification = str(entry.get("classification", ""))
        boundary_type = str(entry.get("boundary_type", ""))
        if classification == "archived_non_mining_app_source":
            archived_count += 1
            if _rooted(root, path).exists():
                raise VerificationError(f"filesystem boundary live non-Mining root remains: {path}")
            archive_path = entry.get("archive_path")
            if not isinstance(archive_path, str) or not archive_path:
                raise VerificationError(f"filesystem boundary archived entry lacks archive_path: {path}")
            if not _rooted(root, archive_path).exists():
                raise VerificationError(f"filesystem boundary archive copy missing for {path}: {archive_path}")
        elif classification in {"mining", "shared_runtime"} or boundary_type == "retained_shared_runtime":
            retained_count += 1
            if not _rooted(root, path).exists():
                raise VerificationError(f"filesystem boundary retained path missing: {path}")
    if archived_count == 0:
        raise VerificationError("filesystem boundary has no archived non-Mining roots to prove")
    return {"archived_non_mining_roots": archived_count, "retained_roots": retained_count}


def assert_audit_matches_manifest(manifest: Mapping[str, Any], audit: Mapping[str, Any]) -> int:
    approved = audit.get("approved_manifest_paths")
    if not isinstance(approved, list) or not all(isinstance(item, str) for item in approved):
        raise VerificationError("strict audit approved_manifest_paths must be a list of strings")
    approved_set = set(approved)
    expected = {
        str(entry["path"])
        for entry in _manifest_entries(manifest, label="strict manifest")
        if entry.get("classification") == "archived_non_mining_app_source"
        or entry.get("proposed_action") in {"archived", "pruned"}
    }
    missing = sorted(expected - approved_set)
    if missing:
        raise VerificationError("strict audit omits approved manifest paths: " + ", ".join(missing[:10]))
    rejected = audit.get("rejected", [])
    if isinstance(rejected, int):
        rejected_count = rejected
    elif isinstance(rejected, list):
        rejected_count = len(rejected)
    else:
        raise VerificationError("strict audit rejected must be a list or integer")
    if rejected_count:
        raise VerificationError(f"strict audit rejected operations are non-zero: {rejected_count}")
    return len(approved_set)


def _string_list(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise VerificationError(f"{context} must be a list of strings")
    return list(value)


def assert_stale_metadata_pruned(installer_manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    modules = installer_manifest.get("modules")
    profiles = installer_manifest.get("profiles")
    if not isinstance(modules, Mapping):
        raise VerificationError("installer manifest modules must be an object")
    if not isinstance(profiles, Mapping):
        raise VerificationError("installer manifest profiles must be an object")

    actual_modules = set(modules)
    if actual_modules != MINING_MODULE_IDS:
        stale = sorted(actual_modules - MINING_MODULE_IDS)
        missing = sorted(MINING_MODULE_IDS - actual_modules)
        raise VerificationError(f"stale metadata module set mismatch: stale={stale} missing={missing}")

    actual_profiles = set(profiles)
    if actual_profiles != {"mining"}:
        stale_profiles = sorted(actual_profiles & NON_MINING_PROFILE_IDS or actual_profiles - {"mining"})
        raise VerificationError(f"stale metadata profiles remain: {stale_profiles}")

    mining_profile = profiles.get("mining")
    if not isinstance(mining_profile, Mapping):
        raise VerificationError("installer mining profile must be an object")
    mining_modules = set(_string_list(mining_profile.get("modules"), context="mining profile modules"))
    if mining_modules != MINING_MODULE_IDS:
        raise VerificationError(f"mining profile modules mismatch: {sorted(mining_modules)}")

    launcher_ids: set[str] = set()
    shortcut_ids: set[str] = set()
    for module_id, module in modules.items():
        if not isinstance(module, Mapping):
            raise VerificationError(f"installer module {module_id!r} must be an object")
        launcher_ids.update(_string_list(module.get("launcher_entry_ids", []), context=f"module {module_id} launcher_entry_ids"))
        shortcuts = module.get("shortcuts", [])
        if not isinstance(shortcuts, list):
            raise VerificationError(f"module {module_id} shortcuts must be a list")
        for index, shortcut in enumerate(shortcuts, start=1):
            if not isinstance(shortcut, Mapping):
                raise VerificationError(f"module {module_id} shortcut #{index} must be an object")
            shortcut_id = shortcut.get("id")
            if not isinstance(shortcut_id, str) or not shortcut_id:
                raise VerificationError(f"module {module_id} shortcut #{index} id must be non-empty")
            shortcut_ids.add(shortcut_id)

    leaked = sorted((launcher_ids | shortcut_ids) & NON_MINING_LAUNCHER_ENTRY_IDS)
    if leaked:
        raise VerificationError("stale metadata non-Mining launcher or shortcut ids remain: " + ", ".join(leaked))
    if launcher_ids != MINING_LAUNCHER_ENTRY_IDS:
        raise VerificationError(f"Mining launcher_entry_ids mismatch: {sorted(launcher_ids)}")
    return {"modules": sorted(actual_modules), "profiles": sorted(actual_profiles), "launcher_entry_ids": sorted(launcher_ids)}


def assert_mining_imports_and_launcher_visibility(*, root: Path = ROOT) -> dict[str, list[str]]:
    import core.launcher_visibility as launcher_visibility  # noqa: F401
    import core.skill_registry as skill_registry  # noqa: F401
    import skill_launcher  # noqa: F401

    discovered = discover_launcher_skills(
        root,
        install_state={"launcher_entry_ids": sorted(MINING_LAUNCHER_ENTRY_IDS)},
    )
    visible_skill_ids = {skill.id for skill in discovered}
    visible_launcher_ids = {skill.launcher_entry_id for skill in discovered if getattr(skill, "launcher_entry_id", "")}
    if visible_skill_ids != MINING_SKILL_IDS:
        raise VerificationError(f"launcher visibility exposed wrong skills: {sorted(visible_skill_ids)}")
    if visible_launcher_ids != MINING_LAUNCHER_ENTRY_IDS:
        raise VerificationError(f"launcher visibility exposed wrong launcher ids: {sorted(visible_launcher_ids)}")

    filtered = redux_mining_launcher.filter_redux_skills(discovered)
    if set(filtered) != MINING_SKILL_IDS:
        raise VerificationError(f"Redux mining launcher filter returned wrong skill ids: {sorted(filtered)}")
    return {"skill_ids": sorted(visible_skill_ids), "launcher_entry_ids": sorted(visible_launcher_ids)}


def assert_installer_mining_smoke(installer_manifest: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, list[str]]:
    try:
        plan = resolve_install_plan("mining", manifest=installer_manifest, root=root)
    except ManifestValidationError as exc:
        raise VerificationError(f"installer mining smoke failed to resolve mining plan: {exc}") from exc
    modules = set(_string_list(plan.get("included_modules"), context="mining install plan included_modules"))
    launcher_ids = set(_string_list(plan.get("launcher_entry_ids"), context="mining install plan launcher_entry_ids"))
    owned_roots = set(_string_list(plan.get("module_owned_file_roots"), context="mining install plan module_owned_file_roots"))
    if modules != MINING_MODULE_IDS:
        raise VerificationError(f"installer mining smoke included wrong modules: {sorted(modules)}")
    if launcher_ids != MINING_LAUNCHER_ENTRY_IDS:
        raise VerificationError(f"installer mining smoke exposed wrong launcher ids: {sorted(launcher_ids)}")
    expected_roots = {"skills/Mining_Loadout", "tools/Mining_Signals"}
    if owned_roots != expected_roots:
        raise VerificationError(f"installer mining smoke owned roots mismatch: {sorted(owned_roots)}")
    return {
        "included_modules": sorted(modules),
        "launcher_entry_ids": sorted(launcher_ids),
        "module_owned_file_roots": sorted(owned_roots),
    }


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("LOCALAPPDATA", str(ROOT / ".localappdata"))
    env.pop("PYTEST_ADDOPTS", None)
    return env


def run(args: Sequence[str], timeout: int = 300) -> CommandResult:
    completed = subprocess.run(
        list(args),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=command_env(),
        check=False,
    )
    return CommandResult(args=tuple(str(arg) for arg in args), returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def assert_prior_cleanup_verifier_composition(*, runner: Runner = run) -> list[str]:
    passed: list[str] = []
    for command in PRIOR_CLEANUP_COMMANDS:
        try:
            result = runner(command, 420)
        except subprocess.TimeoutExpired as exc:
            raise VerificationError(f"prior cleanup verifier timed out: {exc.cmd}") from exc
        if result.returncode != 0:
            raise VerificationError(
                f"prior cleanup verifier failed exit={result.returncode} command={result.command}\n{result.combined_tail}"
            )
        passed.append(result.command)
    return passed


def run_verification(*, runner: Runner = run) -> None:
    manifest = load_json_file(STRICT_MANIFEST_PATH, label="strict source-boundary manifest")
    strict_manifest.verify_manifest(manifest)
    pass_line("strict manifest", "schema and stale profile/launcher/import/filesystem declarations verified")

    filesystem_counts = assert_filesystem_boundary(manifest)
    pass_line(
        "filesystem boundary",
        (
            f"archived_non_mining_roots={filesystem_counts['archived_non_mining_roots']} "
            f"retained_roots={filesystem_counts['retained_roots']}"
        ),
    )

    audit = load_json_file(STRICT_AUDIT_PATH, label="strict source-boundary audit")
    approved_count = assert_audit_matches_manifest(manifest, audit)
    pass_line("stale metadata audit", f"approved_manifest_paths={approved_count} rejected=0")

    installer_manifest = load_installer_manifest(INSTALLER_MANIFEST_PATH)
    metadata = assert_stale_metadata_pruned(installer_manifest)
    pass_line(
        "stale metadata",
        f"profiles={metadata['profiles']} modules={metadata['modules']} launcher_entry_ids={metadata['launcher_entry_ids']}",
    )

    launcher = assert_mining_imports_and_launcher_visibility()
    pass_line(
        "launcher visibility",
        f"skill_ids={launcher['skill_ids']} launcher_entry_ids={launcher['launcher_entry_ids']}",
    )

    installer = assert_installer_mining_smoke(installer_manifest)
    pass_line(
        "installer mining smoke",
        (
            f"included_modules={installer['included_modules']} "
            f"owned_roots={installer['module_owned_file_roots']} "
            f"launcher_entry_ids={installer['launcher_entry_ids']}"
        ),
    )

    prior = assert_prior_cleanup_verifier_composition(runner=runner)
    pass_line("prior cleanup verifier composition", "; ".join(prior))


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Verify the strict Mining Redux source-boundary closeout contract.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    try:
        run_verification()
    except (VerificationError, strict_manifest.BoundaryError, ManifestValidationError, ValueError) as exc:
        fail_line("strict Mining Redux source boundary", str(exc))
        return 1
    print("PASS strict Mining Redux source boundary: repo contract is closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
