from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.installer_profiles import (
    ManifestValidationError,
    load_manifest,
    resolve_install_plan,
    resolve_module_selection_plan,
)
from core.launcher_visibility import discover_launcher_skills
from core.skill_registry import load_launcher_entry_ids_from_install_state

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = ROOT / "build" / "installer_entrypoint.py"
RELEASE_PROFILES = ("mining", "full")
MODIFY_PAIRS = (("mining", "full"), ("full", "mining"))
CATEGORY_IDS = {
    "trading": {"cargo-loader", "market-finder", "trade-hub"},
    "combat": {"dps-calculator", "battle-buddy"},
    "reference": {"mission-database", "craft-database"},
}
MODULE_ROOTS_TO_PROVE = {
    "mining": {"tools/Mining_Signals", "skills/Mining_Loadout"},
    "combat": {"tools/Battle_Buddy", "skills/DPS_Calculator"},
}
SHARED_ROOTS_TO_PROVE = ("core", "shared", "build/installer_entrypoint.py")
REQUIRED_PROFILE_FILES_TO_PROVE = {
    "mining": (
        "tools/Mining_Signals/ocr/sc_ocr/signal_anchor.py",
        "tools/Mining_Signals/ocr/sc_ocr/scan_results_match.py",
        "tools/Mining_Signals/ocr/sc_ocr/signature_diagnostics.py",
        "tools/Mining_Signals/ui/theme.py",
        "tools/Mining_Signals/ui/tutorial_tip.py",
    ),
}
INNO_SCRIPT = ROOT / "build" / "SC_Toolbox_Installer.iss"
INNO_DEFINE_RE = re.compile(r'^\s*#define\s+(?P<name>\w+)\s+"(?P<value>[^"]+)"\s*$', re.MULTILINE)
INNO_SETUP_KEY_RE = re.compile(r'^\s*(?P<key>[A-Za-z][A-Za-z0-9]*)\s*=\s*(?P<value>[^\r\n]+?)\s*$', re.MULTILINE)


def _load_inno_release_artifact_contract() -> dict[str, str]:
    try:
        script = INNO_SCRIPT.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestValidationError(f"Unable to read Inno installer script {INNO_SCRIPT}: {exc}") from exc

    defines = {match.group("name"): match.group("value") for match in INNO_DEFINE_RE.finditer(script)}
    setup_keys = {match.group("key").lower(): match.group("value").strip() for match in INNO_SETUP_KEY_RE.finditer(script)}

    app_version = defines.get("MyAppVersion")
    if not app_version:
        raise ManifestValidationError("Inno installer script is missing #define MyAppVersion \"...\"")

    output_dir = setup_keys.get("outputdir")
    if not output_dir:
        raise ManifestValidationError("Inno installer script is missing [Setup] OutputDir")

    output_base = setup_keys.get("outputbasefilename")
    if not output_base:
        raise ManifestValidationError("Inno installer script is missing [Setup] OutputBaseFilename")

    expected_base = output_base.replace("{#MyAppVersion}", app_version)
    if "{#" in expected_base:
        raise ManifestValidationError(
            f"Inno OutputBaseFilename contains an unsupported unresolved preprocessor token: {output_base!r}"
        )
    if app_version not in expected_base:
        raise ManifestValidationError(
            f"Inno OutputBaseFilename must include MyAppVersion; got {output_base!r} for version {app_version!r}"
        )

    output_path = Path("build") / output_dir / f"{expected_base}.exe"
    return {
        "app_version": app_version,
        "output_dir": str((Path("build") / output_dir).as_posix()),
        "output_base_filename": expected_base,
        "expected_filename": f"{expected_base}.exe",
        "expected_path": output_path.as_posix(),
    }


def _run_entrypoint(*args: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-B", str(ENTRYPOINT), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "installer_entrypoint failed").strip()
        raise ManifestValidationError(detail)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"installer_entrypoint emitted malformed JSON: {exc}") from exc


def _manifest_launcher_ids(manifest: dict[str, Any], profile: str) -> set[str]:
    plan = resolve_install_plan(profile, manifest=manifest, root=ROOT)
    return set(plan.get("launcher_entry_ids", []))


def _visible_launcher_ids(install_state_path: Path) -> set[str]:
    return {
        skill.launcher_entry_id
        for skill in discover_launcher_skills(ROOT, install_state=install_state_path)
        if getattr(skill, "launcher_entry_id", None)
    }


def _available_release_profiles(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    profiles = manifest.get("profiles", {})
    if not isinstance(profiles, Mapping):
        raise ManifestValidationError("manifest 'profiles' must be a JSON object")
    ordered = tuple(profile for profile in RELEASE_PROFILES if profile in profiles)
    extras = tuple(sorted(str(profile) for profile in profiles if profile not in RELEASE_PROFILES))
    return ordered + extras


def _is_strict_mining_only_manifest(manifest: Mapping[str, Any]) -> bool:
    modules = manifest.get("modules", {})
    profiles = manifest.get("profiles", {})
    if not isinstance(modules, Mapping) or not isinstance(profiles, Mapping):
        return False
    return set(modules) == {"mining_loadout", "mining_signals"} and set(profiles) == {"mining"}


def _manifest_module_owned_paths(manifest: Mapping[str, Any]) -> set[str]:
    modules = manifest.get("modules", {})
    if not isinstance(modules, Mapping):
        raise ManifestValidationError("manifest 'modules' must be a JSON object")

    owned_paths: set[str] = set()
    for module_id, module in modules.items():
        if not isinstance(module, Mapping):
            raise ManifestValidationError(f"module '{module_id}' must be a JSON object")
        for owned_path in module.get("owned_paths", []):
            if not isinstance(owned_path, str):
                raise ManifestValidationError(f"module '{module_id}' owned_paths entries must be strings")
            owned_paths.add(owned_path.replace("\\", "/"))
    return owned_paths


def _copy_payload_path(install_root: Path, relative_path: str, *, context: str) -> None:
    source = ROOT / relative_path
    if not source.exists():
        raise ManifestValidationError(f"{context} payload source is missing: {relative_path}")
    target = install_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)


def _copy_broad_payload(install_root: Path, manifest: Mapping[str, Any]) -> None:
    for relative_path in sorted(_manifest_module_owned_paths(manifest).union(SHARED_ROOTS_TO_PROVE)):
        _copy_payload_path(install_root, relative_path, context="broad")


def _copy_module_payload_roots(install_root: Path, manifest: Mapping[str, Any], module_ids: Sequence[str]) -> list[str]:
    modules = manifest.get("modules", {})
    if not isinstance(modules, Mapping):
        raise ManifestValidationError("manifest 'modules' must be a JSON object")

    copied_roots: list[str] = []
    for module_id in module_ids:
        module = modules.get(module_id)
        if not isinstance(module, Mapping):
            raise ManifestValidationError(f"module '{module_id}' must be a JSON object")
        for relative_path in module.get("owned_paths", []):
            if not isinstance(relative_path, str):
                raise ManifestValidationError(f"module '{module_id}' owned_paths entries must be strings")
            normalized = relative_path.replace("\\", "/")
            _copy_payload_path(install_root, normalized, context=f"module '{module_id}'")
            copied_roots.append(normalized)
    return sorted(copied_roots)


def _relative_exists(install_root: Path, relative_path: str) -> bool:
    return (install_root / Path(*relative_path.split("/"))).exists()


def _assert_inno_module_checkbox_handoff() -> dict[str, bool]:
    script = INNO_SCRIPT.read_text(encoding="utf-8")
    if "[Code]" not in script:
        return {
            "code_section_present": False,
            "creates_module_checkbox_page": False,
            "rejects_empty_module_selection": False,
            "serializes_selected_module_ids": False,
            "uses_bundled_entrypoint": False,
            "passes_modules_to_helper": False,
            "applies_selected_modules_to_install_root": False,
            "delegates_install_state_to_runtime_apply": True,
            "preselects_existing_state": False,
            "emits_preselected_modules": False,
            "calls_module_apply_after_install": False,
            "keeps_profile_apply_out_of_inno": True,
        }
    code_section = script.split("[Code]", 1)[1].split("[UninstallDelete]", 1)[0]
    checks = {
        "creates_module_checkbox_page": "Choose SC Toolbox modules" in code_section,
        "rejects_empty_module_selection": "Select at least one SC Toolbox module before continuing." in code_section,
        "serializes_selected_module_ids": "SelectedModules := SelectedModuleIds();" in code_section,
        "uses_bundled_entrypoint": "{app}\\build\\installer_entrypoint.py" in code_section,
        "passes_modules_to_helper": "--modules" in code_section,
        "applies_selected_modules_to_install_root": "--apply-install-root" in code_section
        and "AddQuotes(InstallRoot)" in code_section,
        "delegates_install_state_to_runtime_apply": "--emit-install-state" not in code_section,
        "preselects_existing_state": "--preselect-modules-from-state" in code_section,
        "emits_preselected_modules": "--emit-preselected-modules" in code_section,
        "calls_module_apply_after_install": "ApplySelectedModulesToInstallRoot();" in code_section,
        "keeps_profile_apply_out_of_inno": "ApplySelectedProfileToInstallRoot" not in code_section
        and "SelectedProfileId" not in code_section,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ManifestValidationError(f"Inno module checkbox handoff is missing checks: {failed}")
    return checks


def _validate_install_state_launcher_visibility(
    install_state_path: Path,
    profile: str | None,
    manifest: Mapping[str, Any],
    *,
    context: str,
    expected_launcher_ids: set[str] | None = None,
) -> tuple[dict[str, Any], set[str]]:
    install_state = json.loads(install_state_path.read_text(encoding="utf-8"))
    expected_ids = expected_launcher_ids if expected_launcher_ids is not None else _manifest_launcher_ids(dict(manifest), profile or "")
    allowed_ids = set(load_launcher_entry_ids_from_install_state(install_state_path) or [])
    visible_ids = _visible_launcher_ids(install_state_path)
    if allowed_ids != expected_ids:
        raise ManifestValidationError(
            f"{context} install_state launcher ids for {profile} did not match manifest: "
            f"expected={sorted(expected_ids)} actual={sorted(allowed_ids)}"
        )
    if visible_ids != expected_ids:
        raise ManifestValidationError(
            f"{context} launcher visibility for {profile} did not match install_state: "
            f"expected={sorted(expected_ids)} actual={sorted(visible_ids)}"
        )
    return install_state, visible_ids


def _runtime_output_summary(
    install_root: Path,
    manifest: Mapping[str, Any],
    install_state: Mapping[str, Any],
    visible_ids: set[str],
    removed_roots: Sequence[str],
) -> dict[str, Any]:
    install_state_path = install_root / "install_state.json"
    present_roots = sorted(path for path in _manifest_module_owned_paths(manifest) if _relative_exists(install_root, path))
    state_shortcuts = sorted(install_state.get("shortcut_ids", []))

    return {
        "install_state_path": install_state_path.name,
        "install_state_profile": install_state.get("selected_profile"),
        "launcher_entry_ids": sorted(visible_ids),
        "shortcut_ids": state_shortcuts,
        "present_roots": present_roots,
        "removed_roots": sorted(removed_roots),
        "shared_roots_present": sorted(path for path in SHARED_ROOTS_TO_PROVE if _relative_exists(install_root, path)),
        "required_files_present": sorted(
            path
            for path in REQUIRED_PROFILE_FILES_TO_PROVE.get(str(install_state.get("selected_profile")), ())
            if _relative_exists(install_root, path)
        ),
        "included_modules": list(install_state.get("included_modules", [])),
        "omitted_modules": list(install_state.get("omitted_modules", [])),
        "ownership_ledger_modules": sorted(
            install_state.get("ownership_ledger", {}).get("module_owned_file_roots", {})
            if isinstance(install_state.get("ownership_ledger"), Mapping)
            else []
        ),
    }


def _prove_runtime_profile_output(profile: str, manifest: Mapping[str, Any], temp_root: Path) -> dict[str, Any]:
    install_root = temp_root / profile
    _copy_broad_payload(install_root, manifest)
    plan = _run_entrypoint(profile, "--apply-install-root", str(install_root))
    runtime_apply = plan.get("runtime_apply")
    if not isinstance(runtime_apply, dict):
        raise ManifestValidationError(f"runtime helper did not report apply-install-root for {profile}")

    install_state_path = install_root / "install_state.json"
    install_state, visible_ids = _validate_install_state_launcher_visibility(
        install_state_path,
        profile,
        manifest,
        context="runtime",
    )
    removed_roots = runtime_apply.get("pruning", {}).get("removed_file_roots", [])
    return _runtime_output_summary(install_root, manifest, install_state, visible_ids, removed_roots)


def _prove_module_selection_dry_run(temp_root: Path) -> dict[str, Any]:
    install_state_path = temp_root / "module-selection" / "install_state.json"
    plan = _run_entrypoint(
        "--modules",
        "mining_signals,battle_buddy",
        "--emit-install-state",
        str(install_state_path),
    )
    install_state = json.loads(install_state_path.read_text(encoding="utf-8"))
    if plan.get("selected_profile") is not None:
        raise ManifestValidationError("module selection dry-run unexpectedly selected a profile")
    if plan.get("selected_modules") != ["battle_buddy", "mining_signals"]:
        raise ManifestValidationError(
            "module selection dry-run did not serialize canonical module order: "
            f"{plan.get('selected_modules')!r}"
        )
    if install_state.get("included_modules") != ["battle_buddy", "mining_signals"]:
        raise ManifestValidationError(
            "module selection install_state did not record selected module ids: "
            f"{install_state.get('included_modules')!r}"
        )
    return {
        "selected_profile": plan.get("selected_profile"),
        "selected_modules": list(plan.get("selected_modules", [])),
        "included_modules": list(install_state.get("included_modules", [])),
        "launcher_entry_ids": sorted(install_state.get("launcher_entry_ids", [])),
        "install_state_path": install_state_path.name,
    }


def _prove_module_preselection(temp_root: Path) -> dict[str, Any]:
    install_root = temp_root / "preselection"
    install_root.mkdir(parents=True, exist_ok=True)
    install_state_path = install_root / "install_state.json"
    preselected_path = temp_root / "handoff" / "preselected_modules.txt"
    install_state_path.write_text(
        json.dumps({"included_modules": ["mining_signals", "battle_buddy"]}),
        encoding="utf-8",
    )
    result = _run_entrypoint(
        "--preselect-modules-from-state",
        str(install_state_path),
        "--emit-preselected-modules",
        str(preselected_path),
    )
    handoff = preselected_path.read_text(encoding="utf-8").strip()
    if handoff != "battle_buddy,mining_signals":
        raise ManifestValidationError(f"preselection handoff was not canonical: {handoff!r}")
    return {
        "action": result.get("action"),
        "install_root": Path(result.get("install_root", "")).name,
        "preselection_source": result.get("preselection_source"),
        "selected_modules": list(result.get("selected_modules", [])),
        "selected_module_ids": result.get("selected_module_ids"),
        "handoff": handoff,
    }


def _selected_module_runtime_summary(
    install_root: Path,
    manifest: Mapping[str, Any],
    runtime_apply: Mapping[str, Any],
    expected_launcher_ids: set[str],
    *,
    context: str,
    removed_roots: Sequence[str],
) -> dict[str, Any]:
    install_state_path = install_root / "install_state.json"
    install_state, visible_ids = _validate_install_state_launcher_visibility(
        install_state_path,
        None,
        manifest,
        context=context,
        expected_launcher_ids=expected_launcher_ids,
    )
    summary = _runtime_output_summary(install_root, manifest, install_state, visible_ids, removed_roots)
    summary["action"] = runtime_apply.get("action")
    summary["selected_modules"] = list(runtime_apply.get("selected_modules", []))
    summary["current_modules"] = list(runtime_apply.get("current_modules", []))
    summary["target_modules"] = list(runtime_apply.get("target_modules", []))
    summary["added_modules"] = list(runtime_apply.get("added_modules", []))
    summary["removed_modules"] = list(runtime_apply.get("removed_modules", []))
    summary["retained_modules"] = list(runtime_apply.get("retained_modules", []))
    summary["missing_roots"] = sorted(runtime_apply.get("missing_file_roots", []))
    summary["protected_shared_roots"] = sorted(runtime_apply.get("protected_shared_roots", []))
    summary["shortcut_delta"] = dict(runtime_apply.get("shortcut_delta", {}))
    summary["launcher_delta"] = dict(runtime_apply.get("launcher_delta", {}))
    summary["modify_plan_summary"] = dict(runtime_apply.get("modify_plan_summary", {}))
    return summary


def _prove_module_runtime_apply(manifest: Mapping[str, Any], temp_root: Path) -> dict[str, Any]:
    install_root = temp_root / "selected-module-apply"
    _copy_broad_payload(install_root, manifest)
    selected_modules = "mining_signals,battle_buddy"
    expected_plan = resolve_module_selection_plan(selected_modules, manifest=manifest, root=install_root)
    plan = _run_entrypoint("--modules", selected_modules, "--apply-install-root", str(install_root))
    runtime_apply = plan.get("runtime_apply")
    if not isinstance(runtime_apply, dict):
        raise ManifestValidationError("selected-module runtime helper did not report apply result")
    if runtime_apply.get("action") != "apply-module-selection-install-root":
        raise ManifestValidationError(
            "selected-module runtime helper used unexpected action: "
            f"{runtime_apply.get('action')!r}"
        )

    install_state_path = install_root / "install_state.json"
    install_state, visible_ids = _validate_install_state_launcher_visibility(
        install_state_path,
        None,
        manifest,
        context="selected-module runtime",
        expected_launcher_ids=set(expected_plan["launcher_entry_ids"]),
    )
    if install_state.get("selected_profile") is not None:
        raise ManifestValidationError("selected-module runtime state unexpectedly recorded a profile")
    if install_state.get("included_modules") != expected_plan["included_modules"]:
        raise ManifestValidationError(
            "selected-module runtime state included modules did not match manifest plan: "
            f"expected={expected_plan['included_modules']!r} actual={install_state.get('included_modules')!r}"
        )
    if install_state.get("omitted_modules") != expected_plan["omitted_modules"]:
        raise ManifestValidationError(
            "selected-module runtime state omitted modules did not match manifest plan: "
            f"expected={expected_plan['omitted_modules']!r} actual={install_state.get('omitted_modules')!r}"
        )

    removed_roots = runtime_apply.get("pruning", {}).get("removed_file_roots", [])
    summary = _runtime_output_summary(install_root, manifest, install_state, visible_ids, removed_roots)
    summary["action"] = runtime_apply["action"]
    summary["selected_modules"] = list(runtime_apply.get("selected_modules", []))
    summary["expected_selected_modules"] = list(expected_plan["selected_modules"])
    summary["runtime_included_modules"] = list(runtime_apply.get("included_modules", []))
    summary["runtime_omitted_modules"] = list(runtime_apply.get("omitted_modules", []))
    summary["runtime_shortcut_ids"] = sorted(runtime_apply.get("shortcut_ids", []))
    summary["runtime_launcher_entry_ids"] = sorted(runtime_apply.get("launcher_entry_ids", []))
    return summary


def _prove_selected_module_add_remove_sequence(manifest: Mapping[str, Any], temp_root: Path) -> dict[str, Any]:
    install_root = temp_root / "selected-module-add-remove"
    _copy_broad_payload(install_root, manifest)

    mining_modules = "mining_signals,mining_loadout"
    mining_combat_modules = "mining_signals,mining_loadout,dps_calculator,battle_buddy"
    mining_plan = resolve_module_selection_plan(mining_modules, manifest=manifest, root=install_root)
    mining_combat_plan = resolve_module_selection_plan(mining_combat_modules, manifest=manifest, root=install_root)

    fresh_plan = _run_entrypoint("--modules", mining_modules, "--apply-install-root", str(install_root))
    fresh_apply = fresh_plan.get("runtime_apply")
    if not isinstance(fresh_apply, dict) or fresh_apply.get("action") != "apply-module-selection-install-root":
        raise ManifestValidationError("selected-module add/remove sequence did not start with fresh custom apply")
    fresh_removed_roots = fresh_apply.get("pruning", {}).get("removed_file_roots", [])
    fresh_summary = _selected_module_runtime_summary(
        install_root,
        manifest,
        fresh_apply,
        set(mining_plan["launcher_entry_ids"]),
        context="selected-module fresh mining runtime",
        removed_roots=fresh_removed_roots,
    )
    fresh_summary["expected_selected_modules"] = list(mining_plan["selected_modules"])
    fresh_summary["runtime_included_modules"] = list(fresh_apply.get("included_modules", []))
    fresh_summary["runtime_omitted_modules"] = list(fresh_apply.get("omitted_modules", []))
    fresh_summary["runtime_shortcut_ids"] = sorted(fresh_apply.get("shortcut_ids", []))
    fresh_summary["runtime_launcher_entry_ids"] = sorted(fresh_apply.get("launcher_entry_ids", []))

    added_payload_roots = _copy_module_payload_roots(
        install_root,
        manifest,
        ["dps_calculator", "battle_buddy"],
    )
    add_plan = _run_entrypoint(
        "--modules",
        mining_combat_modules,
        "--apply-install-root",
        str(install_root),
        "--installed-state",
        str(install_root / "install_state.json"),
    )
    add_apply = add_plan.get("runtime_apply")
    if not isinstance(add_apply, dict) or add_apply.get("action") != "apply-modify-module-selection":
        raise ManifestValidationError("selected-module add modify did not report modify action")
    add_summary = _selected_module_runtime_summary(
        install_root,
        manifest,
        add_apply,
        set(mining_combat_plan["launcher_entry_ids"]),
        context="selected-module add combat runtime",
        removed_roots=add_apply.get("removed_file_roots", []),
    )
    add_summary["copied_payload_roots"] = added_payload_roots

    remove_plan = _run_entrypoint(
        "--modules",
        mining_modules,
        "--apply-install-root",
        str(install_root),
        "--installed-state",
        str(install_root / "install_state.json"),
    )
    remove_apply = remove_plan.get("runtime_apply")
    if not isinstance(remove_apply, dict) or remove_apply.get("action") != "apply-modify-module-selection":
        raise ManifestValidationError("selected-module remove modify did not report modify action")
    remove_summary = _selected_module_runtime_summary(
        install_root,
        manifest,
        remove_apply,
        set(mining_plan["launcher_entry_ids"]),
        context="selected-module remove combat runtime",
        removed_roots=remove_apply.get("removed_file_roots", []),
    )

    return {
        "install_root": install_root.name,
        "fresh_custom_apply": fresh_summary,
        "add_combat_modify": add_summary,
        "remove_combat_modify": remove_summary,
    }


def _prove_invalid_selected_module_modify_failure(manifest: Mapping[str, Any], temp_root: Path) -> dict[str, Any]:
    install_root = temp_root / "invalid-selected-module-modify"
    _copy_broad_payload(install_root, manifest)
    _run_entrypoint("--modules", "mining_signals,mining_loadout,dps_calculator,battle_buddy", "--apply-install-root", str(install_root))
    state_path = install_root / "install_state.json"
    installed_state = json.loads(state_path.read_text(encoding="utf-8"))
    installed_state["ownership_ledger"]["module_owned_file_roots"]["battle_buddy"] = ["shared/qt"]
    state_path.write_text(json.dumps(installed_state, indent=2, sort_keys=True), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ENTRYPOINT),
            "--modules",
            "mining_signals,mining_loadout",
            "--apply-install-root",
            str(install_root),
            "--installed-state",
            str(state_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        raise ManifestValidationError("invalid selected-module modify state unexpectedly succeeded")
    stderr = completed.stderr.strip()
    expected_fragments = [
        "ManifestValidationError",
        "modify transition",
        "installed state",
        "claims protected shared root 'shared/qt'",
    ]
    missing_fragments = [fragment for fragment in expected_fragments if fragment not in stderr]
    if missing_fragments:
        raise ManifestValidationError(
            f"invalid selected-module modify failure omitted context {missing_fragments}: {stderr}"
        )
    if not (install_root / "tools" / "Battle_Buddy").exists() or not (install_root / "shared" / "qt").exists():
        raise ManifestValidationError("invalid selected-module modify mutated protected roots before failing")
    return {
        "action": "apply-modify-module-selection",
        "returncode": completed.returncode,
        "stderr_contains": expected_fragments,
        "stderr": stderr,
        "protected_roots_preserved": ["shared/qt"],
        "module_roots_preserved": ["tools/Battle_Buddy", "skills/DPS_Calculator"],
    }


def _prove_full_to_mining_modify_output(manifest: Mapping[str, Any], temp_root: Path) -> dict[str, Any]:
    install_root = temp_root / "full-to-mining-modify"
    _copy_broad_payload(install_root, manifest)

    full_plan = _run_entrypoint("full", "--apply-install-root", str(install_root))
    full_runtime_apply = full_plan.get("runtime_apply")
    if not isinstance(full_runtime_apply, dict):
        raise ManifestValidationError("runtime helper did not report initial full apply-install-root")

    install_state_path = install_root / "install_state.json"
    full_state, full_visible_ids = _validate_install_state_launcher_visibility(
        install_state_path,
        "full",
        manifest,
        context="initial full runtime",
    )
    if full_state.get("selected_profile") != "full":
        raise ManifestValidationError("initial full runtime state did not record selected_profile='full'")
    if "battle-buddy" not in full_visible_ids:
        raise ManifestValidationError("initial full runtime state did not expose full-only launcher entries")

    modify_plan = _run_entrypoint(
        "mining",
        "--apply-install-root",
        str(install_root),
        "--installed-state",
        str(install_state_path),
    )
    runtime_apply = modify_plan.get("runtime_apply")
    if not isinstance(runtime_apply, dict):
        raise ManifestValidationError("runtime helper did not report full-to-mining modify apply-install-root")
    if runtime_apply.get("action") != "apply-modify-install-root":
        raise ManifestValidationError(f"full-to-mining helper used unexpected action: {runtime_apply.get('action')!r}")
    if runtime_apply.get("current_profile") != "full" or runtime_apply.get("target_profile") != "mining":
        raise ManifestValidationError(
            "full-to-mining helper reported wrong transition: "
            f"{runtime_apply.get('current_profile')!r}->{runtime_apply.get('target_profile')!r}"
        )

    install_state, visible_ids = _validate_install_state_launcher_visibility(
        install_state_path,
        "mining",
        manifest,
        context="full-to-mining modify runtime",
    )
    removed_roots = runtime_apply.get("removal", {}).get("removed_file_roots", [])
    missing_roots = runtime_apply.get("removal", {}).get("missing_file_roots", [])
    if missing_roots:
        raise ManifestValidationError(f"full-to-mining modify missed expected file roots: {sorted(missing_roots)}")

    summary = _runtime_output_summary(install_root, manifest, install_state, visible_ids, removed_roots)
    summary["action"] = runtime_apply["action"]
    summary["current_profile"] = runtime_apply["current_profile"]
    summary["target_profile"] = runtime_apply["target_profile"]
    summary["modify_plan_summary"] = runtime_apply.get("modify_plan_summary", {})
    summary["missing_roots"] = sorted(missing_roots)
    return summary


def _dry_run_release_evidence(dry_runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "selected_profile": plan.get("selected_profile"),
            "included_modules": list(plan.get("included_modules", [])),
            "omitted_modules": list(plan.get("omitted_modules", [])),
            "launcher_entry_ids": sorted(plan.get("launcher_entry_ids", [])),
            "shortcut_ids": sorted(plan.get("shortcut_ids", [])),
        }
        for name, plan in sorted(dry_runs.items())
    }


def _modify_run_release_evidence(modify_runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "current_profile": plan.get("current_profile"),
            "target_profile": plan.get("target_profile"),
            "added_modules": list(plan.get("added_modules", [])),
            "removed_modules": list(plan.get("removed_modules", [])),
            "retained_modules": list(plan.get("retained_modules", [])),
            "add_launcher_entry_ids": sorted(plan.get("add_launcher_entry_ids", [])),
            "remove_launcher_entry_ids": sorted(plan.get("remove_launcher_entry_ids", [])),
            "retained_launcher_entry_ids": sorted(plan.get("retained_launcher_entry_ids", [])),
            "remove_file_roots": sorted(plan.get("remove_file_roots", [])),
        }
        for name, plan in sorted(modify_runs.items())
    }


def _release_evidence_payload(
    *,
    selected_profile: str,
    release_artifact: Mapping[str, str],
    dry_runs: Mapping[str, Mapping[str, Any]],
    modify_runs: Mapping[str, Mapping[str, Any]],
    visible_ids: set[str],
    expected_profile_ids: set[str],
    full_ids: set[str],
    hidden_for_mining: Sequence[str],
    runtime_outputs: Mapping[str, Mapping[str, Any]],
    full_to_mining_modify: Mapping[str, Any],
    broad_payload_roots: Sequence[str],
    module_selection_dry_run: Mapping[str, Any],
    module_preselection: Mapping[str, Any],
    module_runtime_apply: Mapping[str, Any],
    module_add_remove_sequence: Mapping[str, Any],
    invalid_module_modify_failure: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "selected_profile": selected_profile,
        "release_artifact": dict(release_artifact),
        "dry_run_profiles": _dry_run_release_evidence(dry_runs),
        "broad_payload": {
            "source": "manifest-owned runtime files",
            "module_owned_roots": sorted(broad_payload_roots),
            "shared_roots": sorted(SHARED_ROOTS_TO_PROVE),
            "proved_profiles": sorted(runtime_outputs),
        },
        "install_state": {
            "path": "install_state.json",
            "profile": selected_profile,
            "launcher_entry_ids": sorted(expected_profile_ids),
        },
        "visible_launcher_ids": sorted(visible_ids),
        "hidden_launcher_ids": list(hidden_for_mining) if selected_profile == "mining" else [],
        "full_launcher_inventory": sorted(full_ids),
        "runtime_output_proof": dict(runtime_outputs),
        "modify_runs": _modify_run_release_evidence(modify_runs),
        "modify_runtime_proof": {
            "full->mining": dict(full_to_mining_modify),
        },
        "module_selection_dry_run": dict(module_selection_dry_run),
        "module_preselection": dict(module_preselection),
        "module_runtime_apply": dict(module_runtime_apply),
        "module_add_remove_sequence": dict(module_add_remove_sequence),
        "invalid_module_modify_failure": dict(invalid_module_modify_failure),
    }


def _run_strict_mining_only_smoke(profile: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if profile != "mining":
        raise ManifestValidationError(f"unknown installer profile '{profile}'")

    dry_runs = {"mining": _run_entrypoint("mining")}
    inno_handoff = _assert_inno_module_checkbox_handoff()
    release_artifact = _load_inno_release_artifact_contract()

    with tempfile.TemporaryDirectory(prefix="sc-toolbox-installer-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        install_state_path = temp_root / "install_state.json"
        _run_entrypoint(profile, "--emit-install-state", str(install_state_path))
        json.loads(install_state_path.read_text(encoding="utf-8"))
        allowed_ids = set(load_launcher_entry_ids_from_install_state(install_state_path) or [])
        visible_ids = _visible_launcher_ids(install_state_path)
        runtime_outputs = {"mining": _prove_runtime_profile_output("mining", manifest, temp_root / "runtime")}

    expected_profile_ids = _manifest_launcher_ids(dict(manifest), profile)
    if allowed_ids != expected_profile_ids:
        raise ManifestValidationError(
            f"install_state launcher ids for {profile} did not match manifest: "
            f"expected={sorted(expected_profile_ids)} actual={sorted(allowed_ids)}"
        )
    if visible_ids != expected_profile_ids:
        raise ManifestValidationError(
            f"launcher visibility for {profile} did not match install_state: "
            f"expected={sorted(expected_profile_ids)} actual={sorted(visible_ids)}"
        )

    hidden_for_mining = sorted(set().union(*CATEGORY_IDS.values()) - expected_profile_ids)
    leaked = sorted(visible_ids.intersection(set().union(*CATEGORY_IDS.values())))
    if leaked:
        raise ManifestValidationError(f"mining profile leaked non-mining launcher entries: {leaked}")

    mining_output = runtime_outputs["mining"]
    mining_present_roots = set(mining_output["present_roots"])
    missing_mining_roots = sorted(MODULE_ROOTS_TO_PROVE["mining"] - mining_present_roots)
    if missing_mining_roots:
        raise ManifestValidationError(f"mining runtime output lost selected roots: {missing_mining_roots}")
    missing_shared = sorted(set(SHARED_ROOTS_TO_PROVE) - set(mining_output["shared_roots_present"]))
    if missing_shared:
        raise ManifestValidationError(f"mining runtime output lost shared/runtime roots: {missing_shared}")

    release_evidence = {
        "selected_profile": profile,
        "release_artifact": dict(release_artifact),
        "dry_run_profiles": _dry_run_release_evidence(dry_runs),
        "broad_payload": {
            "source": "manifest-owned runtime files",
            "module_owned_roots": sorted(_manifest_module_owned_paths(manifest)),
            "shared_roots": sorted(SHARED_ROOTS_TO_PROVE),
            "proved_profiles": ["mining"],
        },
        "install_state": {
            "path": "install_state.json",
            "profile": profile,
            "launcher_entry_ids": sorted(expected_profile_ids),
        },
        "visible_launcher_ids": sorted(visible_ids),
        "hidden_launcher_ids": hidden_for_mining,
        "full_launcher_inventory": [],
        "runtime_output_proof": runtime_outputs,
        "strict_mining_only": True,
    }

    return {
        "ok": True,
        "profile": profile,
        "dry_run_profiles": ["mining"],
        "modify_runs": [],
        "install_state_profile": profile,
        "launcher_entry_ids": sorted(visible_ids),
        "mining_hidden_launcher_entry_ids": hidden_for_mining,
        "full_launcher_entry_ids": [],
        "entrypoint_plan_profile": profile,
        "inno_module_checkbox_handoff": inno_handoff,
        "release_artifact": release_artifact,
        "release_evidence": release_evidence,
        "runtime_outputs": runtime_outputs,
        "strict_mining_only": True,
        "mining_removed_roots": sorted(mining_output["removed_roots"]),
        "mining_present_roots": sorted(mining_output["present_roots"]),
        "full_present_roots": [],
    }


def run_smoke(profile: str = "mining") -> dict[str, Any]:
    manifest = load_manifest(ROOT / "build" / "installer_profiles.json")
    if profile not in manifest.get("profiles", {}):
        raise ManifestValidationError(f"unknown installer profile '{profile}'")
    if _is_strict_mining_only_manifest(manifest):
        return _run_strict_mining_only_smoke(profile, manifest)

    release_profiles = _available_release_profiles(manifest)
    dry_runs = {name: _run_entrypoint(name) for name in release_profiles}
    modify_pairs = tuple((current, target) for current, target in MODIFY_PAIRS if current in release_profiles and target in release_profiles)
    modify_runs = {
        f"{current}->{target}": _run_entrypoint("--modify-from", current, target)
        for current, target in modify_pairs
    }

    inno_handoff = _assert_inno_module_checkbox_handoff()
    release_artifact = _load_inno_release_artifact_contract()

    with tempfile.TemporaryDirectory(prefix="sc-toolbox-installer-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        install_state_path = temp_root / "install_state.json"
        emitted_plan = _run_entrypoint(profile, "--emit-install-state", str(install_state_path))
        json.loads(install_state_path.read_text(encoding="utf-8"))
        allowed_ids = set(load_launcher_entry_ids_from_install_state(install_state_path) or [])
        visible_ids = _visible_launcher_ids(install_state_path)
        runtime_outputs = {
            name: _prove_runtime_profile_output(name, manifest, temp_root / "runtime")
            for name in release_profiles
        }
        full_to_mining_modify = _prove_full_to_mining_modify_output(manifest, temp_root / "runtime")
        module_selection_dry_run = _prove_module_selection_dry_run(temp_root)
        module_preselection = _prove_module_preselection(temp_root)
        module_runtime_apply = _prove_module_runtime_apply(manifest, temp_root)
        module_add_remove_sequence = _prove_selected_module_add_remove_sequence(manifest, temp_root)
        invalid_module_modify_failure = _prove_invalid_selected_module_modify_failure(manifest, temp_root)

    expected_profile_ids = _manifest_launcher_ids(manifest, profile)
    full_ids = _manifest_launcher_ids(manifest, "full")
    if allowed_ids != expected_profile_ids:
        raise ManifestValidationError(
            f"install_state launcher ids for {profile} did not match manifest: "
            f"expected={sorted(expected_profile_ids)} actual={sorted(allowed_ids)}"
        )
    if visible_ids != expected_profile_ids:
        raise ManifestValidationError(
            f"launcher visibility for {profile} did not match install_state: "
            f"expected={sorted(expected_profile_ids)} actual={sorted(visible_ids)}"
        )

    hidden_for_mining = sorted(set().union(*CATEGORY_IDS.values()) - expected_profile_ids)
    if profile == "mining":
        leaked = sorted(visible_ids.intersection(set().union(*CATEGORY_IDS.values())))
        if leaked:
            raise ManifestValidationError(f"mining profile leaked non-mining launcher entries: {leaked}")

    if set(dry_runs["full"].get("launcher_entry_ids", [])) != full_ids:
        raise ManifestValidationError("full dry-run launcher inventory did not match manifest")

    mining_output = runtime_outputs["mining"]
    full_output = runtime_outputs["full"]
    mining_omitted_roots = set().union(*MODULE_ROOTS_TO_PROVE.values()) - MODULE_ROOTS_TO_PROVE["mining"]
    mining_present_roots = set(mining_output["present_roots"])
    full_present_roots = set(full_output["present_roots"])
    missing_mining_roots = sorted(MODULE_ROOTS_TO_PROVE["mining"] - mining_present_roots)
    leaked_mining_roots = sorted(mining_omitted_roots.intersection(mining_present_roots))
    missing_full_roots = sorted(_manifest_module_owned_paths(manifest) - full_present_roots)
    if missing_mining_roots:
        raise ManifestValidationError(f"mining runtime output lost selected roots: {missing_mining_roots}")
    if leaked_mining_roots:
        raise ManifestValidationError(f"mining runtime output retained omitted roots: {leaked_mining_roots}")
    if missing_full_roots:
        raise ManifestValidationError(f"full runtime output lost module-owned roots: {missing_full_roots}")
    for name, output in runtime_outputs.items():
        missing_shared = sorted(set(SHARED_ROOTS_TO_PROVE) - set(output["shared_roots_present"]))
        if missing_shared:
            raise ManifestValidationError(f"{name} runtime output lost shared/runtime roots: {missing_shared}")

    modify_present_roots = set(full_to_mining_modify["present_roots"])
    modify_removed_roots = set(full_to_mining_modify["removed_roots"])
    missing_modify_mining_roots = sorted(MODULE_ROOTS_TO_PROVE["mining"] - modify_present_roots)
    leaked_modify_full_roots = sorted(mining_omitted_roots.intersection(modify_present_roots))
    missing_modify_removed_roots = sorted(mining_omitted_roots - modify_removed_roots)
    missing_modify_shared = sorted(set(SHARED_ROOTS_TO_PROVE) - set(full_to_mining_modify["shared_roots_present"]))
    if missing_modify_mining_roots:
        raise ManifestValidationError(f"full-to-mining modify lost selected roots: {missing_modify_mining_roots}")
    if leaked_modify_full_roots:
        raise ManifestValidationError(f"full-to-mining modify retained deselected roots: {leaked_modify_full_roots}")
    if missing_modify_removed_roots:
        raise ManifestValidationError(f"full-to-mining modify did not remove deselected roots: {missing_modify_removed_roots}")
    if missing_modify_shared:
        raise ManifestValidationError(f"full-to-mining modify lost shared/runtime roots: {missing_modify_shared}")
    if set(full_to_mining_modify["launcher_entry_ids"]) != {"mining-loadout", "mining-signals"}:
        raise ManifestValidationError(
            "full-to-mining modify retained stale launcher visibility: "
            f"{full_to_mining_modify['launcher_entry_ids']}"
        )

    module_selected_roots = {"tools/Mining_Signals", "tools/Battle_Buddy"}
    module_omitted_roots = _manifest_module_owned_paths(manifest) - module_selected_roots
    module_present_roots = set(module_runtime_apply["present_roots"])
    module_removed_roots = set(module_runtime_apply["removed_roots"])
    missing_module_selected_roots = sorted(module_selected_roots - module_present_roots)
    leaked_module_omitted_roots = sorted(module_omitted_roots.intersection(module_present_roots))
    missing_module_removed_roots = sorted(module_omitted_roots - module_removed_roots)
    missing_module_shared = sorted(set(SHARED_ROOTS_TO_PROVE) - set(module_runtime_apply["shared_roots_present"]))
    if missing_module_selected_roots:
        raise ManifestValidationError(f"selected-module runtime output lost selected roots: {missing_module_selected_roots}")
    if leaked_module_omitted_roots:
        raise ManifestValidationError(f"selected-module runtime output retained omitted roots: {leaked_module_omitted_roots}")
    if missing_module_removed_roots:
        raise ManifestValidationError(f"selected-module runtime output did not remove omitted roots: {missing_module_removed_roots}")
    if missing_module_shared:
        raise ManifestValidationError(f"selected-module runtime output lost shared/runtime roots: {missing_module_shared}")
    if set(module_runtime_apply["launcher_entry_ids"]) != {"battle-buddy", "mining-signals"}:
        raise ManifestValidationError(
            "selected-module runtime output exposed wrong launcher visibility: "
            f"{module_runtime_apply['launcher_entry_ids']}"
        )
    if set(module_runtime_apply["shortcut_ids"]) != {"battle-buddy", "mining-signals"}:
        raise ManifestValidationError(
            "selected-module runtime output exposed wrong shortcut ids: "
            f"{module_runtime_apply['shortcut_ids']}"
        )
    if set(module_runtime_apply["ownership_ledger_modules"]) != {"battle_buddy", "mining_signals"}:
        raise ManifestValidationError(
            "selected-module runtime output wrote unexpected ownership ledger modules: "
            f"{module_runtime_apply['ownership_ledger_modules']}"
        )

    sequence_fresh = module_add_remove_sequence["fresh_custom_apply"]
    sequence_add = module_add_remove_sequence["add_combat_modify"]
    sequence_remove = module_add_remove_sequence["remove_combat_modify"]
    mining_combat_roots = MODULE_ROOTS_TO_PROVE["mining"].union(MODULE_ROOTS_TO_PROVE["combat"])
    if sequence_fresh["action"] != "apply-module-selection-install-root":
        raise ManifestValidationError("selected-module sequence fresh step used wrong action")
    if set(sequence_fresh["included_modules"]) != {"mining_loadout", "mining_signals"}:
        raise ManifestValidationError(
            f"selected-module sequence fresh step wrote wrong modules: {sequence_fresh['included_modules']}"
        )
    if not MODULE_ROOTS_TO_PROVE["mining"].issubset(set(sequence_fresh["present_roots"])):
        raise ManifestValidationError("selected-module sequence fresh step lost Mining roots")
    if MODULE_ROOTS_TO_PROVE["combat"].intersection(set(sequence_fresh["present_roots"])):
        raise ManifestValidationError("selected-module sequence fresh step retained Combat roots")
    if set(sequence_add["added_modules"]) != {"dps_calculator", "battle_buddy"}:
        raise ManifestValidationError(f"selected-module sequence add step reported wrong additions: {sequence_add['added_modules']}")
    if set(sequence_add["present_roots"]) & mining_combat_roots != mining_combat_roots:
        raise ManifestValidationError("selected-module sequence add step did not expose Mining and Combat roots")
    if set(sequence_add["launcher_entry_ids"]) != {
        "dps-calculator",
        "mining-loadout",
        "battle-buddy",
        "mining-signals",
    }:
        raise ManifestValidationError(f"selected-module sequence add step exposed wrong launchers: {sequence_add['launcher_entry_ids']}")
    if set(sequence_remove["removed_modules"]) != {"dps_calculator", "battle_buddy"}:
        raise ManifestValidationError(
            f"selected-module sequence remove step reported wrong removals: {sequence_remove['removed_modules']}"
        )
    if MODULE_ROOTS_TO_PROVE["combat"].intersection(set(sequence_remove["present_roots"])):
        raise ManifestValidationError("selected-module sequence remove step retained Combat roots")
    if not MODULE_ROOTS_TO_PROVE["combat"].issubset(set(sequence_remove["removed_roots"])):
        raise ManifestValidationError("selected-module sequence remove step did not remove Combat roots")
    if not MODULE_ROOTS_TO_PROVE["mining"].issubset(set(sequence_remove["present_roots"])):
        raise ManifestValidationError("selected-module sequence remove step lost Mining roots")
    for step_name, output in [
        ("fresh", sequence_fresh),
        ("add", sequence_add),
        ("remove", sequence_remove),
    ]:
        missing_shared = sorted(set(SHARED_ROOTS_TO_PROVE) - set(output["shared_roots_present"]))
        if missing_shared:
            raise ManifestValidationError(f"selected-module sequence {step_name} step lost shared/runtime roots: {missing_shared}")
    if invalid_module_modify_failure.get("returncode") != 2:
        raise ManifestValidationError("selected-module invalid modify proof did not fail with helper error")

    release_evidence = _release_evidence_payload(
        selected_profile=profile,
        release_artifact=release_artifact,
        dry_runs=dry_runs,
        modify_runs=modify_runs,
        visible_ids=visible_ids,
        expected_profile_ids=expected_profile_ids,
        full_ids=full_ids,
        hidden_for_mining=hidden_for_mining,
        runtime_outputs=runtime_outputs,
        full_to_mining_modify=full_to_mining_modify,
        broad_payload_roots=_manifest_module_owned_paths(manifest),
        module_selection_dry_run=module_selection_dry_run,
        module_preselection=module_preselection,
        module_runtime_apply=module_runtime_apply,
        module_add_remove_sequence=module_add_remove_sequence,
        invalid_module_modify_failure=invalid_module_modify_failure,
    )

    return {
        "ok": True,
        "profile": profile,
        "dry_run_profiles": sorted(dry_runs),
        "modify_runs": sorted(modify_runs),
        "install_state_profile": profile,
        "launcher_entry_ids": sorted(visible_ids),
        "mining_hidden_launcher_entry_ids": hidden_for_mining if profile == "mining" else [],
        "full_launcher_entry_ids": sorted(full_ids),
        "entrypoint_plan_profile": profile,
        "inno_module_checkbox_handoff": inno_handoff,
        "release_artifact": release_artifact,
        "release_evidence": release_evidence,
        "runtime_outputs": runtime_outputs,
        "full_to_mining_modify": full_to_mining_modify,
        "module_selection_dry_run": module_selection_dry_run,
        "module_preselection": module_preselection,
        "module_runtime_apply": module_runtime_apply,
        "module_add_remove_sequence": module_add_remove_sequence,
        "invalid_module_modify_failure": invalid_module_modify_failure,
        "mining_removed_roots": sorted(mining_output["removed_roots"]),
        "mining_present_roots": sorted(mining_output["present_roots"]),
        "full_present_roots": sorted(full_output["present_roots"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a release smoke proof for installer profiles without compiling Inno Setup."
    )
    parser.add_argument(
        "profile",
        nargs="?",
        default="mining",
        help="Profile whose emitted install_state.json should be validated through launcher visibility.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        help="Run smoke validation for one or more profiles and emit a combined result.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.profiles:
            result = {
                "ok": True,
                "profiles": {profile: run_smoke(profile) for profile in args.profiles},
            }
        else:
            result = run_smoke(args.profile)
    except ManifestValidationError as exc:
        print(f"ManifestValidationError: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
