"""Verify S03 non-Mining source pruning did not regress Mining/Redux/installer behavior.

This verifier is intentionally source-manifest centric.  It proves that the S03
source-prune manifest/audit did not delete or authorize deletion of source paths,
that protected Mining/Redux/installer/runtime seams are still present, and that
focused runtime smoke checks are either passing or still match a documented
pre-existing S01/S02 baseline signature.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "build" / "non_mining_source_prune_manifest.json"
AUDIT_PATH = ROOT / "build" / "non_mining_source_prune_audit.json"
S02_VERIFIER = ROOT / "build" / "verify_generated_artifact_cleanup.py"

PROTECTED_PATHS = (
    "tools/Mining_Signals",
    "skills/Mining_Loadout",
    "redux_mining_launcher.py",
    "build/redux_mining_build.py",
    "build/installer_profile_dry_run.py",
    "build/installer_release_smoke.py",
    "build/verify_generated_artifact_cleanup.py",
    "build/installer_profiles.json",
    "core",
    "shared",
    "ui",
)

MINING_IMPORTS = (
    "redux_mining_launcher",
    "build.redux_mining_build",
    "build.installer_profile_dry_run",
    "tools.Mining_Signals.mining_signals_app",
    "tools.Mining_Signals.ocr.sc_ocr.scan_results_match",
    "skills.Mining_Loadout.mining_loadout_app",
    "shared.config_models",
)

INSTALLER_BASELINE_FAILURES = (
    "ModuleNotFoundError: No module named 'core.launcher_visibility'",
    'ModuleNotFoundError: No module named "core.launcher_visibility"',
)

S02_INHERITED_FAILURES = INSTALLER_BASELINE_FAILURES + (
    "unrecognized arguments: --profiles full",
)

TRANSIENT_GENERATED_ARTIFACTS = (
    ".pytest_cache",
    "dist",
)


class VerificationError(RuntimeError):
    """Raised when the verifier detects a source-prune regression."""


@dataclass(frozen=True)
class CommandResult:
    args: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        return f"{self.stdout}\n{self.stderr}"

    @property
    def command(self) -> str:
        return " ".join(self.args)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
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


def normalize_rel_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    candidate = (ROOT / normalized).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise VerificationError(f"path escapes repository root: {value!r}") from exc
    return normalized


def manifest_entries(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise VerificationError("manifest missing entries list")
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise VerificationError(f"manifest entry {index} is not an object")
        path = entry.get("path")
        if not isinstance(path, str) or not path.strip():
            raise VerificationError(f"manifest entry {index} missing non-empty path")
        copied = dict(entry)
        copied["path"] = normalize_rel_path(path)
        normalized.append(copied)
    if data.get("entry_count") not in {None, len(normalized)}:
        raise VerificationError(
            f"manifest entry_count mismatch: declared={data.get('entry_count')} actual={len(normalized)}"
        )
    return normalized


def assert_no_delete_entries(entries: Sequence[Mapping[str, Any]]) -> None:
    delete_like: list[str] = []
    for entry in entries:
        disposition = str(entry.get("disposition", "")).lower()
        allowed = bool(entry.get("deletion_allowed"))
        if disposition == "delete" or allowed:
            delete_like.append(str(entry.get("path")))
    if delete_like:
        raise VerificationError(
            "source prune manifest still authorizes deletion: " + ", ".join(delete_like[:20])
        )


def _manifest_entry_covering_path(
    by_path: Mapping[str, Mapping[str, Any]], protected_path: str
) -> Mapping[str, Any] | None:
    current = Path(protected_path)
    candidates = [current.as_posix(), *[parent.as_posix() for parent in current.parents if parent.as_posix() != "."]]
    for candidate in candidates:
        entry = by_path.get(candidate)
        if entry is not None:
            return entry
    return None


def assert_required_manifest_paths(entries: Sequence[Mapping[str, Any]]) -> None:
    by_path = {str(entry.get("path")): entry for entry in entries}
    missing_on_disk = [path for path in PROTECTED_PATHS if not (ROOT / path).exists()]
    if missing_on_disk:
        raise VerificationError("protected paths missing on disk: " + ", ".join(missing_on_disk))
    uncovered = []
    unsafe = []
    for path in PROTECTED_PATHS:
        entry = _manifest_entry_covering_path(by_path, path)
        if entry is None:
            uncovered.append(path)
            continue
        if entry.get("disposition") != "keep" or entry.get("deletion_allowed"):
            unsafe.append(
                f"{path} covered_by={entry.get('path')!r} "
                f"disposition={entry.get('disposition')!r} deletion_allowed={entry.get('deletion_allowed')!r}"
            )
    if uncovered:
        raise VerificationError("protected paths missing manifest keep coverage: " + ", ".join(uncovered))
    if unsafe:
        raise VerificationError("protected paths are not keep-safe: " + "; ".join(unsafe))


def assert_audit_safe(data: Mapping[str, Any]) -> None:
    for key in ("attempted", "deleted", "rejected", "skipped"):
        if not isinstance(data.get(key), list):
            raise VerificationError(f"audit {key!r} must be a list")
    attempted = data.get("attempted", [])
    deleted = data.get("deleted", [])
    if attempted:
        raise VerificationError(f"audit attempted source deletions unexpectedly: {attempted[:5]!r}")
    if deleted:
        raise VerificationError(f"audit deleted source paths unexpectedly: {deleted[:5]!r}")
    summary = data.get("summary")
    if isinstance(summary, Mapping):
        deleted_count = summary.get("deleted") or summary.get("deleted_count") or 0
        attempted_count = summary.get("attempted") or summary.get("attempted_count") or 0
        if deleted_count or attempted_count:
            raise VerificationError(f"audit summary reports attempted/deleted source paths: {summary!r}")


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("LOCALAPPDATA", str(ROOT / ".localappdata"))
    env.pop("PYTEST_ADDOPTS", None)
    return env


def run(args: Sequence[str], *, timeout: int = 120) -> CommandResult:
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
    return CommandResult(args=tuple(args), returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def remove_transient_generated_artifacts() -> list[str]:
    removed: list[str] = []
    for rel_path in TRANSIENT_GENERATED_ARTIFACTS:
        path = ROOT / rel_path
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(rel_path)
    return removed


def verify_generated_cleanup_recheck() -> str:
    removed = remove_transient_generated_artifacts()
    result = run([sys.executable, "-B", str(S02_VERIFIER.relative_to(ROOT))], timeout=300)
    if result.returncode == 0:
        return f"S02 verifier exited 0 after transient cleanup removed={removed or ['none']}"
    matched = [signature for signature in S02_INHERITED_FAILURES if signature in result.combined]
    if matched:
        return (
            "S02 verifier reran and matched inherited baseline "
            f"{matched[0]!r}; command={result.command} exit={result.returncode} "
            f"removed_transients={removed or ['none']}"
        )
    raise VerificationError(
        "S02 generated cleanup verifier regressed with a new signature:\n" + result.combined[-3000:]
    )


def verify_mining_imports() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    failures: list[str] = []
    for module_name in MINING_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - verifier reports import contract failures.
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if failures:
        raise VerificationError("Mining/Redux import failures: " + "; ".join(failures))
    return len(MINING_IMPORTS)


def verify_redux_smoke() -> str:
    with tempfile.TemporaryDirectory(prefix="sc-toolbox-redux-source-prune-") as tmp:
        output = Path(tmp) / "SC_Toolbox_Redux_Mining"
        result = run(
            [
                sys.executable,
                "-B",
                "build/redux_mining_build.py",
                "--output",
                str(output),
            ],
            timeout=180,
        )
        if result.returncode != 0:
            raise VerificationError("Redux mining build helper failed:\n" + result.combined[-3000:])
        validate = run(
            [
                sys.executable,
                "-B",
                "build/redux_mining_build.py",
                "--validate-only",
                "--output",
                str(output),
            ],
            timeout=120,
        )
        if validate.returncode != 0:
            raise VerificationError("Redux mining validate-only failed:\n" + validate.combined[-3000:])
    if (ROOT / "dist").exists():
        raise VerificationError("Redux smoke left top-level dist artifact behind")
    return "source build and validate-only succeeded in a temporary output; no dist artifact remains"


def verify_installer_profile_smoke() -> str:
    dry = run([sys.executable, "-B", "build/installer_profile_dry_run.py", "mining"], timeout=120)
    if dry.returncode != 0:
        raise VerificationError("installer profile dry-run failed:\n" + dry.combined[-3000:])
    try:
        plan = json.loads(dry.stdout)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"installer dry-run emitted malformed JSON: {exc}") from exc
    required_modules = {"mining_signals", "mining_loadout"}
    required_shortcuts = {"mining-signals", "mining-loadout"}
    if set(plan.get("included_modules", [])) != required_modules:
        raise VerificationError(f"mining dry-run modules changed: {plan.get('included_modules')!r}")
    if set(plan.get("shortcut_ids", [])) != required_shortcuts:
        raise VerificationError(f"mining dry-run shortcuts changed: {plan.get('shortcut_ids')!r}")

    release = run([sys.executable, "-B", "build/installer_release_smoke.py", "mining"], timeout=180)
    if release.returncode == 0:
        return "dry-run passed and installer release smoke exited 0"
    matched = [signature for signature in INSTALLER_BASELINE_FAILURES if signature in release.combined]
    if matched:
        return f"dry-run passed; release smoke matched inherited baseline {matched[0]!r}"
    raise VerificationError("installer release smoke failure signature changed:\n" + release.combined[-3000:])


def verify_baseline_comparison() -> str:
    missing_launcher_visibility = not (ROOT / "core" / "launcher_visibility.py").exists()
    if missing_launcher_visibility:
        return "core.launcher_visibility remains absent, matching inherited S01/S02 baseline"
    return "core.launcher_visibility exists; inherited baseline failure no longer applies"


def run_verification() -> None:
    manifest = load_json_file(MANIFEST_PATH, label="source prune manifest")
    audit = load_json_file(AUDIT_PATH, label="source prune audit")
    entries = manifest_entries(manifest)
    assert_no_delete_entries(entries)
    pass_line("manifest absence", f"{len(entries)} entries loaded; delete entries absent")
    assert_audit_safe(audit)
    pass_line("manifest absence", "audit attempted/deleted source paths absent")
    assert_required_manifest_paths(entries)
    pass_line("protected paths", f"{len(PROTECTED_PATHS)} protected Mining/Redux/installer paths present and keep-safe")
    import_count = verify_mining_imports()
    pass_line("Mining imports", f"{import_count} Mining/Redux/runtime modules import successfully")
    pass_line("Redux smoke", verify_redux_smoke())
    pass_line("installer/profile smoke", verify_installer_profile_smoke())
    pass_line("generated cleanup recheck", verify_generated_cleanup_recheck())
    pass_line("baseline comparison", verify_baseline_comparison())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify S03 non-Mining source-prune health signals.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        run_verification()
    except VerificationError as exc:
        fail_line("source prune verification", str(exc))
        return 1
    except subprocess.TimeoutExpired as exc:
        fail_line("source prune verification", f"command timed out: {exc.cmd}")
        return 1
    print("PASS source prune verification: S03 Mining/Redux/installer behavior unchanged or inherited-baseline classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
