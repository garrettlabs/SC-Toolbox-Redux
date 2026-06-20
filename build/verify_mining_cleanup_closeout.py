"""Final M007/S04 Mining cleanup closeout verifier.

This command ties together the S02 generated-artifact cleanup proof, the S03
non-Mining source-prune proof, the restored installer smoke proof, and the R006
closeout story.  It is intentionally read/check only: it reads manifests and
runs deterministic verifier/smoke commands, but never invokes cleanup or delete
helpers directly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
GENERATED_MANIFEST_PATH = ROOT / "build" / "generated_artifact_cleanup_manifest.json"
SOURCE_MANIFEST_PATH = ROOT / "build" / "non_mining_source_prune_manifest.json"
SOURCE_AUDIT_PATH = ROOT / "build" / "non_mining_source_prune_audit.json"

PROTECTED_PATHS = (
    "tools/Mining_Signals",
    "skills/Mining_Loadout",
    "redux_mining_launcher.py",
    "core",
    "core/launcher_visibility.py",
    "core/skill_registry.py",
    "shared",
    "shared/config_models.py",
    "ui",
    "build/redux_mining_build.py",
    "build/installer_profiles.json",
    "build/installer_profile_dry_run.py",
    "build/installer_release_smoke.py",
    "build/verify_generated_artifact_cleanup.py",
    "build/verify_non_mining_source_prune.py",
    "build/generated_artifact_cleanup_manifest.json",
    "build/non_mining_source_prune_manifest.json",
    "build/non_mining_source_prune_audit.json",
)

S02_VERIFIER_COMMAND = (sys.executable, "-B", "build/verify_generated_artifact_cleanup.py")
S03_VERIFIER_COMMAND = (sys.executable, "-B", "build/verify_non_mining_source_prune.py")
def installer_smoke_profiles(manifest_path: Path = ROOT / "build" / "installer_profiles.json") -> tuple[str, ...]:
    data = load_json_file(manifest_path, label="installer profiles manifest")
    profiles = data.get("profiles")
    if not isinstance(profiles, Mapping):
        raise VerificationError("installer profiles manifest missing profiles object")
    ordered = tuple(profile for profile in ("mining", "full") if profile in profiles)
    extras = tuple(sorted(str(profile) for profile in profiles if profile not in {"mining", "full"}))
    selected = ordered + extras
    if not selected:
        raise VerificationError("installer profiles manifest contains no profiles to smoke")
    return selected


def installer_smoke_command(profiles: Sequence[str]) -> tuple[str, ...]:
    return (
        sys.executable,
        "-B",
        "build/installer_release_smoke.py",
        "--profiles",
        *profiles,
    )


class VerificationError(RuntimeError):
    """Raised when closeout proof detects an unsafe or regressed state."""


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


def normalize_rel_path(value: str, *, root: Path = ROOT) -> str:
    normalized = value.replace("\\", "/").strip("/")
    if not normalized:
        raise VerificationError("manifest entry path must not be empty")
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise VerificationError(f"path escapes repository root: {value!r}") from exc
    return normalized


def manifest_entries(data: Mapping[str, Any], *, root: Path = ROOT, label: str = "manifest") -> list[dict[str, Any]]:
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
        copied["path"] = normalize_rel_path(path, root=root)
        normalized.append(copied)
    declared = data.get("entry_count")
    if declared not in {None, len(normalized)}:
        raise VerificationError(f"{label} entry_count mismatch: declared={declared} actual={len(normalized)}")
    return normalized


def generated_manifest_count(data: Mapping[str, Any], *, root: Path = ROOT) -> int:
    return len(manifest_entries(data, root=root, label="generated cleanup manifest"))


def assert_no_source_delete_entries(entries: Sequence[Mapping[str, Any]]) -> int:
    unsafe: list[str] = []
    for entry in entries:
        disposition = str(entry.get("disposition", "")).lower()
        deletion_allowed = bool(entry.get("deletion_allowed"))
        if disposition == "delete" or deletion_allowed:
            unsafe.append(str(entry.get("path")))
    if unsafe:
        raise VerificationError("source manifest authorizes deletion: " + ", ".join(unsafe[:20]))
    return len(entries)


def assert_source_audit_zero(data: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in ("attempted", "deleted", "rejected"):
        value = data.get(key)
        if not isinstance(value, list):
            raise VerificationError(f"source audit {key!r} must be a list")
        counts[key] = len(value)
        if value:
            raise VerificationError(f"source audit {key} paths unexpectedly non-zero: {value[:5]!r}")
    skipped = data.get("skipped", [])
    if not isinstance(skipped, list):
        raise VerificationError("source audit 'skipped' must be a list")
    counts["skipped"] = len(skipped)

    summary = data.get("summary")
    if isinstance(summary, Mapping):
        summary_counts = {
            "attempted": summary.get("attempted", summary.get("attempted_count", 0)) or 0,
            "deleted": summary.get("deleted", summary.get("deleted_count", 0)) or 0,
            "rejected": summary.get("rejected", summary.get("rejected_count", 0)) or 0,
        }
        nonzero = {key: value for key, value in summary_counts.items() if value}
        if nonzero:
            raise VerificationError(f"source audit summary reports unsafe counts: {nonzero!r}")
    return counts


def assert_protected_paths_present(*, root: Path = ROOT, paths: Sequence[str] = PROTECTED_PATHS) -> int:
    missing = [path for path in paths if not (root / path).exists()]
    if missing:
        raise VerificationError("protected paths missing: " + ", ".join(missing))
    return len(paths)


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


def require_success(result: CommandResult, *, label: str) -> str:
    if result.returncode != 0:
        raise VerificationError(f"{label} failed exit={result.returncode} command={result.command}\n{result.combined[-3000:]}")
    return f"exit=0 command={result.command}"


def classify_installer_smoke(result: CommandResult, *, expected_profiles: Sequence[str] | None = None) -> str:
    require_success(result, label="installer smoke")
    expected = tuple(expected_profiles or installer_smoke_profiles())
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"installer smoke emitted malformed JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise VerificationError("installer smoke JSON did not report ok=true")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise VerificationError("installer smoke JSON missing profiles object")
    missing = [profile for profile in expected if profile not in profiles]
    if missing:
        raise VerificationError("installer smoke JSON missing profiles: " + ", ".join(missing))
    unexpected = [profile for profile in profiles if profile not in expected]
    if unexpected:
        raise VerificationError("installer smoke JSON reported unexpected profiles: " + ", ".join(unexpected))
    for profile in expected:
        profile_payload = profiles[profile]
        if not isinstance(profile_payload, dict) or profile_payload.get("ok") is not True:
            raise VerificationError(f"installer smoke profile {profile!r} did not report ok=true")
    return "exit=0 profiles=" + ",".join(expected)


def run_required_command(
    args: Sequence[str],
    *,
    label: str,
    timeout: int,
    runner: Runner = run,
    classifier: Callable[[CommandResult], str] | None = None,
) -> str:
    try:
        result = runner(args, timeout)
    except subprocess.TimeoutExpired as exc:
        raise VerificationError(f"{label} timed out: {exc.cmd}") from exc
    if classifier is not None:
        return classifier(result)
    return require_success(result, label=label)


def run_verification(*, runner: Runner = run) -> None:
    generated_manifest = load_json_file(GENERATED_MANIFEST_PATH, label="generated cleanup manifest")
    generated_count = generated_manifest_count(generated_manifest)
    pass_line("generated manifest", f"{generated_count} entries loaded")

    source_manifest = load_json_file(SOURCE_MANIFEST_PATH, label="source prune manifest")
    source_entries = manifest_entries(source_manifest, label="source prune manifest")
    source_count = assert_no_source_delete_entries(source_entries)

    source_audit = load_json_file(SOURCE_AUDIT_PATH, label="source prune audit")
    audit_counts = assert_source_audit_zero(source_audit)
    pass_line(
        "source manifest/audit",
        (
            f"manifest_entries={source_count} delete_entries=0 "
            f"attempted={audit_counts['attempted']} deleted={audit_counts['deleted']} "
            f"rejected={audit_counts['rejected']} skipped={audit_counts['skipped']}"
        ),
    )

    protected_count = assert_protected_paths_present()
    pass_line("protected boundaries", f"{protected_count} Mining/shared/runtime/Redux/installer paths present")

    s02_status = run_required_command(
        S02_VERIFIER_COMMAND,
        label="S02 generated cleanup verifier",
        timeout=360,
        runner=runner,
    )
    pass_line("S02 verifier status", s02_status)

    s03_status = run_required_command(
        S03_VERIFIER_COMMAND,
        label="S03 source prune verifier",
        timeout=420,
        runner=runner,
    )
    pass_line("S03 verifier status", s03_status)

    smoke_profiles = installer_smoke_profiles()
    installer_status = run_required_command(
        installer_smoke_command(smoke_profiles),
        label="installer smoke",
        timeout=240,
        runner=runner,
        classifier=lambda result: classify_installer_smoke(result, expected_profiles=smoke_profiles),
    )
    pass_line("installer smoke status", installer_status)

    pass_line(
        "R006 proof summary",
        (
            "generated cleanup absent via S02, source prune delete/audit counts zero via S03, "
            "protected Mining/shared/Redux/installer boundaries present, installer smoke passes active manifest profiles"
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise VerificationError("verify_mining_cleanup_closeout.py does not accept arguments")
    try:
        run_verification()
    except VerificationError as exc:
        fail_line("Mining cleanup closeout", str(exc))
        return 1
    except subprocess.TimeoutExpired as exc:
        fail_line("Mining cleanup closeout", f"command timed out: {exc.cmd}")
        return 1
    print("PASS Mining cleanup closeout: final M007/S04 cleanup evidence is healthy for R006")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
