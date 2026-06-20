from __future__ import annotations

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build import strict_mining_source_boundary_manifest as strict_manifest
from build import verify_strict_mining_source_boundary as verify_strict


def _minimal_boundary_manifest(root: Path) -> dict[str, object]:
    (root / "skills" / "Mining_Loadout").mkdir(parents=True)
    (root / "tools" / "Mining_Signals").mkdir(parents=True)
    (root / "shared" / "qt").mkdir(parents=True)
    (root / "archived_source" / "M007_S05_strict_mining_boundary" / "skills" / "Cargo_loader").mkdir(parents=True)
    return {
        "entries": [
            {
                "path": "skills/Mining_Loadout",
                "classification": "mining",
                "boundary_type": "app_source_root",
            },
            {
                "path": "tools/Mining_Signals",
                "classification": "mining",
                "boundary_type": "app_source_root",
            },
            {
                "path": "shared/qt",
                "classification": "shared_runtime",
                "boundary_type": "retained_shared_runtime",
            },
            {
                "path": "skills/Cargo_loader",
                "classification": "archived_non_mining_app_source",
                "boundary_type": "archived_app_source_root",
                "archive_path": "archived_source/M007_S05_strict_mining_boundary/skills/Cargo_loader",
            },
        ]
    }


def _valid_installer_manifest() -> dict[str, object]:
    return copy.deepcopy(verify_strict.load_installer_manifest(verify_strict.INSTALLER_MANIFEST_PATH))


class VerifyStrictMiningSourceBoundaryTests(unittest.TestCase):
    def test_filesystem_boundary_accepts_archived_non_mining_and_retained_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _minimal_boundary_manifest(root)

            counts = verify_strict.assert_filesystem_boundary(manifest, root=root)

        self.assertEqual(counts["archived_non_mining_roots"], 1)
        self.assertEqual(counts["retained_roots"], 3)

    def test_filesystem_boundary_rejects_live_non_mining_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _minimal_boundary_manifest(root)
            (root / "skills" / "Cargo_loader").mkdir(parents=True)

            with self.assertRaisesRegex(verify_strict.VerificationError, "live non-Mining root remains"):
                verify_strict.assert_filesystem_boundary(manifest, root=root)

    def test_filesystem_boundary_rejects_missing_archive_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _minimal_boundary_manifest(root)
            archive_path = root / "archived_source" / "M007_S05_strict_mining_boundary" / "skills" / "Cargo_loader"
            archive_path.rmdir()

            with self.assertRaisesRegex(verify_strict.VerificationError, "archive copy missing"):
                verify_strict.assert_filesystem_boundary(manifest, root=root)

    def test_audit_matches_manifest_accepts_archived_and_pruned_manifest_paths(self) -> None:
        manifest = strict_manifest.build_manifest()
        expected = [
            entry["path"]
            for entry in manifest["entries"]
            if entry.get("classification") == "archived_non_mining_app_source"
            or entry.get("proposed_action") in {"archived", "pruned"}
        ]

        approved_count = verify_strict.assert_audit_matches_manifest(
            manifest,
            {"approved_manifest_paths": expected, "rejected": []},
        )

        self.assertEqual(approved_count, len(expected))

    def test_audit_matches_manifest_rejects_missing_pruned_metadata_path(self) -> None:
        manifest = strict_manifest.build_manifest()
        approved = [
            entry["path"]
            for entry in manifest["entries"]
            if entry.get("classification") == "archived_non_mining_app_source"
        ]

        with self.assertRaisesRegex(verify_strict.VerificationError, "omits approved manifest paths"):
            verify_strict.assert_audit_matches_manifest(
                manifest,
                {"approved_manifest_paths": approved, "rejected": []},
            )

    def test_stale_metadata_accepts_mining_only_installer_manifest(self) -> None:
        result = verify_strict.assert_stale_metadata_pruned(_valid_installer_manifest())

        self.assertEqual(result["profiles"], ["mining"])
        self.assertEqual(result["modules"], ["mining_loadout", "mining_signals"])
        self.assertEqual(result["launcher_entry_ids"], ["mining-loadout", "mining-signals"])

    def test_stale_metadata_rejects_non_mining_profile(self) -> None:
        manifest = _valid_installer_manifest()
        manifest["profiles"]["trading"] = {"modules": []}  # type: ignore[index]

        with self.assertRaisesRegex(verify_strict.VerificationError, "stale metadata profiles remain"):
            verify_strict.assert_stale_metadata_pruned(manifest)

    def test_stale_metadata_rejects_non_mining_launcher_id(self) -> None:
        manifest = _valid_installer_manifest()
        manifest["modules"]["mining_loadout"]["launcher_entry_ids"].append("cargo-loader")  # type: ignore[index]

        with self.assertRaisesRegex(verify_strict.VerificationError, "non-Mining launcher"):
            verify_strict.assert_stale_metadata_pruned(manifest)

    def test_installer_mining_smoke_resolves_only_mining_owned_roots(self) -> None:
        result = verify_strict.assert_installer_mining_smoke(_valid_installer_manifest())

        self.assertEqual(result["included_modules"], ["mining_loadout", "mining_signals"])
        self.assertEqual(result["launcher_entry_ids"], ["mining-loadout", "mining-signals"])
        self.assertEqual(result["module_owned_file_roots"], ["skills/Mining_Loadout", "tools/Mining_Signals"])

    def test_installer_mining_smoke_rejects_wrong_profile_modules(self) -> None:
        manifest = _valid_installer_manifest()
        manifest["profiles"]["mining"]["modules"] = ["mining_signals"]  # type: ignore[index]

        with self.assertRaisesRegex(verify_strict.VerificationError, "included wrong modules"):
            verify_strict.assert_installer_mining_smoke(manifest)

    def test_launcher_visibility_rejects_unknown_or_stale_launcher_entry(self) -> None:
        with mock.patch.object(verify_strict, "MINING_LAUNCHER_ENTRY_IDS", {"mining-signals", "cargo-loader"}):
            with self.assertRaisesRegex(ValueError, "unknown launcher_entry_ids"):
                verify_strict.assert_mining_imports_and_launcher_visibility()

    def test_prior_cleanup_verifier_composition_accepts_zero_exit_subprocesses(self) -> None:
        calls: list[tuple[tuple[str, ...], int]] = []

        def runner(args, timeout):
            calls.append((tuple(args), timeout))
            return verify_strict.CommandResult(args=args, returncode=0, stdout="ok", stderr="")

        passed = verify_strict.assert_prior_cleanup_verifier_composition(runner=runner)

        self.assertEqual(len(passed), len(verify_strict.PRIOR_CLEANUP_COMMANDS))
        self.assertEqual(len(calls), len(verify_strict.PRIOR_CLEANUP_COMMANDS))
        self.assertTrue(all(timeout == 420 for _args, timeout in calls))

    def test_prior_cleanup_verifier_composition_rejects_subprocess_failure(self) -> None:
        def runner(args, timeout):
            return verify_strict.CommandResult(args=args, returncode=7, stdout="", stderr="boom")

        with self.assertRaisesRegex(verify_strict.VerificationError, "prior cleanup verifier failed exit=7"):
            verify_strict.assert_prior_cleanup_verifier_composition(runner=runner)

    def test_prior_cleanup_verifier_composition_bubbles_timeout_as_failure_mode(self) -> None:
        def runner(args, timeout):
            raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

        with self.assertRaisesRegex(verify_strict.VerificationError, "timed out"):
            verify_strict.assert_prior_cleanup_verifier_composition(runner=runner)


if __name__ == "__main__":
    unittest.main()
