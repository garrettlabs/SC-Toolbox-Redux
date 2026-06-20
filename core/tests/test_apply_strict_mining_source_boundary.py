from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build import apply_strict_mining_source_boundary as apply_boundary
from build import strict_mining_source_boundary_manifest as strict_manifest


class ApplyStrictMiningSourceBoundaryTests(unittest.TestCase):
    def test_verify_audit_rejects_unapproved_archive_path(self) -> None:
        manifest = strict_manifest.build_manifest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_path = Path(tmp_dir) / "audit.json"
            manifest_path = Path(tmp_dir) / "manifest.json"
            strict_manifest.write_manifest(manifest_path, manifest)
            approved_paths = [
                entry["path"]
                for entry in manifest["entries"]
                if entry.get("boundary_type") == "archived_app_source_root"
            ]
            metadata_approvals = apply_boundary.metadata_pruning_approvals(manifest)
            audit_path.write_text(
                json.dumps(
                    {
                        "approved_manifest_paths": [*approved_paths, *[item["path"] for item in metadata_approvals]],
                        "metadata_pruning_approvals": metadata_approvals,
                        "rejected": [],
                        "operations": [
                            {"path": "shared/qt", "action": "archive", "status": "archived"}
                        ],
                        "protected_runtime_snapshots": [{"path": "shared/qt", "exists": True, "kind": "directory"}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(apply_boundary.ApplyBoundaryError, "unapproved path"):
                apply_boundary.verify_audit(audit_path=audit_path, manifest_path=manifest_path)

    def test_archive_app_source_root_rejects_protected_app_root(self) -> None:
        with self.assertRaisesRegex(apply_boundary.ApplyBoundaryError, "protected-boundary mismatch"):
            apply_boundary.archive_app_source_root(
                {"path": "tools/Mining_Signals", "rationale": "test"},
                dry_run=True,
            )

    def test_archive_existing_destination_requires_matching_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "skills" / "Cargo_loader"
            archive = root / "archived_source" / "M007_S05_strict_mining_boundary" / "skills" / "Cargo_loader"
            source.mkdir(parents=True)
            archive.mkdir(parents=True)
            (source / "cargo_app.py").write_text("print('source')\n", encoding="utf-8")
            (archive / "cargo_app.py").write_text("print('different')\n", encoding="utf-8")

            with mock.patch.object(apply_boundary, "ROOT", root), mock.patch.object(apply_boundary, "ARCHIVE_ROOT", root / "archived_source" / "M007_S05_strict_mining_boundary"):
                with self.assertRaisesRegex(apply_boundary.ApplyBoundaryError, "hash/type mismatch"):
                    apply_boundary.archive_app_source_root(
                        {"path": "skills/Cargo_loader", "rationale": "test"},
                        dry_run=False,
                    )

    def test_prune_installer_profiles_keeps_mining_and_shared_runtime_only(self) -> None:
        original = {
            "profiles": {
                "mining": {"modules": ["mining_signals", "mining_loadout"]},
                "trading": {"modules": ["cargo_loader"]},
                "full": {"modules": ["mining_signals", "mining_loadout", "cargo_loader"]},
            },
            "modules": {
                "mining_loadout": {"owned_paths": ["skills/Mining_Loadout"]},
                "mining_signals": {"owned_paths": ["tools/Mining_Signals"]},
                "cargo_loader": {"owned_paths": ["skills/Cargo_loader"]},
            },
            "shared_dependencies": {"qt_runtime": {"owned_paths": ["shared/qt"]}},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            profiles_path = Path(tmp_dir) / "installer_profiles.json"
            profiles_path.parent.mkdir(parents=True, exist_ok=True)
            profiles_path.write_text(json.dumps(original), encoding="utf-8")

            with mock.patch.object(apply_boundary, "INSTALLER_PROFILES_PATH", profiles_path), mock.patch.object(apply_boundary, "ROOT", Path(tmp_dir)):
                operation = apply_boundary.prune_installer_profiles(dry_run=False)
                pruned = json.loads(profiles_path.read_text(encoding="utf-8"))

        self.assertEqual(operation["status"], "pruned")
        self.assertEqual(set(pruned["profiles"]), {"mining"})
        self.assertEqual(set(pruned["modules"]), {"mining_loadout", "mining_signals"})
        self.assertEqual(set(pruned["shared_dependencies"]), {"qt_runtime"})

    def test_verify_audit_accepts_already_removed_app_source_roots(self) -> None:
        manifest = strict_manifest.build_manifest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_path = Path(tmp_dir) / "audit.json"
            manifest_path = Path(tmp_dir) / "manifest.json"
            strict_manifest.write_manifest(manifest_path, manifest)
            removed_paths = [
                entry["path"]
                for entry in manifest["entries"]
                if entry.get("boundary_type") == "removed_app_source_root"
            ]
            metadata_approvals = apply_boundary.metadata_pruning_approvals(manifest)
            audit_path.write_text(
                json.dumps(
                    {
                        "approved_manifest_paths": [*removed_paths, *[item["path"] for item in metadata_approvals]],
                        "metadata_pruning_approvals": metadata_approvals,
                        "rejected": [],
                        "operations": [],
                        "protected_runtime_snapshots": [{"path": "shared/qt", "exists": True, "kind": "directory"}],
                    }
                ),
                encoding="utf-8",
            )

            result = apply_boundary.verify_audit(audit_path=audit_path, manifest_path=manifest_path)

        self.assertEqual(result["operations"], [])
        self.assertIn("skills/Cargo_loader", result["approved_manifest_paths"])

    def test_verify_audit_requires_metadata_pruning_approval_details(self) -> None:
        manifest = strict_manifest.build_manifest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_path = Path(tmp_dir) / "audit.json"
            manifest_path = Path(tmp_dir) / "manifest.json"
            strict_manifest.write_manifest(manifest_path, manifest)
            archive_paths = [
                entry["path"]
                for entry in manifest["entries"]
                if entry.get("boundary_type") == "archived_app_source_root"
            ]
            metadata_paths = [
                entry["path"]
                for entry in manifest["entries"]
                if entry.get("proposed_action") == "pruned"
            ]
            operations = [
                {"path": path, "action": "archive", "status": "skipped"}
                for path in archive_paths
            ]
            audit_path.write_text(
                json.dumps(
                    {
                        "approved_manifest_paths": [*archive_paths, *metadata_paths],
                        "metadata_pruning_approvals": [],
                        "rejected": [],
                        "operations": operations,
                        "protected_runtime_snapshots": [{"path": "shared/qt", "exists": True, "kind": "directory"}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(apply_boundary.ApplyBoundaryError, "metadata pruning approval records"):
                apply_boundary.verify_audit(audit_path=audit_path, manifest_path=manifest_path)


if __name__ == "__main__":
    unittest.main()
