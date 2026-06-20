from __future__ import annotations

import copy
import unittest

from build import strict_mining_source_boundary_manifest as strict_manifest


def _valid_manifest() -> dict[str, object]:
    return strict_manifest.build_manifest()


class StrictMiningSourceBoundaryManifestTests(unittest.TestCase):
    def test_generated_manifest_passes_strict_verification(self) -> None:
        manifest = _valid_manifest()

        strict_manifest.verify_manifest(manifest)

        by_path = {entry["path"]: entry for entry in manifest["entries"]}  # type: ignore[index]
        self.assertEqual(by_path["skills/Cargo_loader"]["classification"], "removed_non_mining_app_source")
        self.assertFalse(by_path["skills/Cargo_loader"]["exists"])
        self.assertFalse(by_path["skills/Cargo_loader"]["exists"])
        self.assertEqual(by_path["core/skill_registry.py#non_mining_launcher_defaults"]["proposed_action"], "pruned")
        self.assertIn("shared/qt", by_path)
        self.assertEqual(manifest["summary"]["live_non_mining_app_source_roots"], 0)  # type: ignore[index]

    def test_manifest_entries_reject_path_escape(self) -> None:
        with self.assertRaisesRegex(strict_manifest.BoundaryError, "escapes repository root"):
            strict_manifest.manifest_entries({"entries": [{"path": "../outside"}]})

    def test_manifest_entries_reject_entry_count_mismatch(self) -> None:
        with self.assertRaisesRegex(strict_manifest.BoundaryError, "entry_count mismatch"):
            strict_manifest.manifest_entries({"summary": {"entry_count": 2}, "entries": [{"path": "shared"}]})

    def test_verify_rejects_missing_archived_non_mining_root_classification(self) -> None:
        manifest = _valid_manifest()
        manifest["entries"] = [  # type: ignore[index]
            entry for entry in manifest["entries"] if entry["path"] != "skills/Cargo_loader"  # type: ignore[index]
        ]
        manifest["summary"]["entry_count"] = len(manifest["entries"])  # type: ignore[index]

        with self.assertRaisesRegex(strict_manifest.BoundaryError, "missing removed non-Mining root classification"):
            strict_manifest.verify_manifest(manifest)

    def test_verify_rejects_non_archived_non_mining_action(self) -> None:
        manifest = _valid_manifest()
        for entry in manifest["entries"]:  # type: ignore[index]
            if entry["path"] == "skills/Cargo_loader":
                entry["proposed_action"] = "archive"
                break

        with self.assertRaisesRegex(strict_manifest.BoundaryError, "live non-Mining root still requires cleanup"):
            strict_manifest.verify_manifest(manifest)

    def test_verify_rejects_reintroduced_removed_source_root(self) -> None:
        manifest = _valid_manifest()
        for entry in manifest["entries"]:  # type: ignore[index]
            if entry["path"] == "skills/Cargo_loader":
                entry["exists"] = True
                break

        with self.assertRaisesRegex(strict_manifest.BoundaryError, "non-Mining source root still exists"):
            strict_manifest.verify_manifest(manifest)

    def test_verify_rejects_stale_profile_metadata(self) -> None:
        manifest = _valid_manifest()
        for entry in manifest["entries"]:  # type: ignore[index]
            if entry["path"] == "build/installer_profiles.json#profiles.trading":
                entry["exists"] = True
                break

        with self.assertRaisesRegex(strict_manifest.BoundaryError, "stale profile metadata remains live"):
            strict_manifest.verify_manifest(manifest)

    def test_verify_rejects_stale_launcher_metadata(self) -> None:
        manifest = _valid_manifest()
        for entry in manifest["entries"]:  # type: ignore[index]
            if entry["path"] == "core/skill_registry.py#non_mining_launcher_defaults":
                entry["exists"] = True
                break

        with self.assertRaisesRegex(strict_manifest.BoundaryError, "stale launcher metadata remains live"):
            strict_manifest.verify_manifest(manifest)

    def test_verify_rejects_stale_import_from_mining_redux_path(self) -> None:
        manifest = _valid_manifest()
        for entry in manifest["entries"]:  # type: ignore[index]
            if entry["path"] == "skills/Cargo_loader":
                entry["mining_redux_import_references"] = [
                    {"path": "redux_mining_launcher.py", "line": 99, "token": "cargo_loader"}
                ]
                break

        with self.assertRaisesRegex(strict_manifest.BoundaryError, "Mining Redux still references removed non-Mining root"):
            strict_manifest.verify_manifest(manifest)

    def test_verify_rejects_shared_runtime_without_rationale(self) -> None:
        manifest = _valid_manifest()
        for entry in manifest["entries"]:  # type: ignore[index]
            if entry["path"] == "shared/qt":
                entry["rationale"] = ""
                break

        with self.assertRaisesRegex(strict_manifest.BoundaryError, "lacks rationale"):
            strict_manifest.verify_manifest(manifest)

    def test_verify_rejects_missing_reference_audit_field(self) -> None:
        manifest = _valid_manifest()
        for entry in manifest["entries"]:  # type: ignore[index]
            if entry["path"] == "skills/Cargo_loader":
                del entry["mining_redux_import_references"]
                break

        with self.assertRaisesRegex(strict_manifest.BoundaryError, "missing Mining Redux reference audit"):
            strict_manifest.verify_manifest(manifest)

    def test_verify_rejects_duplicate_paths(self) -> None:
        manifest = _valid_manifest()
        entries = manifest["entries"]  # type: ignore[index]
        entries.append(copy.deepcopy(entries[0]))
        manifest["summary"]["entry_count"] = len(entries)  # type: ignore[index]

        with self.assertRaisesRegex(strict_manifest.BoundaryError, "duplicate manifest path"):
            strict_manifest.verify_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
