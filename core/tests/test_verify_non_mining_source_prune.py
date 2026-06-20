from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from build import verify_non_mining_source_prune as verifier


class VerifyNonMiningSourcePruneTests(unittest.TestCase):
    def test_manifest_entries_rejects_missing_entries_list(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "entries list"):
            verifier.manifest_entries({})

    def test_manifest_entries_rejects_path_escape(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "escapes repository root"):
            verifier.manifest_entries({"entries": [{"path": "../outside.py"}]})

    def test_manifest_entries_rejects_entry_count_mismatch(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "entry_count mismatch"):
            verifier.manifest_entries({"entry_count": 2, "entries": [{"path": "core"}]})

    def test_assert_no_delete_entries_rejects_delete_disposition(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "authorizes deletion"):
            verifier.assert_no_delete_entries(
                [{"path": "obsolete.py", "disposition": "delete", "deletion_allowed": False}]
            )

    def test_assert_no_delete_entries_rejects_deletion_allowed_flag(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "authorizes deletion"):
            verifier.assert_no_delete_entries(
                [{"path": "obsolete.py", "disposition": "keep", "deletion_allowed": True}]
            )

    def test_assert_audit_safe_requires_list_sections(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "must be a list"):
            verifier.assert_audit_safe({"attempted": {}, "deleted": [], "rejected": [], "skipped": []})

    def test_assert_audit_safe_rejects_attempted_or_deleted_paths(self) -> None:
        safe_base = {"attempted": [], "deleted": [], "rejected": [], "skipped": []}
        with self.assertRaisesRegex(verifier.VerificationError, "attempted source deletions"):
            verifier.assert_audit_safe({**safe_base, "attempted": [{"path": "x.py"}]})
        with self.assertRaisesRegex(verifier.VerificationError, "deleted source paths"):
            verifier.assert_audit_safe({**safe_base, "deleted": [{"path": "x.py"}]})

    def test_assert_required_manifest_paths_rejects_missing_manifest_path(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "protected paths missing manifest keep coverage"):
            verifier.assert_required_manifest_paths([])

    def test_assert_required_manifest_paths_rejects_unsafe_protected_entry(self) -> None:
        entries = [
            {"path": path, "disposition": "keep", "deletion_allowed": False}
            for path in verifier.PROTECTED_PATHS
        ]
        entries[0] = {"path": verifier.PROTECTED_PATHS[0], "disposition": "investigate", "deletion_allowed": False}
        with self.assertRaisesRegex(verifier.VerificationError, "not keep-safe"):
            verifier.assert_required_manifest_paths(entries)

    def test_load_json_file_rejects_malformed_json(self) -> None:
        with patch.object(Path, "is_file", return_value=True), patch.object(
            Path, "read_text", return_value="{not json"
        ):
            with self.assertRaisesRegex(verifier.VerificationError, "malformed JSON"):
                verifier.load_json_file(Path("manifest.json"), label="manifest")


if __name__ == "__main__":
    unittest.main()
