from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build import verify_mining_cleanup_closeout as verifier


class VerifyMiningCleanupCloseoutTests(unittest.TestCase):
    def test_generated_manifest_count_rejects_mismatch(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "entry_count mismatch"):
            verifier.generated_manifest_count({"entry_count": 2, "entries": [{"path": "build/file.txt"}]})

    def test_manifest_entries_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(verifier.VerificationError, "escapes repository root"):
                verifier.manifest_entries({"entries": [{"path": "../outside.py"}]}, root=Path(tmp))

    def test_source_manifest_rejects_delete_disposition(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "authorizes deletion"):
            verifier.assert_no_source_delete_entries(
                [{"path": "tools/Other/app.py", "disposition": "delete", "deletion_allowed": False}]
            )

    def test_source_manifest_rejects_deletion_allowed_flag(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "authorizes deletion"):
            verifier.assert_no_source_delete_entries(
                [{"path": "tools/Other/app.py", "disposition": "keep", "deletion_allowed": True}]
            )

    def test_source_audit_zero_accepts_empty_counts(self) -> None:
        counts = verifier.assert_source_audit_zero(
            {"attempted": [], "deleted": [], "rejected": [], "skipped": ["tools/Other"], "summary": {}}
        )
        self.assertEqual(counts, {"attempted": 0, "deleted": 0, "rejected": 0, "skipped": 1})

    def test_source_audit_zero_rejects_rejected_paths(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "rejected paths unexpectedly non-zero"):
            verifier.assert_source_audit_zero({"attempted": [], "deleted": [], "rejected": ["core/runtime.py"]})

    def test_source_audit_zero_rejects_summary_counts(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "summary reports unsafe counts"):
            verifier.assert_source_audit_zero(
                {"attempted": [], "deleted": [], "rejected": [], "summary": {"attempted_count": 1}}
            )

    def test_protected_paths_present_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "present").write_text("ok", encoding="utf-8")
            with self.assertRaisesRegex(verifier.VerificationError, "protected paths missing: missing"):
                verifier.assert_protected_paths_present(root=root, paths=("present", "missing"))

    def test_installer_smoke_classifier_requires_exit_zero(self) -> None:
        result = verifier.CommandResult(
            args=("python", "build/installer_release_smoke.py"),
            returncode=1,
            stdout="",
            stderr="boom",
        )
        with self.assertRaisesRegex(verifier.VerificationError, "installer smoke failed exit=1"):
            verifier.classify_installer_smoke(result)

    def test_installer_smoke_classifier_rejects_malformed_json(self) -> None:
        result = verifier.CommandResult(args=("python",), returncode=0, stdout="not json", stderr="")
        with self.assertRaisesRegex(verifier.VerificationError, "malformed JSON"):
            verifier.classify_installer_smoke(result)

    def test_installer_smoke_classifier_requires_mining_and_full_ok(self) -> None:
        result = verifier.CommandResult(
            args=("python",),
            returncode=0,
            stdout=json.dumps({"ok": True, "profiles": {"mining": {"ok": True}, "full": {"ok": False}}}),
            stderr="",
        )
        with self.assertRaisesRegex(verifier.VerificationError, "profile 'full' did not report ok=true"):
            verifier.classify_installer_smoke(result, expected_profiles=("mining", "full"))

    def test_installer_smoke_classifier_accepts_combined_success(self) -> None:
        result = verifier.CommandResult(
            args=("python",),
            returncode=0,
            stdout=json.dumps({"ok": True, "profiles": {"mining": {"ok": True}, "full": {"ok": True}}}),
            stderr="",
        )
        self.assertEqual(
            verifier.classify_installer_smoke(result, expected_profiles=("mining", "full")),
            "exit=0 profiles=mining,full",
        )

    def test_installer_smoke_classifier_accepts_current_manifest_mining_only_success(self) -> None:
        result = verifier.CommandResult(
            args=("python",),
            returncode=0,
            stdout=json.dumps({"ok": True, "profiles": {"mining": {"ok": True}}}),
            stderr="",
        )
        self.assertEqual(verifier.classify_installer_smoke(result), "exit=0 profiles=mining")

    def test_run_required_command_bubbles_nonzero_result(self) -> None:
        def fake_runner(args, timeout):
            return verifier.CommandResult(args=tuple(args), returncode=2, stdout="", stderr="bad")

        with self.assertRaisesRegex(verifier.VerificationError, "S03 failed exit=2"):
            verifier.run_required_command(("cmd",), label="S03", timeout=1, runner=fake_runner)


if __name__ == "__main__":
    unittest.main()
