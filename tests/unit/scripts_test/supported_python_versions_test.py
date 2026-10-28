"""``.github/scripts/supported_python_versions.py`` parses and validates the supported-versions list."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from absl.testing import absltest

from tests.test_assets.test_case import TestCase

_SCRIPT = (
    Path(__file__).parents[3] / ".github" / "scripts" / "supported_python_versions.py"
)


class SupportedPythonVersionsTest(TestCase):
    def _run(self, versions: str, default: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "gigl").mkdir()
            (Path(root) / "gigl" / "dep_vars.env").write_text(
                f"# comment\nOTHER=1\nSUPPORTED_PYTHON_VERSIONS={versions}\n"
            )
            (Path(root) / ".python-version").write_text(default + "\n")
            return subprocess.run(
                [sys.executable, str(_SCRIPT), "--repo-root", root],
                capture_output=True,
                text=True,
            )

    def test_outputs(self) -> None:
        result = self._run("3.11.14,3.12.14,3.13.15", "3.12.14")
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = dict(line.split("=", 1) for line in result.stdout.splitlines())
        self.assertEqual(json.loads(outputs["minors"]), ["3.11", "3.12", "3.13"])
        self.assertEqual(outputs["default_minor"], "3.12")
        self.assertEqual(json.loads(outputs["nondefault_minors"]), ["3.11", "3.13"])
        self.assertEqual(outputs["minor_count"], "3")
        self.assertEqual(
            json.loads(outputs["builds"])[2], {"python": "3.13.15", "tag": "py313"}
        )

    def test_rejects_default_not_listed(self) -> None:
        result = self._run("3.11.14,3.12.14", "3.13.15")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not an entry of SUPPORTED_PYTHON_VERSIONS", result.stderr)

    def test_rejects_default_with_unlisted_patch(self) -> None:
        result = self._run("3.11.14,3.12.14", "3.11.9")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not an entry of SUPPORTED_PYTHON_VERSIONS", result.stderr)

    def test_rejects_duplicate_minor(self) -> None:
        result = self._run("3.11.14,3.11.13", "3.11.14")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("more than once", result.stderr)

    def test_rejects_minor_only_entry(self) -> None:
        result = self._run("3.11,3.12.14", "3.12.14")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("X.Y.Z", result.stderr)


if __name__ == "__main__":
    absltest.main()
