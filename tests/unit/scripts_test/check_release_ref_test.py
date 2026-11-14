"""``.github/scripts/check_release_ref.py`` must pass only on the tag ``v<pyproject version>``."""

import os
import subprocess
import sys
from pathlib import Path

from absl.testing import absltest

from tests.test_assets.test_case import TestCase

_REPO_ROOT = Path(__file__).parents[3]
_SCRIPT = _REPO_ROOT / ".github" / "scripts" / "check_release_ref.py"


def _project_version() -> str:
    import tomllib

    return tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]


class CheckReleaseRefTest(TestCase):
    def _run(self, ref_type: str, ref_name: str) -> subprocess.CompletedProcess:
        environment = {
            **os.environ,
            "GITHUB_REF_TYPE": ref_type,
            "GITHUB_REF_NAME": ref_name,
        }
        return subprocess.run(
            [sys.executable, str(_SCRIPT)],
            env=environment,
            capture_output=True,
            text=True,
        )

    def test_matching_tag_passes(self) -> None:
        result = self._run("tag", f"v{_project_version()}")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_branch_fails(self) -> None:
        result = self._run("branch", f"v{_project_version()}")
        self.assertEqual(result.returncode, 1)
        self.assertIn("dispatched on branch", result.stderr)

    def test_mismatched_tag_fails(self) -> None:
        result = self._run("tag", "v0.0.0-not-this-version")
        self.assertEqual(result.returncode, 1)
        self.assertIn("!= tag v0.0.0-not-this-version", result.stderr)


if __name__ == "__main__":
    absltest.main()
