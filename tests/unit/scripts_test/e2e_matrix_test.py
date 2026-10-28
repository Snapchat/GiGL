"""``.github/scripts/e2e_matrix.py`` picks the e2e legs from the PR comment."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from absl.testing import absltest
from parameterized import parameterized

from tests.test_assets.test_case import TestCase

_SCRIPT = Path(__file__).parents[3] / ".github" / "scripts" / "e2e_matrix.py"
_DEFAULT_ONLY = [{"python": "3.11", "target": "run_all_e2e_tests"}]
_ALL_MINORS = _DEFAULT_ONLY + [
    {"python": "3.12", "target": "run_glt_e2e_tests"},
    {"python": "3.13", "target": "run_glt_e2e_tests"},
]


class E2EMatrixTest(TestCase):
    @parameterized.expand(
        [
            ("/e2e_test", _DEFAULT_ONLY),
            ("/e2e_test all", _ALL_MINORS),
            ("/E2E_TEST ALL", _ALL_MINORS),
            ("/all_test", _DEFAULT_ONLY),
            ("/e2e_test  all", _DEFAULT_ONLY),
            ("/e2e_testall", _DEFAULT_ONLY),
        ]
    )
    def test_matrix(self, comment_body: str, expected: list[dict[str, str]]) -> None:
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "gigl").mkdir()
            (Path(root) / "gigl" / "dep_vars.env").write_text(
                "SUPPORTED_PYTHON_VERSIONS=3.11.14,3.12.14,3.13.15\n"
            )
            (Path(root) / ".python-version").write_text("3.11.14\n")
            result = subprocess.run(
                [sys.executable, str(_SCRIPT), "--repo-root", root],
                capture_output=True,
                text=True,
                env={**os.environ, "COMMENT_BODY": comment_body},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.removeprefix("matrix=")), expected)


if __name__ == "__main__":
    absltest.main()
