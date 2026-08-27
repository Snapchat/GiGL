"""``post_install.py`` must propagate ``install_glt.sh``'s exit status as its own.

Image builds invoke the file directly (``requirements/install_py_deps.sh`` runs
``uv run python .../post_install.py``), so a swallowed child failure produces a
SUCCESSFUL Docker layer whose GLT wheel never built, installed, or verified. That is
the exact silent failure the patch-verification gate in ``install_glt.sh`` exists to
prevent, which makes the wrapper's exit code part of the gate.

These tests run the real file in a subprocess with ``bash`` shimmed ahead on ``PATH``,
so the entire path under test -- argument handling, ``__main__`` guard, exit-code
plumbing -- is the shipped one, with only the child's exit status under our control.
"""

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from absl.testing import absltest

from tests.test_assets.test_case import TestCase

_POST_INSTALL = Path(__file__).parents[3] / "gigl" / "scripts" / "post_install.py"


class PostInstallExitCodeTest(TestCase):
    def _run_with_shimmed_bash(self, bash_exit_code: int) -> int:
        """Run the real post_install.py with a fake ``bash`` that exits as told."""
        with tempfile.TemporaryDirectory() as shim_dir:
            shim = Path(shim_dir) / "bash"
            shim.write_text(f"#!/bin/sh\nexit {bash_exit_code}\n")
            shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
            environment = os.environ.copy()
            environment["PATH"] = f"{shim_dir}:{environment['PATH']}"
            completed = subprocess.run(
                [sys.executable, str(_POST_INSTALL)],
                env=environment,
                capture_output=True,
                text=True,
            )
        return completed.returncode

    def test_a_failing_install_script_fails_the_wrapper_process(self) -> None:
        self.assertEqual(self._run_with_shimmed_bash(7), 7)

    def test_a_succeeding_install_script_exits_zero(self) -> None:
        self.assertEqual(self._run_with_shimmed_bash(0), 0)


if __name__ == "__main__":
    absltest.main()
