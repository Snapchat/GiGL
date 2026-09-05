"""Exit-code contract for `post_install.py` run as a script.

This is the path `requirements/install_py_deps.sh` takes, so every Docker base image
build treats this exit code as the verdict on GLT: a zero exit publishes the image, and
a failed `install_glt.sh` that reports success ships an environment with no working GLT.
These tests run the real script as a subprocess against a stub `install_glt.sh` and
assert the process exit code, which is the only signal a build observes. The
`gigl-post-install` console script reaches `main()` by a different route and is not
covered here.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import gigl.scripts.post_install
from tests.test_assets.test_case import TestCase

# `post_install.py` resolves `install_glt.sh` as `Path(__file__).parent / "install_glt.sh"`,
# so a copy of the real script in a temp dir runs whichever stub sits beside it.
_POST_INSTALL_PATH: Path = Path(gigl.scripts.post_install.__file__)


class PostInstallTest(TestCase):
    def _run_post_install(
        self, install_glt_exit_code: Optional[int]
    ) -> "subprocess.CompletedProcess[str]":
        """Runs the real post-install script beside a stub `install_glt.sh`.

        Args:
            install_glt_exit_code (Optional[int]): Status the stub exits with, or None to
                leave the directory without an `install_glt.sh` at all.

        Returns:
            subprocess.CompletedProcess[str]: The finished process, for its `returncode`.
        """
        with tempfile.TemporaryDirectory() as script_dir:
            script_path = Path(script_dir) / _POST_INSTALL_PATH.name
            shutil.copyfile(_POST_INSTALL_PATH, script_path)
            if install_glt_exit_code is not None:
                # A two-line stub keeps the test hermetic: no network, no package install,
                # no real GLT build.
                (Path(script_dir) / "install_glt.sh").write_text(
                    f"#!/usr/bin/env bash\nexit {install_glt_exit_code}\n"
                )
            return subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
            )

    def test_exits_zero_when_install_glt_succeeds(self) -> None:
        completed = self._run_post_install(install_glt_exit_code=0)
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_propagates_install_glt_exit_code(self) -> None:
        # 7 is neither success nor the 1 the missing-script path uses, so matching it proves
        # the child's own status reached the caller.
        completed = self._run_post_install(install_glt_exit_code=7)
        self.assertEqual(completed.returncode, 7, completed.stdout)

    def test_exits_one_when_install_glt_is_missing(self) -> None:
        completed = self._run_post_install(install_glt_exit_code=None)
        self.assertEqual(completed.returncode, 1, completed.stdout)
