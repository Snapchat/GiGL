"""``post_install.py`` must exit non-zero when the GLT install fails.

Image builds invoke the file directly (``requirements/install_py_deps.sh`` runs
``uv run python .../post_install.py``), so a swallowed failure produces a SUCCESSFUL Docker layer
whose GLT wheel never built, installed, or verified. The wrapper's exit code is part of the gate.

Runs the real file in a subprocess with ``bash`` shimmed ahead on ``PATH`` and the wheel cache
pointed at an empty temp dir, so the build step is the one under our control.
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
    def test_a_failing_build_fails_the_wrapper_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shim = Path(tmp) / "bin" / "bash"
            shim.parent.mkdir()
            shim.write_text("#!/bin/sh\nexit 7\n")
            shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
            environment = os.environ.copy()
            environment["PATH"] = f"{shim.parent}:{environment['PATH']}"
            environment["GIGL_GLT_WHEEL_CACHE_DIR"] = str(Path(tmp) / "cache")
            completed = subprocess.run(
                [sys.executable, str(_POST_INSTALL)],
                env=environment,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    absltest.main()
