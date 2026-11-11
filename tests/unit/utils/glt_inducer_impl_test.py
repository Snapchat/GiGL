"""make unit_test_py PY_TEST_FILES="glt_inducer_impl_test.py"

The flat hash table half of ``gigl/scripts/patches/0004-glt-inducer-flat-hash.patch``:
``GIGL_GLT_INDUCER_IMPL=flat`` must give element-wise identical inducer output to the default.

Skips when the installed graphlearn_torch has no ``inducer_impl`` (an unpatched wheel).
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import torch
from graphlearn_torch import py_graphlearn_torch as pywrap

from tests.test_assets.test_case import TestCase

_GIGL_ROOT = Path(__file__).resolve().parents[3]


def _run_scenarios(impl: Optional[str]) -> dict:
    """Runs the scenarios in a fresh process with ``GIGL_GLT_INDUCER_IMPL`` set to ``impl``."""
    env = {k: v for k, v in os.environ.items() if k != "GIGL_GLT_INDUCER_IMPL"}
    if impl is not None:
        env["GIGL_GLT_INDUCER_IMPL"] = impl
    with tempfile.TemporaryDirectory() as tmp:
        output_path = str(Path(tmp) / "out.pt")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.unit.utils.glt_inducer_scenarios",
                output_path,
            ],
            cwd=_GIGL_ROOT,
            env=env,
            check=True,
        )
        return torch.load(output_path)


@unittest.skipUnless(
    hasattr(pywrap, "inducer_impl"),
    "installed graphlearn_torch has no inducer_impl: the flat inducer patch is not in this wheel",
)
class GltInducerImplTest(TestCase):
    def test_flat_matches_std_element_wise(self) -> None:
        """Covers the homogeneous and heterogeneous inducers over reused instances with repeated
        ids, zero-neighbor sources, ids near 2**62 and INT64_MIN, and a 1e6-edge batch."""
        std = _run_scenarios(None)
        flat = _run_scenarios("flat")
        self.assertEqual(std["impl"], "std")
        self.assertEqual(flat["impl"], "flat")
        self.assertEqual(len(std["outputs"]), len(flat["outputs"]))
        self.assertGreater(len(std["outputs"]), 200)
        for expected, actual in zip(std["outputs"], flat["outputs"]):
            self.assertTrue(torch.equal(expected, actual))

    def test_unknown_impl_falls_back_to_std(self) -> None:
        self.assertEqual(_run_scenarios("bogus")["impl"], "std")


if __name__ == "__main__":
    unittest.main()
