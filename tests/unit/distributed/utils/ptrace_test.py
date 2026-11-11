import sys
from unittest import mock

from gigl.distributed.utils import ptrace
from tests.test_assets.test_case import TestCase


class AllowAnyPtracerTest(TestCase):
    def test_returns_true_on_linux(self) -> None:
        if not sys.platform.startswith("linux"):
            self.skipTest("prctl(PR_SET_PTRACER) is Linux-only")
        self.assertTrue(ptrace.allow_any_ptracer())

    def test_returns_false_when_prctl_fails(self) -> None:
        fake_libc = mock.Mock()
        fake_libc.prctl.return_value = -1
        with mock.patch.object(ptrace.ctypes, "CDLL", return_value=fake_libc):
            self.assertFalse(ptrace.allow_any_ptracer())

    def test_returns_false_when_libc_unavailable(self) -> None:
        with mock.patch.object(ptrace.ctypes, "CDLL", side_effect=OSError("no libc")):
            self.assertFalse(ptrace.allow_any_ptracer())
