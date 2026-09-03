import importlib

from gigl.common.utils.parse import str_to_bool
from tests.test_assets.test_case import TestCase

_TRUE_SPELLINGS = ("y", "yes", "t", "true", "on", "1")
_FALSE_SPELLINGS = ("n", "no", "f", "false", "off", "0")
# `" true"` belongs here because `str_to_bool` does not strip surrounding whitespace.
_INVALID_VALUES = ("", " true", "2", "none")


def _casings(spelling: str) -> tuple[str, ...]:
    """All the casings `str_to_bool` must accept for one spelling."""
    return (spelling.lower(), spelling.upper(), spelling.capitalize())


class ParseUtilsTest(TestCase):
    def test_accepted_spellings(self) -> None:
        for spelling in _TRUE_SPELLINGS:
            for value in _casings(spelling):
                with self.subTest(value=value):
                    self.assertIs(str_to_bool(value), True)
        for spelling in _FALSE_SPELLINGS:
            for value in _casings(spelling):
                with self.subTest(value=value):
                    self.assertIs(str_to_bool(value), False)

    def test_rejects_unrecognized_values(self) -> None:
        for value in _INVALID_VALUES:
            with self.subTest(value=value):
                self.assertRaises(ValueError, str_to_bool, value)

    def test_matches_distutils_strtobool(self) -> None:
        # The import is dynamic because a static `from distutils.util import ...` fails the
        # 3.13 pass of `make type_check`. The reference is whichever `distutils` is
        # importable: CPython's on 3.11 without `setuptools`, otherwise the copy
        # `setuptools` injects through `distutils-precedence.pth`, which is what GiGL's dev
        # and build environments resolve on every Python version.
        try:
            strtobool = importlib.import_module("distutils.util").strtobool
        except ImportError:
            self.skipTest("distutils is not importable in this environment")

        for spelling in _TRUE_SPELLINGS + _FALSE_SPELLINGS:
            for value in _casings(spelling):
                with self.subTest(value=value):
                    self.assertEqual(str_to_bool(value), bool(strtobool(value)))
        for value in _INVALID_VALUES:
            with self.subTest(value=value):
                self.assertRaises(ValueError, strtobool, value)
