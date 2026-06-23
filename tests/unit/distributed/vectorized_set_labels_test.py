"""Unit tests for the vectorized ABLP label-remap kernel and the collate-impl flag.

These tests exercise the pure-tensor label-remap logic directly (no GLT, no
distributed runtime), so they run in-process without ``mp.spawn``.
"""

import os
from unittest import mock

from absl.testing import absltest, parameterized

from gigl.distributed.utils.neighborloader import (
    COLLATE_IMPL_ENV_VAR,
    resolve_collate_impl,
)


class ResolveCollateImplTest(parameterized.TestCase):
    @parameterized.parameters(
        ("python", "python"),
        ("vectorized", "vectorized"),
        ("cpp", "cpp"),
        ("VECTORIZED", "vectorized"),  # case-insensitive
    )
    def test_valid_values(self, env_value: str, expected: str) -> None:
        with mock.patch.dict(os.environ, {COLLATE_IMPL_ENV_VAR: env_value}):
            self.assertEqual(resolve_collate_impl(), expected)

    def test_unset_defaults_to_python(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_collate_impl(), "python")

    def test_invalid_value_raises(self) -> None:
        with mock.patch.dict(os.environ, {COLLATE_IMPL_ENV_VAR: "rust"}):
            with self.assertRaises(ValueError):
                resolve_collate_impl()


if __name__ == "__main__":
    absltest.main()
