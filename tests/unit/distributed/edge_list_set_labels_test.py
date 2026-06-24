"""Unit tests for the dense edge-list ABLP label kernel and its format selector.

These exercise the pure-tensor label-remap logic directly (no GLT, no
distributed runtime), so they run in-process without ``mp.spawn``.
"""

import os
from unittest import mock

from absl.testing import absltest, parameterized

from gigl.distributed.utils.neighborloader import (
    ABLP_LABEL_FORMAT_ENV_VAR,
    resolve_ablp_label_format,
)


class ResolveAblpLabelFormatTest(parameterized.TestCase):
    @parameterized.parameters(
        ("dict", "dict"),
        ("edge_list", "edge_list"),
        ("EDGE_LIST", "edge_list"),
    )
    def test_valid_values(self, env_value: str, expected: str) -> None:
        with mock.patch.dict(os.environ, {ABLP_LABEL_FORMAT_ENV_VAR: env_value}):
            self.assertEqual(resolve_ablp_label_format(), expected)

    def test_unset_defaults_to_dict(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_ablp_label_format(), "dict")

    def test_invalid_value_raises(self) -> None:
        with mock.patch.dict(os.environ, {ABLP_LABEL_FORMAT_ENV_VAR: "ragged"}):
            with self.assertRaises(ValueError):
                resolve_ablp_label_format()


if __name__ == "__main__":
    absltest.main()
