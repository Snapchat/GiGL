import os
from unittest import mock

from parameterized import param, parameterized

from gigl.distributed.constants import (
    RANK_LOCAL_CSR_ROWS_ENV,
    resolve_rank_local_csr_rows,
)
from tests.test_assets.test_case import TestCase


class ResolveRankLocalCsrRowsTest(TestCase):
    @parameterized.expand(
        [
            param("empty", value="", expected=True),
            param("on", value="on", expected=True),
            param("on_uppercase", value="ON", expected=True),
            param("off", value="off", expected=False),
            param("off_uppercase", value="OFF", expected=False),
        ]
    )
    def test_values(self, _name: str, value: str, expected: bool) -> None:
        with mock.patch.dict(os.environ, {RANK_LOCAL_CSR_ROWS_ENV: value}):
            self.assertEqual(resolve_rank_local_csr_rows(), expected)

    def test_unset_means_on(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop(RANK_LOCAL_CSR_ROWS_ENV, None)
            self.assertTrue(resolve_rank_local_csr_rows())

    def test_an_unknown_value_raises(self) -> None:
        with mock.patch.dict(os.environ, {RANK_LOCAL_CSR_ROWS_ENV: "bogus"}):
            with self.assertRaises(ValueError):
                resolve_rank_local_csr_rows()


if __name__ == "__main__":
    from absl.testing import absltest

    absltest.main()
