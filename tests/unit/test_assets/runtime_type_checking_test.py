"""Unit tests for the test-only Jaxtyping hook in runtime_type_checking."""

import cloudpickle
import numpy as np
from absl.testing import absltest
from jaxtyping import Float, TypeCheckError

from tests.test_assets.runtime_type_checking import (
    _jaxtyped_shape_contracts_only,
    shape_contract_typechecker,
)
from tests.test_assets.test_case import TestCase


def _decorate(function):
    # The import hook emits ``@jaxtyping.jaxtyped(typechecker=...)`` on every function.
    return _jaxtyped_shape_contracts_only(typechecker=shape_contract_typechecker)(
        function
    )


class JaxtypedShapeContractsOnlyTest(TestCase):
    def test_function_without_shape_contract_is_left_unwrapped_and_picklable(
        self,
    ) -> None:
        def no_contract(x: int) -> int:
            return x + 1

        decorated = _decorate(no_contract)

        self.assertIs(decorated, no_contract)
        self.assertEqual(cloudpickle.loads(cloudpickle.dumps(decorated))(1), 2)

    def test_function_with_shape_contract_is_still_checked(self) -> None:
        def with_contract(x: Float[np.ndarray, "n 2"]) -> int:
            return x.shape[0]

        decorated = _decorate(with_contract)

        self.assertIsNot(decorated, with_contract)
        self.assertEqual(decorated(np.zeros((3, 2))), 3)
        with self.assertRaises(TypeCheckError):
            decorated(np.zeros((3, 5)))


if __name__ == "__main__":
    absltest.main()
