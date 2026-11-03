import jaxtyping
import torch
from absl.testing import absltest
from jaxtyping import Float

from tests.test_assets.runtime_type_checking import (
    install_runtime_typechecking,
    shape_contract_typechecker,
)
from tests.test_assets.test_case import TestCase


def _contracted(function):
    return jaxtyping.jaxtyped(typechecker=shape_contract_typechecker)(function)


@_contracted
def _double(x: Float[torch.Tensor, "rows 2"]) -> Float[torch.Tensor, "rows 2"]:
    return x * 2


@_contracted
def _increment(x: Float[torch.Tensor, "rows 2"]) -> Float[torch.Tensor, "rows 2"]:
    return x + 1


class RuntimeTypeCheckingTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        install_runtime_typechecking()
        torch._dynamo.reset()

    def test_eager_call_with_wrong_shape_raises(self) -> None:
        with self.assertRaises(jaxtyping.TypeCheckError):
            _double(torch.ones(3, 4))

    def test_check_is_skipped_inside_a_compiled_frame(self) -> None:
        def run(x: torch.Tensor) -> torch.Tensor:
            return _double(x) + 0

        compiled = torch.compile(run, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(torch.ones(3, 4)), torch.full((3, 4), 2.0))

    def test_contracted_functions_as_frame_roots_do_not_recompile(self) -> None:
        # Both functions share the code object of Jaxtyping's wrapper.
        with torch._dynamo.config.patch(
            recompile_limit=1, fail_on_recompile_limit_hit=True
        ):
            torch.compile(_double, backend="eager")(torch.ones(3, 2))
            torch.compile(_increment, backend="eager")(torch.ones(3, 2))


if __name__ == "__main__":
    absltest.main()
