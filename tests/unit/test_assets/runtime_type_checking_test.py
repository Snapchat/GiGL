"""Tests for test-only runtime Shape Contract checking."""

import os
from types import FunctionType
from unittest.mock import patch

import torch
from jaxtyping import Float32, TypeCheckError, jaxtyped

from tests.test_assets import runtime_type_checking
from tests.test_assets.test_case import TestCase


def _identity(
    tensor: Float32[torch.Tensor, "rows columns"],
) -> Float32[torch.Tensor, "rows columns"]:
    return tensor


def _typechecked_identity() -> FunctionType:
    return jaxtyped(typechecker=runtime_type_checking.shape_contract_typechecker)(
        _identity
    )


class RuntimeTypeCheckingTest(TestCase):
    def test_summary_reports_process_and_instrumented_function_count(self) -> None:
        with patch.object(runtime_type_checking.logger, "info") as mock_info:
            runtime_type_checking._log_runtime_typechecking_summary()

        summary = mock_info.call_args.args[0]
        self.assertIn(f"pid={os.getpid()}", summary)
        self.assertIn(
            f"instrumented_functions={len(runtime_type_checking._instrumented_functions)}",
            summary,
        )

    def test_shape_contract_typechecker_instruments_function(self) -> None:
        wrapped_identity = _typechecked_identity()

        tensor = torch.zeros((2, 3), dtype=torch.float32)
        self.assertIs(wrapped_identity(tensor), tensor)
        self.assertIn(
            f"{_identity.__module__}.{_identity.__qualname__}",
            runtime_type_checking._instrumented_functions,
        )

    def test_shape_contract_typechecker_rejects_wrong_rank(self) -> None:
        wrapped_identity = _typechecked_identity()

        with self.assertRaises(TypeCheckError):
            wrapped_identity(torch.zeros(3, dtype=torch.float32))
