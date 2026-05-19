"""Regression tests for ``apply_sparse_optimizer`` in ``gigl.experimental.knowledge_graph_embedding.common.torchrec.utils``.

These tests guard the optimizer-flow bug surfaced by the ty migration: the
previous guard ``if not optimizer_cls and optimizer_kwargs`` only fired when
both arguments were truthy, allowing ``None`` to leak through into
``apply_optimizer_in_backward`` whenever ``optimizer_cls`` was unset but
``optimizer_kwargs`` was the default empty dict.
"""

from typing import Any
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.optim import SGD

from gigl.experimental.knowledge_graph_embedding.common.torchrec import utils
from tests.test_assets.test_case import TestCase


def _make_dummy_parameters() -> list[nn.Parameter]:
    """Build a one-parameter list so the optimizer call has something to bind to."""
    return [nn.Parameter(torch.zeros(1))]


class ApplySparseOptimizerTest(TestCase):
    def test_none_optimizer_cls_defaults_to_rowwise_adagrad(self) -> None:
        """Calling without ``optimizer_cls`` must default to RowWiseAdagrad.

        Regression: the old guard ``if not optimizer_cls and optimizer_kwargs``
        skipped the assignment when ``optimizer_kwargs`` defaulted to ``{}``,
        leaving ``optimizer_cls=None`` to crash ``apply_optimizer_in_backward``.
        """
        parameters = _make_dummy_parameters()
        with patch.object(utils, "apply_optimizer_in_backward") as mock_apply:
            utils.apply_sparse_optimizer(parameters=parameters)
            mock_apply.assert_called_once()
            forwarded_cls, forwarded_params, forwarded_kwargs = mock_apply.call_args[0]
            self.assertIs(forwarded_cls, utils.RowWiseAdagrad)
            self.assertEqual(list(forwarded_params), parameters)
            self.assertEqual(forwarded_kwargs, {"lr": 0.01})

    def test_none_optimizer_cls_with_kwargs_still_defaults_to_rowwise_adagrad(
        self,
    ) -> None:
        """``optimizer_cls=None`` with user kwargs uses RowWiseAdagrad + user kwargs.

        The user-supplied kwargs (when non-empty) are preserved verbatim; only
        the missing class is filled in.
        """
        parameters = _make_dummy_parameters()
        user_kwargs: dict[str, Any] = {"lr": 0.5}
        with patch.object(utils, "apply_optimizer_in_backward") as mock_apply:
            utils.apply_sparse_optimizer(
                parameters=parameters, optimizer_kwargs=user_kwargs
            )
            forwarded_cls, _, forwarded_kwargs = mock_apply.call_args[0]
            self.assertIs(forwarded_cls, utils.RowWiseAdagrad)
            self.assertEqual(forwarded_kwargs, {"lr": 0.5})

    def test_explicit_optimizer_cls_is_forwarded(self) -> None:
        """An explicit ``optimizer_cls`` must be forwarded unchanged."""
        parameters = _make_dummy_parameters()
        with patch.object(utils, "apply_optimizer_in_backward") as mock_apply:
            utils.apply_sparse_optimizer(parameters=parameters, optimizer_cls=SGD)
            forwarded_cls, _, forwarded_kwargs = mock_apply.call_args[0]
            self.assertIs(forwarded_cls, SGD)
            # No kwargs were supplied; the call site sees an empty dict, not None.
            self.assertEqual(forwarded_kwargs, {})

    def test_explicit_optimizer_cls_and_kwargs_forwarded(self) -> None:
        """Explicit ``optimizer_cls`` and kwargs must flow through verbatim."""
        parameters = _make_dummy_parameters()
        user_kwargs: dict[str, Any] = {"lr": 0.1, "momentum": 0.9}
        with patch.object(utils, "apply_optimizer_in_backward") as mock_apply:
            utils.apply_sparse_optimizer(
                parameters=parameters,
                optimizer_cls=SGD,
                optimizer_kwargs=user_kwargs,
            )
            forwarded_cls, _, forwarded_kwargs = mock_apply.call_args[0]
            self.assertIs(forwarded_cls, SGD)
            self.assertEqual(forwarded_kwargs, {"lr": 0.1, "momentum": 0.9})
