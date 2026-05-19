"""Regression tests for ``apply_sparse_optimizer``."""

from unittest.mock import patch

import torch
import torch.nn as nn

from gigl.experimental.knowledge_graph_embedding.common.torchrec import utils
from tests.test_assets.test_case import TestCase


class ApplySparseOptimizerTest(TestCase):
    def test_default_optimizer_uses_rowwise_adagrad(self) -> None:
        """When called with neither ``optimizer_cls`` nor ``optimizer_kwargs``,
        ``apply_sparse_optimizer`` falls back to ``RowWiseAdagrad`` with
        ``lr=0.01`` and forwards them to ``apply_optimizer_in_backward``.
        """
        parameters = [nn.Parameter(torch.zeros(1))]
        with patch.object(utils, "apply_optimizer_in_backward") as mock_apply:
            utils.apply_sparse_optimizer(parameters=parameters)
        forwarded_cls, _, forwarded_kwargs = mock_apply.call_args[0]
        self.assertIs(forwarded_cls, utils.RowWiseAdagrad)
        self.assertEqual(forwarded_kwargs, {"lr": 0.01})
