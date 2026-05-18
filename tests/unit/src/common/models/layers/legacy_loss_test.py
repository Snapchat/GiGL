"""Regression tests for the legacy loss layers under ``gigl.src.common.models.layers.loss``.

These tests guard against latent ``None`` propagation bugs that the ty migration
surfaced. In particular, ``SoftmaxLoss.softmax_temperature`` used to be typed as
``Optional[float]`` with a ``None`` default, which made the in-loop expression
``all_scores / self.softmax_temperature`` raise ``TypeError`` at runtime whenever
the constructor was called without an explicit temperature.
"""

import torch

from gigl.src.common.models.layers.loss import SoftmaxLoss
from gigl.src.common.types.graph_data import CondensedEdgeType
from gigl.src.common.types.task_inputs import BatchScores
from tests.test_assets.test_case import TestCase


def _make_batch_scores() -> dict[CondensedEdgeType, BatchScores]:
    """Build a minimal non-empty BatchScores dict suitable for SoftmaxLoss.forward.

    ``BatchScores`` is annotated as ``FloatTensor`` but PyTorch factories return
    plain ``Tensor`` — PR 3 of the ty-cleanup plan will relax these annotations
    to ``Tensor``. Until then, ty rejects the call, so we ignore the
    specialization mismatch at the construction site.
    """
    pos_scores = torch.tensor([[0.8, 0.6]])
    hard_neg_scores = torch.tensor([[0.1, 0.2]])
    random_neg_scores = torch.tensor([[0.05, 0.15]])
    return {
        CondensedEdgeType(0): BatchScores(
            pos_scores=pos_scores,  # ty: ignore[invalid-argument-type] TODO(ty-torch-tensor-specialization): cleared by PR 3 relaxing BatchScores to Tensor.
            hard_neg_scores=hard_neg_scores,  # ty: ignore[invalid-argument-type] TODO(ty-torch-tensor-specialization): cleared by PR 3 relaxing BatchScores to Tensor.
            random_neg_scores=random_neg_scores,  # ty: ignore[invalid-argument-type] TODO(ty-torch-tensor-specialization): cleared by PR 3 relaxing BatchScores to Tensor.
        )
    }


class SoftmaxLossTest(TestCase):
    def test_default_temperature_does_not_raise(self) -> None:
        """SoftmaxLoss must be constructible without explicit temperature.

        Regression: before the ty-driven fix, ``softmax_temperature`` defaulted to
        ``None``, so ``all_scores / self.softmax_temperature`` raised
        ``TypeError: unsupported operand type(s) for /: 'Tensor' and 'NoneType'``.
        """
        loss_fn = SoftmaxLoss()
        # The default is 1.0 (no scaling), and dividing by 1.0 must not raise.
        self.assertEqual(loss_fn.softmax_temperature, 1.0)
        loss, sample_size = loss_fn(loss_input=[_make_batch_scores()])
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(sample_size, 0)

    def test_default_temperature_matches_unit_scale(self) -> None:
        """Default temperature ``1.0`` produces the same loss as explicit ``1.0``."""
        default_loss, _ = SoftmaxLoss()(loss_input=[_make_batch_scores()])
        explicit_loss, _ = SoftmaxLoss(softmax_temperature=1.0)(
            loss_input=[_make_batch_scores()]
        )
        self.assert_tensor_equality(default_loss, explicit_loss)

    def test_custom_temperature_scales_loss(self) -> None:
        """Smaller temperatures sharpen the softmax, yielding a different loss."""
        unit_loss, _ = SoftmaxLoss(softmax_temperature=1.0)(
            loss_input=[_make_batch_scores()]
        )
        sharper_loss, _ = SoftmaxLoss(softmax_temperature=0.5)(
            loss_input=[_make_batch_scores()]
        )
        # The two scalar losses must differ; both must be finite.
        self.assertTrue(torch.isfinite(unit_loss).item())
        self.assertTrue(torch.isfinite(sharper_loss).item())
        self.assertNotAlmostEqual(unit_loss.item(), sharper_loss.item())

    def test_empty_input_returns_zero_loss(self) -> None:
        """An empty ``loss_input`` list yields a zero loss with sample_size 1."""
        loss, sample_size = SoftmaxLoss()(loss_input=[])
        self.assert_tensor_equality(loss, torch.tensor(0.0))
        self.assertEqual(sample_size, 1)
