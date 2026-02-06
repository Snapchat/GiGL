"""
Tests for loss_utils.py functions.

This module contains comprehensive tests for all functions in the loss_utils module,
including mathematical correctness verification and various test scenarios.
"""

import torch
import torch.nn.functional as F
from absl.testing import absltest

from gigl.experimental.knowledge_graph_embedding.lib.model.loss_utils import (
    average_pos_neg_scores,
    bpr_loss,
    hit_rate_at_k,
    infonce_loss,
    mean_reciprocal_rank,
)
from tests.test_assets.test_case import TestCase


class TestBPRLoss(TestCase):
    """Test suite for the BPR (Bayesian Personalized Ranking) loss function."""

    def test_bpr_loss_basic_functionality(self):
        """Test BPR loss with typical input shapes and values."""
        # Create test data: batch_size=2, 1 positive + 3 negatives
        scores = torch.tensor(
            [
                [2.0, 1.0, 0.5, 0.0],  # positive: 2.0, negatives: [1.0, 0.5, 0.0]
                [1.5, 0.8, 0.3, -0.2],  # positive: 1.5, negatives: [0.8, 0.3, -0.2]
            ]
        )
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])

        loss = bpr_loss(scores, labels)

        # Verify loss is scalar and positive
        self.assertEqual(loss.shape, torch.Size([]), "BPR loss should be scalar")
        self.assertGreaterEqual(loss.item(), 0, "BPR loss should be non-negative")

        # Manual calculation verification for first sample
        # diff = [2.0-1.0, 2.0-0.5, 2.0-0.0] = [1.0, 1.5, 2.0]
        # -log(sigmoid(diff)).mean() across all differences
        expected_diffs = torch.tensor([[1.0, 1.5, 2.0], [0.7, 1.2, 1.7]])
        expected_loss = -F.logsigmoid(expected_diffs).mean()

        self.assertTrue(
            torch.allclose(loss, expected_loss, atol=1e-6),
            f"Expected {expected_loss}, got {loss}",
        )

    def test_bpr_loss_single_sample(self):
        """Test BPR loss with single sample and single negative."""
        scores = torch.tensor([[1.5, 0.5]])  # 1 positive, 1 negative
        labels = torch.tensor([[1, 0]])

        loss = bpr_loss(scores, labels)

        # Manual calculation: -log(sigmoid(1.5 - 0.5)) = -log(sigmoid(1.0))
        expected_diff = 1.0
        expected_loss = -F.logsigmoid(torch.tensor(expected_diff))

        self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-6))

    def test_bpr_loss_perfect_ranking(self):
        """Test BPR loss when positive scores are much higher than negatives."""
        scores = torch.tensor(
            [
                [10.0, -5.0, -6.0, -7.0],  # Large positive, very negative negatives
                [8.0, -3.0, -4.0, -5.0],
            ]
        )
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])

        loss = bpr_loss(scores, labels)

        # With large differences, sigmoid should be close to 1, so loss should be close to 0
        self.assertLess(
            loss.item(),
            0.01,
            f"Loss should be very small for perfect ranking, got {loss}",
        )

    def test_bpr_loss_poor_ranking(self):
        """Test BPR loss when positive scores are lower than negatives."""
        scores = torch.tensor(
            [
                [-1.0, 2.0, 3.0, 4.0],  # Negative positive, positive negatives
                [-0.5, 1.0, 1.5, 2.0],
            ]
        )
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])

        loss = bpr_loss(scores, labels)

        # With negative differences, loss should be large
        self.assertGreater(
            loss.item(), 1.0, f"Loss should be large for poor ranking, got {loss}"
        )

    def test_bpr_loss_many_negatives(self):
        """Test BPR loss with large number of negatives."""
        batch_size = 3
        num_negatives = 10
        scores = torch.randn(batch_size, 1 + num_negatives)
        # Ensure positive scores are higher
        scores[:, 0] = scores[:, 0] + 2.0

        labels = torch.zeros_like(scores)
        labels[:, 0] = 1  # First column is positive

        loss = bpr_loss(scores, labels)

        self.assertEqual(loss.shape, torch.Size([]), "Loss should be scalar")
        self.assertGreaterEqual(loss.item(), 0, "Loss should be non-negative")


class TestInfoNCELoss(TestCase):
    """Test suite for the InfoNCE contrastive loss function."""

    def test_infonce_loss_basic_functionality(self):
        """Test InfoNCE loss with typical input shapes and values."""
        scores = torch.tensor([[2.0, 1.0, 0.5, 0.0], [1.5, 0.8, 0.3, -0.2]])
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])
        temperature = 1.0

        loss = infonce_loss(scores, labels, temperature)

        # Verify loss is scalar and positive
        self.assertEqual(loss.shape, torch.Size([]), "InfoNCE loss should be scalar")
        self.assertGreaterEqual(loss.item(), 0, "InfoNCE loss should be non-negative")

        # Manual verification: this should be equivalent to cross_entropy with target=0
        expected_loss = F.cross_entropy(
            scores, torch.zeros(scores.size(0), dtype=torch.long)
        )
        self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-6))

    def test_infonce_loss_temperature_scaling(self):
        """Test InfoNCE loss with different temperature values."""
        scores = torch.tensor([[2.0, 1.0, 0.5], [1.5, 0.8, 0.3]])
        labels = torch.tensor([[1, 0, 0], [1, 0, 0]])

        # Test different temperatures
        loss_temp_1 = infonce_loss(scores, labels, temperature=1.0)
        loss_temp_2 = infonce_loss(scores, labels, temperature=2.0)
        loss_temp_half = infonce_loss(scores, labels, temperature=0.5)

        # Higher temperature should generally give lower loss (softer distribution)
        # Lower temperature should give higher loss (sharper distribution)
        self.assertLess(
            loss_temp_half.item(),
            loss_temp_1.item(),
            "Lower temperature should increase loss",
        )
        self.assertLess(
            loss_temp_1.item(),
            loss_temp_2.item(),
            "Higher temperature should decrease loss",
        )

    def test_infonce_loss_perfect_ranking(self):
        """Test InfoNCE loss when positive scores are much higher."""
        scores = torch.tensor([[10.0, -5.0, -6.0], [8.0, -3.0, -4.0]])
        labels = torch.tensor([[1, 0, 0], [1, 0, 0]])

        loss = infonce_loss(scores, labels, temperature=1.0)

        # With large score differences, loss should be very small
        self.assertLess(
            loss.item(),
            0.01,
            f"Loss should be very small for perfect ranking, got {loss}",
        )

    def test_infonce_loss_single_sample(self):
        """Test InfoNCE loss with single sample."""
        scores = torch.tensor([[2.0, 1.0, 0.5]])
        labels = torch.tensor([[1, 0, 0]])

        loss = infonce_loss(scores, labels, temperature=1.0)

        # Manual calculation using cross entropy
        expected_loss = F.cross_entropy(scores, torch.tensor([0]))
        self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-6))


class TestAveragePosNegScores(TestCase):
    """Test suite for the average_pos_neg_scores function."""

    def test_average_pos_neg_scores_basic(self):
        """Test average positive and negative scores computation."""
        scores = torch.tensor([[2.0, 1.0, 0.5, 0.0], [1.5, 0.8, 0.3, -0.2]])
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])

        avg_pos, avg_neg = average_pos_neg_scores(scores, labels)

        # Manual calculation
        positive_scores = scores[labels == 1]  # [2.0, 1.5]
        negative_scores = scores[labels == 0]  # [1.0, 0.5, 0.0, 0.8, 0.3, -0.2]

        expected_avg_pos = positive_scores.mean()
        expected_avg_neg = negative_scores.mean()

        self.assertTrue(torch.allclose(avg_pos, expected_avg_pos, atol=1e-6))
        self.assertTrue(torch.allclose(avg_neg, expected_avg_neg, atol=1e-6))

    def test_average_pos_neg_scores_all_positive(self):
        """Test function when all labels are positive."""
        scores = torch.tensor([[2.0, 1.5, 1.0]])
        labels = torch.tensor([[1, 1, 1]])

        avg_pos, avg_neg = average_pos_neg_scores(scores, labels)

        expected_avg_pos = scores.mean()
        self.assertTrue(torch.allclose(avg_pos, expected_avg_pos, atol=1e-6))
        # avg_neg should be NaN when no negatives exist
        self.assertTrue(
            torch.isnan(avg_neg),
            "Average negative should be NaN when no negatives exist",
        )

    def test_average_pos_neg_scores_all_negative(self):
        """Test function when all labels are negative."""
        scores = torch.tensor([[2.0, 1.5, 1.0]])
        labels = torch.tensor([[0, 0, 0]])

        avg_pos, avg_neg = average_pos_neg_scores(scores, labels)

        expected_avg_neg = scores.mean()
        self.assertTrue(torch.allclose(avg_neg, expected_avg_neg, atol=1e-6))
        # avg_pos should be NaN when no positives exist
        self.assertTrue(
            torch.isnan(avg_pos),
            "Average positive should be NaN when no positives exist",
        )

    def test_average_pos_neg_scores_different_shapes(self):
        """Test function with different tensor shapes."""
        # 1D tensors
        scores_1d = torch.tensor([2.0, 1.0, 0.5, 0.0])
        labels_1d = torch.tensor([1, 0, 0, 1])

        avg_pos, avg_neg = average_pos_neg_scores(scores_1d, labels_1d)

        positive_scores = scores_1d[labels_1d == 1]  # [2.0, 0.0]
        negative_scores = scores_1d[labels_1d == 0]  # [1.0, 0.5]

        self.assertTrue(torch.allclose(avg_pos, positive_scores.mean(), atol=1e-6))
        self.assertTrue(torch.allclose(avg_neg, negative_scores.mean(), atol=1e-6))


class TestHitRateAtK(TestCase):
    """Test suite for the hit_rate_at_k function."""

    def test_hit_rate_at_k_single_k(self):
        """Test hit rate calculation with a single k value."""
        # Create scores where positive is at different ranks
        scores = torch.tensor(
            [
                [3.0, 2.0, 1.0, 0.5],  # positive is rank 1 (highest)
                [1.0, 3.0, 2.0, 0.5],  # positive is rank 3 (third highest)
                [0.5, 3.0, 2.0, 1.0],  # positive is rank 4 (lowest)
            ]
        )
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Test Hit@1
        hit_rate_1 = hit_rate_at_k(scores, labels, ks=1)
        self.assertTrue(
            torch.allclose(hit_rate_1, torch.tensor([1 / 3]), atol=1e-6),
            "Only first sample should hit at k=1",
        )

        # Test Hit@2
        hit_rate_2 = hit_rate_at_k(scores, labels, ks=2)
        self.assertTrue(
            torch.allclose(hit_rate_2, torch.tensor([1 / 3]), atol=1e-6),
            "Only first sample should hit at k=2",
        )

        # Test Hit@3
        hit_rate_3 = hit_rate_at_k(scores, labels, ks=3)
        self.assertTrue(
            torch.allclose(hit_rate_3, torch.tensor([2 / 3]), atol=1e-6),
            "First two samples should hit at k=3",
        )

    def test_hit_rate_at_k_multiple_k(self):
        """Test hit rate calculation with multiple k values."""
        scores = torch.tensor(
            [
                [3.0, 2.0, 1.0, 0.5],  # positive is rank 1
                [1.0, 3.0, 2.0, 0.5],  # positive is rank 3
            ]
        )
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])

        hit_rates = hit_rate_at_k(scores, labels, ks=[1, 2, 3])

        expected_hit_rates = torch.tensor([0.5, 0.5, 1.0])  # [Hit@1, Hit@2, Hit@3]
        self.assertTrue(torch.allclose(hit_rates, expected_hit_rates, atol=1e-6))

    def test_hit_rate_at_k_perfect_ranking(self):
        """Test hit rate when all positives are ranked first."""
        scores = torch.tensor(
            [[5.0, 2.0, 1.0, 0.5], [4.0, 3.0, 2.0, 1.0], [3.0, 2.5, 2.0, 1.5]]
        )
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        hit_rates = hit_rate_at_k(scores, labels, ks=[1, 2, 3])
        expected_hit_rates = torch.tensor([1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(hit_rates, expected_hit_rates, atol=1e-6))

    def test_hit_rate_at_k_worst_ranking(self):
        """Test hit rate when all positives are ranked last."""
        scores = torch.tensor(
            [
                [0.5, 3.0, 2.0, 1.0],  # positive is rank 4
                [1.0, 4.0, 3.0, 2.0],  # positive is rank 4
            ]
        )
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])

        hit_rates = hit_rate_at_k(scores, labels, ks=[1, 2, 3])
        expected_hit_rates = torch.tensor([0.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(hit_rates, expected_hit_rates, atol=1e-6))

    def test_hit_rate_at_k_single_sample(self):
        """Test hit rate with single sample."""
        scores = torch.tensor([[2.0, 1.5, 1.0, 0.5]])
        labels = torch.tensor([[1, 0, 0, 0]])

        hit_rate = hit_rate_at_k(scores, labels, ks=1)
        self.assertTrue(torch.allclose(hit_rate, torch.tensor([1.0]), atol=1e-6))


class TestMeanReciprocalRank(TestCase):
    """Test suite for the mean_reciprocal_rank function."""

    def test_mean_reciprocal_rank_basic(self):
        """Test MRR calculation with typical rankings."""
        scores = torch.tensor(
            [
                [3.0, 2.0, 1.0, 0.5],  # positive is rank 1, RR = 1/1 = 1.0
                [1.0, 3.0, 2.0, 0.5],  # positive is rank 3, RR = 1/3 ≈ 0.333
                [0.5, 1.0, 3.0, 2.0],  # positive is rank 4, RR = 1/4 = 0.25
            ]
        )
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        mrr = mean_reciprocal_rank(scores, labels)

        # Manual calculation: (1.0 + 1/3 + 1/4) / 3 = (1.0 + 0.333... + 0.25) / 3
        expected_mrr = (1.0 + 1 / 3 + 1 / 4) / 3
        self.assertTrue(torch.allclose(mrr, torch.tensor(expected_mrr), atol=1e-6))

    def test_mean_reciprocal_rank_perfect_ranking(self):
        """Test MRR when all positives are ranked first."""
        scores = torch.tensor(
            [[5.0, 2.0, 1.0, 0.5], [4.0, 3.0, 2.0, 1.0], [3.0, 2.5, 2.0, 1.5]]
        )
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        mrr = mean_reciprocal_rank(scores, labels)
        self.assertTrue(
            torch.allclose(mrr, torch.tensor(1.0), atol=1e-6),
            "MRR should be 1.0 for perfect ranking",
        )

    def test_mean_reciprocal_rank_worst_ranking(self):
        """Test MRR when all positives are ranked last."""
        scores = torch.tensor(
            [
                [0.5, 3.0, 2.0, 1.0],  # positive is rank 4, RR = 1/4 = 0.25
                [1.0, 4.0, 3.0, 2.0],  # positive is rank 4, RR = 1/4 = 0.25
            ]
        )
        labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])

        mrr = mean_reciprocal_rank(scores, labels)
        expected_mrr = 0.25  # (1/4 + 1/4) / 2 = 0.25
        self.assertTrue(torch.allclose(mrr, torch.tensor(expected_mrr), atol=1e-6))

    def test_mean_reciprocal_rank_tied_scores(self):
        """Test MRR with tied scores (positive should get the better rank)."""
        scores = torch.tensor(
            [
                [2.0, 2.0, 1.0, 0.5],  # positive tied for rank 1
            ]
        )
        labels = torch.tensor([[1, 0, 0, 0]])

        mrr = mean_reciprocal_rank(scores, labels)

        # With tied scores, argmax returns the first occurrence
        # So positive should be considered rank 1
        self.assertTrue(torch.allclose(mrr, torch.tensor(1.0), atol=1e-6))

    def test_mean_reciprocal_rank_single_sample(self):
        """Test MRR with single sample."""
        scores = torch.tensor([[1.5, 2.0, 0.5]])  # positive is rank 2
        labels = torch.tensor([[1, 0, 0]])

        mrr = mean_reciprocal_rank(scores, labels)
        expected_mrr = 1.0 / 2.0  # rank 2
        self.assertTrue(torch.allclose(mrr, torch.tensor(expected_mrr), atol=1e-6))


if __name__ == "__main__":
    absltest.main()
