import unittest

import torch

from gigl.src.common.utils.eval_metrics import hit_rate_at_k, mean_reciprocal_rank


class EvalMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.neg_scores = torch.FloatTensor([3, 5, 7])

    def test_can_compute_hit_rates(self):
        ks = torch.LongTensor([1, 2, 3])

        # Compute perfect hit rate.
        hit_rates = hit_rate_at_k(
            pos_scores=torch.FloatTensor([9]), neg_scores=self.neg_scores, ks=ks
        )
        true_hit_rates = [1.0] * 3
        self.assertEqual(len(hit_rates), 3)
        for hit_rate, true_hit_rate in zip(hit_rates, true_hit_rates):
            self.assertEqual(hit_rate, true_hit_rate)

        # Compute imperfect hit rate.
        hit_rates = hit_rate_at_k(
            pos_scores=torch.FloatTensor([6]), neg_scores=self.neg_scores, ks=ks
        )
        true_hit_rates = [0.0, 1.0, 1.0]
        for hit_rate, true_hit_rate in zip(hit_rates, true_hit_rates):
            self.assertEqual(hit_rate, true_hit_rate)

        # Check that we always return hit rate of 1.0 for k if is the length of the total score vector.
        # In this case, we always have 1 pos score and 3 neg scores, so hits@4 should always = 1.0.
        for pos_score in [2, 4, 6, 8]:
            hit_rates = hit_rate_at_k(
                pos_scores=torch.FloatTensor([pos_score]),
                neg_scores=self.neg_scores,
                ks=torch.LongTensor([4]),
            )
            self.assertEqual(hit_rates[0].item(), 1.0)

        # Check that we can return hit-rate of 1 when k specified larger than viable based on input.
        hit_rates = hit_rate_at_k(
            pos_scores=torch.FloatTensor([2]),
            neg_scores=self.neg_scores,
            ks=torch.LongTensor([1, 2, 3, 4, 5]),
        )
        self.assertEqual(hit_rates.numel(), 5)
        self.assertEqual(hit_rates[-1].item(), 1.0)

        with self.assertRaises(AssertionError):
            hit_rates = hit_rate_at_k(
                pos_scores=torch.FloatTensor([9]),
                neg_scores=self.neg_scores,
                ks=torch.LongTensor([0]),
            )

    def test_can_compute_mean_reicprocal_rank(self):
        # Compute perfect MRR.
        mrr = mean_reciprocal_rank(
            pos_scores=torch.FloatTensor([9]),
            neg_scores=self.neg_scores,
        )
        true_mrr = 1.0
        self.assertEqual(mrr, true_mrr)

        # Compute imperfect MRR.
        mrr = mean_reciprocal_rank(
            pos_scores=torch.FloatTensor([6]),
            neg_scores=self.neg_scores,
        )
        true_mrr = 0.5
        self.assertEqual(mrr, true_mrr)

    def tearDown(self) -> None:
        pass
