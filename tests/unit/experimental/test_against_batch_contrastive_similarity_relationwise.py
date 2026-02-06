import torch
from absl.testing import absltest

from gigl.experimental.knowledge_graph_embedding.lib.model.negative_sampling import (
    against_batch_relationwise_contrastive_similarity,
)
from gigl.experimental.knowledge_graph_embedding.lib.model.types import (
    NegativeSamplingCorruptionType,
    SimilarityType,
)
from tests.test_assets.test_case import TestCase


class TestAgainstBatchRelationwiseContrastiveSimilarity(TestCase):
    def test_correctness_dot_dst_one_negative(self):
        # Test dot product similarity with destination-side corruption and one negative sample.
        src_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dst_embeddings = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        condensed_edge_types = torch.tensor([0, 0])
        batch_src_embeddings = torch.tensor([[1.5, 0.1], [0.2, 2.5]])
        batch_dst_embeddings = torch.tensor([[2.1, 0.2], [0.3, 3.1]])
        batch_condensed_edge_types = torch.tensor([0, 1])
        temperature = 1.0
        num_negatives = 1
        corrupt_side = NegativeSamplingCorruptionType.DST
        scoring_function = SimilarityType.DOT

        logits, labels = against_batch_relationwise_contrastive_similarity(
            positive_src_embeddings=src_embeddings,
            positive_dst_embeddings=dst_embeddings,
            positive_condensed_edge_types=condensed_edge_types,
            negative_batch_src_embeddings=batch_src_embeddings,
            negative_batch_dst_embeddings=batch_dst_embeddings,
            batch_condensed_edge_types=batch_condensed_edge_types,
            temperature=temperature,
            scoring_function=scoring_function,
            corrupt_side=corrupt_side,
            num_negatives=num_negatives,
        )

        # Positive logits (src @ dst.T).diagonal(): [2., 3.]
        # Negative logits (src @ batch_dst.T) where relations match (type 0): [[1.0 * 2.1 + 0.0 * 0.2], [0.0 * 2.1 + 1.0 * 0.2]] => [[2.1], [0.2]]
        expected_logits = torch.tensor([[2.0, 2.1], [3.0, 0.2]])
        expected_labels = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        self.assertTrue(torch.allclose(logits, expected_logits))
        self.assertTrue(torch.allclose(labels, expected_labels))

    def test_correctness_cosine_src_two_negatives(self):
        # Test cosine similarity with source-side corruption and two negative samples.
        src_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dst_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        condensed_edge_types = torch.tensor([0, 0])
        batch_src_embeddings = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        batch_dst_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        batch_condensed_edge_types = torch.tensor([0, 0, 1])
        temperature = 1.0
        num_negatives = 2
        corrupt_side = NegativeSamplingCorruptionType.SRC
        scoring_function = SimilarityType.COSINE

        torch.manual_seed(42)  # For deterministic negative sampling

        logits, labels = against_batch_relationwise_contrastive_similarity(
            positive_src_embeddings=src_embeddings,
            positive_dst_embeddings=dst_embeddings,
            positive_condensed_edge_types=condensed_edge_types,
            negative_batch_src_embeddings=batch_src_embeddings,
            negative_batch_dst_embeddings=batch_dst_embeddings,
            batch_condensed_edge_types=batch_condensed_edge_types,
            temperature=temperature,
            scoring_function=scoring_function,
            corrupt_side=corrupt_side,
            num_negatives=num_negatives,
        )

        # Normalized embeddings for positive: [[1., 0.], [0., 1.]]
        # Positive logits: [1., 1.]
        # Normalized batch_src_embeddings: [[0., 1.], [1., 0.], [0.707, 0.707]]
        # Negative similarities (normalized batch_src @ normalized dst.T) for matching relation (type 0):
        # Row 0: [[0.*1 + 1.*0], [0.*0 + 1.*1]] => [0., 1.]
        # Row 1: [[1.*1 + 0.*0], [1.*0 + 0.*1]] => [1., 0.]
        # After masking non-matching (none here), and sampling 2: depends on seed, should pick the two.
        # Let's manually calculate the cosine similarities:
        # cos([1, 0], [0, 1]) = 0; cos([1, 0], [1, 0]) = 1
        # cos([0, 1], [0, 1]) = 1; cos([0, 1], [1, 0]) = 0
        expected_positive_logits = torch.tensor([[1.0], [1.0]])
        # Due to seeding, the top 2 should be the two type 0 examples.
        # The order might vary, but the values should be present.
        expected_neg_logits_row0 = torch.tensor([0.0, 1.0])
        expected_neg_logits_row1 = torch.tensor([1.0, 0.0])
        expected_labels = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        self.assertTrue(
            torch.allclose(logits[:, 0].unsqueeze(1), expected_positive_logits)
        )
        # Check if the negative logits contain the expected values (order might differ due to sampling)
        for i in range(2):
            self.assertEqual(logits[0, 1:].numel(), 2)
            self.assertEqual(logits[1, 1:].numel(), 2)
            for val in expected_neg_logits_row0:
                self.assertTrue(torch.any(torch.isclose(logits[0, 1:], val)))
            for val in expected_neg_logits_row1:
                self.assertTrue(torch.any(torch.isclose(logits[1, 1:], val)))
        self.assertTrue(torch.allclose(labels, expected_labels))

    def test_correctness_euclidean_both_none_negatives(self):
        # Test Euclidean similarity with both-side corruption and no explicit negative sampling.
        src_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dst_embeddings = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        condensed_edge_types = torch.tensor([0, 0])
        batch_src_embeddings = torch.tensor([[1.0, 1.0], [-1.0, 0.0]])
        batch_dst_embeddings = torch.tensor([[0.0, 0.0], [0.0, -1.0]])
        batch_condensed_edge_types = torch.tensor([0, 1])
        temperature = 1.0
        num_negatives = None
        corrupt_side = NegativeSamplingCorruptionType.BOTH
        scoring_function = SimilarityType.EUCLIDEAN

        torch.manual_seed(10)  # For deterministic 'both' corruption

        logits, labels = against_batch_relationwise_contrastive_similarity(
            positive_src_embeddings=src_embeddings,
            positive_dst_embeddings=dst_embeddings,
            positive_condensed_edge_types=condensed_edge_types,
            negative_batch_src_embeddings=batch_src_embeddings,
            negative_batch_dst_embeddings=batch_dst_embeddings,
            batch_condensed_edge_types=batch_condensed_edge_types,
            temperature=temperature,
            scoring_function=scoring_function,
            corrupt_side=corrupt_side,
            num_negatives=num_negatives,
        )

        # Positive squared Euclidean distance:
        # ||[1, 0] - [0, 1]||^2 = 1^2 + (-1)^2 = 2 => similarity = -2
        # ||[0, 1] - [1, 0]||^2 = (-1)^2 + 1^2 = 2 => similarity = -2
        expected_positive_logits = torch.tensor([[-2.0], [-2.0]])
        pos_logits = logits[:, 0].unsqueeze(1)
        self.assertTrue(torch.allclose(pos_logits, expected_positive_logits))
        self.assertEqual(logits.shape, (2, 3))

        expected_labels = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(labels, expected_labels))

    def test_correctness_dot_dst_zero_negatives(self):
        # Test dot product with destination-side corruption and zero negative samples.
        src_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dst_embeddings = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        condensed_edge_types = torch.tensor([0, 0])
        batch_src_embeddings = torch.tensor([[1.5, 0.1], [0.2, 2.5]])
        batch_dst_embeddings = torch.tensor([[2.1, 0.2], [0.3, 3.1]])
        batch_condensed_edge_types = torch.tensor([0, 1])
        temperature = 1.0
        num_negatives = 0
        corrupt_side = NegativeSamplingCorruptionType.DST
        scoring_function = SimilarityType.DOT

        logits, labels = against_batch_relationwise_contrastive_similarity(
            positive_src_embeddings=src_embeddings,
            positive_dst_embeddings=dst_embeddings,
            positive_condensed_edge_types=condensed_edge_types,
            negative_batch_src_embeddings=batch_src_embeddings,
            negative_batch_dst_embeddings=batch_dst_embeddings,
            batch_condensed_edge_types=batch_condensed_edge_types,
            temperature=temperature,
            scoring_function=scoring_function,
            num_negatives=num_negatives,
        )

        # Only positive logits
        expected_logits = torch.tensor([[2.0], [3.0]])
        expected_labels = torch.tensor([[1.0], [1.0]])

        self.assertTrue(torch.allclose(logits, expected_logits))
        self.assertTrue(torch.allclose(labels, expected_labels))
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(labels.shape, (2, 1))

    def test_correctness_dot_src_none_negatives(self):
        # Test dot product with source-side corruption and "full" negative sampling (all in-batch negatives used).
        src_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dst_embeddings = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        condensed_edge_types = torch.tensor([0, 0])
        batch_src_embeddings = torch.tensor([[1.5, 0.1], [0.0, 1.0], [1.0, 0.0]])
        batch_dst_embeddings = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
        batch_condensed_edge_types = torch.tensor([0, 0, 1])
        temperature = 1.0
        num_negatives = None
        corrupt_side = NegativeSamplingCorruptionType.SRC
        scoring_function = SimilarityType.DOT

        logits, labels = against_batch_relationwise_contrastive_similarity(
            positive_src_embeddings=src_embeddings,
            positive_dst_embeddings=dst_embeddings,
            positive_condensed_edge_types=condensed_edge_types,
            negative_batch_src_embeddings=batch_src_embeddings,
            negative_batch_dst_embeddings=batch_dst_embeddings,
            batch_condensed_edge_types=batch_condensed_edge_types,
            temperature=temperature,
            scoring_function=scoring_function,
            corrupt_side=corrupt_side,
            num_negatives=num_negatives,
        )

        # Positive logits (src @ dst.T).diagonal(): [2., 3.]
        expected_positive_logits = torch.tensor([[2.0], [3.0]])

        # Negative logits (batch_src @ dst.T) where relations match (type 0):
        # Row 0: [1.5*2 + 0.1*0, 0*2 + 1*0] => [3.0, 0.0]
        # Row 1: [1.5*0 + 0.1*3, 0*0 + 1*3] => [0.3, 3.0]
        expected_neg_logits_masked = torch.tensor(
            [[3.0, 0.0, float("-inf")], [0.3, 3.0, float("-inf")]]
        )

        expected_logits = torch.cat(
            [expected_positive_logits, expected_neg_logits_masked], dim=1
        )
        expected_labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        self.assertTrue(torch.allclose(logits, expected_logits))
        self.assertTrue(torch.equal(labels, expected_labels))
        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(labels.shape, (2, 4))


if __name__ == "__main__":
    absltest.main()
