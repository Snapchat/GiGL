import torch
from absl.testing import absltest

from gigl.experimental.knowledge_graph_embedding.lib.model.negative_sampling import (
    in_batch_relationwise_contrastive_similarity,
)
from gigl.experimental.knowledge_graph_embedding.lib.model.types import (
    NegativeSamplingCorruptionType,
    SimilarityType,
)
from tests.test_assets.test_case import TestCase


class TestInBatchRelationwiseContrastiveSimilarity(TestCase):
    def test_correctness_dot_dst_one_negative(self):
        # Test dot product similarity with destination-side corruption and one negative sample.
        src_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dst_embeddings = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        condensed_edge_types = torch.tensor([0, 0])
        temperature = 1.0
        num_negatives = 1
        corrupt_side = NegativeSamplingCorruptionType.DST
        scoring_function = SimilarityType.DOT

        logits, labels = in_batch_relationwise_contrastive_similarity(
            src_embeddings=src_embeddings,
            dst_embeddings=dst_embeddings,
            condensed_edge_types=condensed_edge_types,
            temperature=temperature,
            scoring_function=scoring_function,
            corrupt_side=corrupt_side,
            num_negatives=num_negatives,
        )

        # Expected similarity matrix (src @ dst.T): [[2., 0.], [0., 3.]]
        # Positive logits (diagonal): [2., 3.]
        # Negative logits (other elements in the row, same relation): For the first row, the other is 0. For the second, it's 0.
        expected_logits = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
        expected_labels = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        self.assertTrue(torch.allclose(logits, expected_logits))
        self.assertTrue(torch.allclose(labels, expected_labels))

    def test_correctness_cosine_src_one_negative(self):
        # Test cosine similarity with source-side corruption and one negative sample.
        src_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dst_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        condensed_edge_types = torch.tensor([0, 0])
        temperature = 1.0
        num_negatives = 1
        corrupt_side = NegativeSamplingCorruptionType.SRC
        scoring_function = SimilarityType.COSINE

        logits, labels = in_batch_relationwise_contrastive_similarity(
            src_embeddings=src_embeddings,
            dst_embeddings=dst_embeddings,
            condensed_edge_types=condensed_edge_types,
            temperature=temperature,
            scoring_function=scoring_function,
            corrupt_side=corrupt_side,
            num_negatives=num_negatives,
        )

        # Normalized embeddings are the same
        # Similarity matrix (normalized dst @ normalized src.T): [[1., 0.], [0., 1.]]
        # Positive logits: [1., 1.]
        # Negative logits: [0., 0.]
        expected_logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        expected_labels = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        self.assertTrue(torch.allclose(logits, expected_logits))
        self.assertTrue(torch.allclose(labels, expected_labels))

    def test_correctness_euclidean_both_two_negatives(self):
        # Test Euclidean similarity with both-side corruption and two negative samples.
        # Note: Due to the 'both' corruption's randomness, we focus on the positive logit and label.
        torch.manual_seed(0)  # For deterministic behavior of 'both' corruption
        src_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        dst_embeddings = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        condensed_edge_types = torch.tensor([0, 0, 1])
        temperature = 1.0
        num_negatives = 2
        corrupt_side = NegativeSamplingCorruptionType.BOTH
        scoring_function = SimilarityType.EUCLIDEAN

        logits, labels = in_batch_relationwise_contrastive_similarity(
            src_embeddings=src_embeddings,
            dst_embeddings=dst_embeddings,
            condensed_edge_types=condensed_edge_types,
            temperature=temperature,
            scoring_function=scoring_function,
            corrupt_side=corrupt_side,
            num_negatives=num_negatives,
        )

        # Squared Euclidean distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a @ b
        # Similarity is the negative of this.

        # Positive similarities:
        # For 0: -((1^2 + 0^2) + (0^2 + 1^2) - 2 * (1*0 + 0*1)) = -2
        # For 1: -((0^2 + 1^2) + (1^2 + 0^2) - 2 * (0*1 + 1*0)) = -2
        # For 2: -((1^2 + 1^2) + (0^2 + 0^2) - 2 * (1*0 + 1*0)) = -2
        expected_positive_logits = torch.tensor([-2.0, -2.0, -2.0]).unsqueeze(1)
        expected_positive_labels = torch.ones(3, 1)

        self.assertTrue(
            torch.allclose(logits[:, 0].unsqueeze(1), expected_positive_logits)
        )
        self.assertTrue(
            torch.allclose(labels[:, 0].unsqueeze(1), expected_positive_labels)
        )
        self.assertEqual(logits.shape, (3, 3))
        self.assertEqual(labels.shape, (3, 3))
        self.assertTrue(torch.all(labels[:, 0] == 1.0))
        self.assertTrue(torch.all(labels[:, 1:] == 0.0))

    def test_correctness_dot_dst_none_negatives(self):
        # Test dot product with destination-side corruption and "full" negative sampling (all in-batch negatives used).
        src_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dst_embeddings = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        condensed_edge_types = torch.tensor([0, 0])
        temperature = 1.0
        num_negatives = None
        corrupt_side = NegativeSamplingCorruptionType.DST
        scoring_function = SimilarityType.DOT

        logits, labels = in_batch_relationwise_contrastive_similarity(
            src_embeddings=src_embeddings,
            dst_embeddings=dst_embeddings,
            condensed_edge_types=condensed_edge_types,
            temperature=temperature,
            scoring_function=scoring_function,
            corrupt_side=corrupt_side,
            num_negatives=num_negatives,
        )

        # Similarity matrix: [[2., 0.], [0., 3.]]
        # Positive logits: [2., 3.]
        # Negative logits (same relation): [[-inf, 0.], [0., -inf]]
        # The negative logits are set to -inf for the non-positive examples.
        expected_logits = torch.tensor(
            [[2.0, -float("inf"), 0.0], [3.0, 0.0, -float("inf")]]
        )
        expected_labels = torch.tensor(
            [[1, 0, 0], [1, 0, 0]]
        )  # Indices of the positive example in the logits

        self.assertTrue(torch.allclose(logits, expected_logits))
        self.assertTrue(torch.equal(labels, expected_labels))
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(labels.shape, (2, 3))

    def test_correctness_num_negatives_zero(self):
        # Test the case where num_negatives is 0, expecting only positive logits and labels as 1.
        src_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dst_embeddings = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        condensed_edge_types = torch.tensor([0, 0])
        temperature = 1.0
        num_negatives = 0
        scoring_function = SimilarityType.DOT

        logits, labels = in_batch_relationwise_contrastive_similarity(
            src_embeddings=src_embeddings,
            dst_embeddings=dst_embeddings,
            condensed_edge_types=condensed_edge_types,
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


if __name__ == "__main__":
    absltest.main()
