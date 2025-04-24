import unittest

import torch
import torch.nn.functional as F

from gigl.src.common.models.layers.loss import RetrievalLoss


class RetrievalLossTest(unittest.TestCase):
    query_embeddings = torch.tensor([])
    positive_embeddings = torch.tensor([])
    random_neg_embeddings = torch.tensor([])
    candidate_embeddings = torch.tensor([])
    candidate_ids = torch.tensor([], dtype=torch.int64)
    query_ids = torch.tensor([], dtype=torch.int64)

    @classmethod
    def setUpClass(cls):
        cls.query_embeddings = F.normalize(
            torch.tensor(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.9, 1.1],
                    [0.0, 0.0, 0.9, 1.1],
                ]
            ),
            p=2,
            dim=1,
        )
        cls.positive_embeddings = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.2, 0.0, 0.0],
                    [0.3, 1.0, 0.2, 0.0],
                    [0.0, 0.0, 1.0, 0.4],
                    [0.0, 0.0, 0.4, 0.9],
                ]
            ),
            p=2,
            dim=1,
        )
        cls.random_neg_embeddings = F.normalize(
            torch.tensor(
                [
                    [0.21, 0.22, 0.23, 0.24],
                    [0.24, 0.23, 0.22, 0.21],
                ]
            ),
            p=2,
            dim=1,
        )
        cls.candidate_embeddings = torch.concat(
            [cls.positive_embeddings, cls.random_neg_embeddings], dim=0
        )
        # one random negative collides with a positive
        cls.candidate_ids = torch.tensor([1, 2, 3, 4, 1, 5], dtype=torch.int64)
        # each anchor node has two positives
        cls.query_ids = torch.tensor([11, 11, 12, 12], dtype=torch.int64)

    def test_mask_by_query_ids(self):
        loss = RetrievalLoss()
        actual = loss._mask_by_query_ids(
            self.query_ids,
            self.query_ids.shape[0],
            self.candidate_ids.shape[0],
            torch.float32,
        )
        expected = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(expected, actual))

    def test_mask_by_candidate_ids(self):
        loss = RetrievalLoss()
        actual = loss._mask_by_candidate_ids(
            self.candidate_ids, self.query_ids.shape[0], torch.float32
        )
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(expected, actual))

    def test_loss_value(self):
        expected_labels = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        min_float = torch.finfo(torch.float).min
        loss = RetrievalLoss(remove_accidental_hits=False)
        scores = torch.mm(self.query_embeddings, self.candidate_embeddings.T)
        actual1 = loss.calculate_batch_retrieval_loss(scores)
        expected1 = torch.tensor(
            [
                [0.8321, 0.8647, 0.0000, 0.0000, 0.6748, 0.7376],
                [0.8321, 0.8647, 0.0000, 0.0000, 0.6748, 0.7376],
                [0.0000, 0.1191, 0.8754, 0.9644, 0.7355, 0.6699],
                [0.0000, 0.1191, 0.8754, 0.9644, 0.7355, 0.6699],
            ]
        )

        self.assertTrue(
            torch.isclose(
                F.cross_entropy(expected1, expected_labels, reduction="sum"),
                actual1,
                atol=1e-6,
            )
        )

        loss = RetrievalLoss(remove_accidental_hits=True)
        scores = torch.mm(self.query_embeddings, self.candidate_embeddings.T)
        actual2 = loss.calculate_batch_retrieval_loss(
            scores,
            candidate_ids=self.candidate_ids,
        )
        expected2 = torch.tensor(
            [
                [0.8321, 0.8647, 0.0000, 0.0000, min_float, 0.7376],
                [0.8321, 0.8647, 0.0000, 0.0000, 0.6748, 0.7376],
                [0.0000, 0.1191, 0.8754, 0.9644, 0.7355, 0.6699],
                [0.0000, 0.1191, 0.8754, 0.9644, 0.7355, 0.6699],
            ]
        )
        self.assertTrue(
            torch.isclose(
                F.cross_entropy(expected2, expected_labels, reduction="sum"),
                actual2,
                atol=1e-6,
            )
        )

        # by filtering out other positives from the same query, we should see smaller loss
        actual3 = loss.calculate_batch_retrieval_loss(
            scores,
            candidate_ids=self.candidate_ids,
            query_ids=self.query_ids,
        )
        expected3 = torch.tensor(
            [
                [0.8321, min_float, 0.0000, 0.0000, min_float, 0.7376],
                [min_float, 0.8647, 0.0000, 0.0000, 0.6748, 0.7376],
                [0.0000, 0.1191, 0.8754, min_float, 0.7355, 0.6699],
                [0.0000, 0.1191, min_float, 0.9644, 0.7355, 0.6699],
            ]
        )
        self.assertTrue(
            torch.isclose(
                F.cross_entropy(expected3, expected_labels, reduction="sum"),
                actual3,
                atol=1e-6,
            )
        )
