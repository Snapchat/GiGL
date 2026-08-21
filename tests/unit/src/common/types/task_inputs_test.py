import torch

from gigl.src.common.types.graph_data import CondensedEdgeType, CondensedNodeType
from gigl.src.common.types.task_inputs import (
    BatchCombinedScores,
    BatchEmbeddings,
    BatchScores,
)
from tests.test_assets.test_case import TestCase


class TaskInputsTest(TestCase):
    def test_empty_task_inputs_retain_matrix_rank(self) -> None:
        condensed_edge_type = CondensedEdgeType(0)
        condensed_node_type = CondensedNodeType(0)

        batch_embeddings = BatchEmbeddings(
            query_embeddings=torch.FloatTensor(0, 8),
            repeated_query_embeddings={condensed_edge_type: torch.FloatTensor(0, 8)},
            pos_embeddings={condensed_edge_type: torch.FloatTensor(0, 8)},
            hard_neg_embeddings={condensed_edge_type: torch.FloatTensor(0, 8)},
            random_neg_embeddings={condensed_node_type: torch.FloatTensor(0, 8)},
        )
        batch_scores = BatchScores(
            pos_scores=torch.FloatTensor(1, 0),
            hard_neg_scores=torch.FloatTensor(1, 0),
            random_neg_scores=torch.FloatTensor(1, 0),
        )
        combined_scores = BatchCombinedScores(
            repeated_candidate_scores=torch.FloatTensor(0, 0),
            positive_ids=torch.LongTensor([]),
            hard_neg_ids=torch.LongTensor([]),
            random_neg_ids=torch.LongTensor([]),
            repeated_query_ids=torch.LongTensor([]),
            num_unique_query_ids=0,
        )

        self.assertEqual(batch_embeddings.query_embeddings.shape, (0, 8))
        self.assertEqual(batch_scores.pos_scores.shape, (1, 0))
        self.assertEqual(combined_scores.repeated_candidate_scores.shape, (0, 0))
