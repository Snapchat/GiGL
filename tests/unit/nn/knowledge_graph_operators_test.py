import unittest

import torch

from gigl.experimental.knowledge_graph_embedding.lib.model.operators import (
    ComplexDiagonalOperator,
    DiagonalOperator,
    IdentityOperator,
    LinearOperator,
    TranslationOperator,
)


class RelationwiseOperatorTest(unittest.TestCase):
    def test_operators_preserve_batch_embedding_shape(self) -> None:
        embeddings = torch.randn(3, 4)
        condensed_edge_types = torch.tensor([0, 1, 0])

        operators = (
            TranslationOperator(num_edge_types=2, node_emb_dim=4),
            DiagonalOperator(num_edge_types=2, node_emb_dim=4),
            ComplexDiagonalOperator(num_edge_types=2, node_emb_dim=4),
            LinearOperator(num_edge_types=2, node_emb_dim=4),
            IdentityOperator(num_edge_types=2, node_emb_dim=4),
        )

        for operator in operators:
            with self.subTest(operator=type(operator).__name__):
                output = operator(embeddings, condensed_edge_types)
                self.assertEqual(output.shape, embeddings.shape)

    def test_linear_operator_uses_edge_type_per_example(self) -> None:
        operator = LinearOperator(num_edge_types=2, node_emb_dim=2)
        with torch.no_grad():
            operator.edge_type_projection[0] = torch.eye(2)
            operator.edge_type_projection[1] = 2 * torch.eye(2)

        embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        output = operator(embeddings, torch.tensor([0, 1]))

        torch.testing.assert_close(
            output,
            torch.tensor([[1.0, 2.0], [6.0, 8.0]]),
        )


if __name__ == "__main__":
    unittest.main()
