import torch
import torch.nn as nn


class RelationwiseOperatorBase(nn.Module):
    """
    Base class for relationwise operators in heterogeneous graph embeddings.
    Each operator applies a transformation to the node embeddings based on
    the context of a specific relation / edge-type.
    """

    def __init__(self, num_edge_types: int, node_emb_dim: int):
        super().__init__()

    def forward(
        self, embeddings: torch.Tensor, condensed_edge_types: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class TranslationOperator(RelationwiseOperatorBase):
    """
    A translation operator for heterogeneous graph embeddings.

    This operator adds the edge type embeddings to the node embeddings.
    It is used to model the relationship between nodes in a heterogeneous graph.
    The edge type embeddings are learned during training and are used to
    represent the different types of relationships between nodes.

    See https://papers.nips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf
    """

    def __init__(self, num_edge_types: int, node_emb_dim: int):
        super().__init__(num_edge_types=num_edge_types, node_emb_dim=node_emb_dim)
        self.edge_type_embeddings = nn.Embedding(
            num_embeddings=num_edge_types,
            embedding_dim=node_emb_dim,
        )

    def forward(self, embeddings: torch.Tensor, condensed_edge_types: torch.Tensor):
        edge_type_embeddings = self.edge_type_embeddings(condensed_edge_types)
        return embeddings + edge_type_embeddings


class DiagonalOperator(RelationwiseOperatorBase):
    """
    A diagonal operator for heterogeneous graph embeddings.

    This operator multiplies the node embeddings by the edge type embeddings.
    """

    def __init__(self, num_edge_types: int, node_emb_dim: int):
        super().__init__(num_edge_types=num_edge_types, node_emb_dim=node_emb_dim)
        self.edge_type_embeddings = nn.Embedding(
            num_embeddings=num_edge_types,
            embedding_dim=node_emb_dim,
        )

    def forward(self, embeddings: torch.Tensor, condensed_edge_types: torch.Tensor):
        edge_type_embeddings = self.edge_type_embeddings(condensed_edge_types)
        return embeddings * edge_type_embeddings


class ComplexDiagonalOperator(RelationwiseOperatorBase):
    """
    A complex diagonal operator for heterogeneous graph embeddings.

    This operator splits the node embeddings into real and imaginary parts,
    and then applies a diagonal operator to each part separately.

    The edge type embeddings are also split into real and imaginary parts.

    See https://proceedings.mlr.press/v48/trouillon16.pdf.
    """

    def __init__(self, num_edge_types: int, node_emb_dim: int):
        super().__init__(num_edge_types=num_edge_types, node_emb_dim=node_emb_dim)
        if node_emb_dim % 2 != 0:
            raise ValueError("Complex embeddings require an even embedding dimension.")
        self.edge_type_embeddings = nn.Embedding(
            num_embeddings=num_edge_types,
            embedding_dim=node_emb_dim,
        )

    def real_part(self, embeddings: torch.Tensor):
        return embeddings[:, : embeddings.shape[1] // 2]

    def imag_part(self, embeddings: torch.Tensor):
        return embeddings[:, embeddings.shape[1] // 2 :]

    def forward(self, embeddings: torch.Tensor, condensed_edge_types: torch.Tensor):
        edge_type_embeddings = self.edge_type_embeddings(condensed_edge_types)
        # Split the embeddings into real and imaginary parts

        src_embeddings_real = self.real_part(embeddings)
        src_embeddings_imag = self.imag_part(embeddings)
        edge_type_embeddings_real = self.real_part(edge_type_embeddings)
        edge_type_embeddings_imag = self.imag_part(edge_type_embeddings)

        # Apply the complex diagonal operator
        # Following eq10 here: https://proceedings.mlr.press/v48/trouillon16.pdf
        first = (
            edge_type_embeddings_real * src_embeddings_real
            - edge_type_embeddings_imag * src_embeddings_imag
        )
        second = (
            edge_type_embeddings_real * src_embeddings_imag
            + edge_type_embeddings_imag * src_embeddings_real
        )
        return torch.cat((first, second), dim=1)


class LinearOperator(RelationwiseOperatorBase):
    """
    A linear operator for heterogeneous graph embeddings.

    This operator projects the node embeddings using a learned projection matrix
    for each edge type. The projection matrix is learned during training and
    is used to represent the different types of relationships between nodes.
    """

    def __init__(self, num_edge_types: int, node_emb_dim: int):
        super().__init__(num_edge_types=num_edge_types, node_emb_dim=node_emb_dim)
        self.edge_type_projection = nn.Parameter(
            torch.empty(num_edge_types, node_emb_dim, node_emb_dim),
        )
        nn.init.xavier_normal_(self.edge_type_projection)

    def forward(self, embeddings: torch.Tensor, condensed_edge_types: torch.Tensor):
        return (
            embeddings @ self.edge_type_projection
        )  # [num_edge_types, batch_size, node_emb_dim]


class IdentityOperator(RelationwiseOperatorBase):
    """
    An identity operator for heterogeneous graph embeddings.

    This operator does not apply any transformation to the node embeddings.
    It is used when no relation operator is needed.
    """

    def forward(self, embeddings: torch.Tensor, condensed_edge_types: torch.Tensor):
        return embeddings
