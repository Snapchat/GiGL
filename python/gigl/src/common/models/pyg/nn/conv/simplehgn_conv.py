from typing import Optional

import torch
import torch_geometric
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class SimpleHGNConv(MessagePassing):
    """
    The SimpleHGN convolution layer based on https://arxiv.org/pdf/2112.14936

    Here, we adopt a form which includes support for edge-features in addition to node-features for attention calculation.
    This layer is based on the adaptation for link prediction tasks listed below Eq.14 in the paper.

    Args:
        in_channels (int): the input dimension of node features
        edge_in_channels (Optional[int]): the input dimension of edge features
        out_channels (int): the output dimension of node features
        edge_type_dim (int): the hidden dimension allocated to edge-type embeddings (per head)
        num_heads (int): the number of heads
        num_edge_types (int): the number of edge types
        dropout (float): the feature drop rate
        negative_slope (float): the negative slope used in the LeakyReLU
        should_use_node_residual (boolean): whether we need the node residual operation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int,
        edge_in_channels: Optional[int] = None,
        num_heads: int = 1,
        edge_type_dim: int = 16,
        should_use_node_residual: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.in_dim = in_channels
        self.out_dim = out_channels
        self.edge_in_dim = edge_in_channels
        self.edge_type_dim = edge_type_dim

        self.num_edge_types = num_edge_types
        self.num_heads = num_heads

        # Encodes embeddings for each edge-type.
        self.edge_type_emb = nn.Parameter(
            torch.empty(size=(self.num_edge_types, self.edge_type_dim))
        )

        # Multi-headed linear projection for edge-type embedding
        self.W_etype = torch_geometric.nn.HeteroLinear(
            self.edge_type_dim, self.edge_type_dim * self.num_heads, self.num_edge_types
        )

        # Linear projection for node features (for each head)
        self.W_nfeat = nn.Parameter(
            torch.FloatTensor(self.in_dim, self.out_dim * self.num_heads)
        )

        if self.edge_in_dim:
            # Linear projection for edge features (for each head)
            self.W_efeat = nn.Parameter(
                torch.FloatTensor(self.edge_in_dim, self.edge_in_dim * self.num_heads)
            )
            # Attention weights for edge features
            self.a_efeat = nn.Parameter(
                torch.empty(size=(1, self.num_heads, self.edge_in_dim))
            )
            # Dropout for edge features
            self.efeat_drop = nn.Dropout(dropout)

        self.a_l = nn.Parameter(torch.empty(size=(1, self.num_heads, self.out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, self.num_heads, self.out_dim)))
        self.a_etype = nn.Parameter(
            torch.empty(size=(1, self.num_heads, self.edge_type_dim))
        )

        self.nfeat_drop = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope)

        if should_use_node_residual:
            self.residual = nn.Linear(self.in_dim, self.out_dim * self.num_heads)
        else:
            self.register_buffer("residual", None)

        self.reset_parameters()

    def reset_parameters(self):
        for param in [
            self.edge_type_emb,
            self.W_nfeat,
            self.a_l,
            self.a_r,
            self.a_etype,
        ]:
            nn.init.xavier_uniform_(param, gain=1.414)
        if self.edge_in_dim:
            for param in [self.W_efeat, self.a_efeat]:
                nn.init.xavier_uniform_(param, gain=1.414)
        self.residual.reset_parameters()
        self.W_etype.reset_parameters()

    def forward(
        self,
        edge_index: torch.LongTensor,
        node_feat: torch.FloatTensor,
        edge_type: torch.LongTensor,
        edge_feat: Optional[torch.FloatTensor] = None,
    ):
        # edge_index shape: [2, num_edges]
        # node_feat shape: [num_nodes, in_dim]
        # edge_feat shape: None | [num_edges, edge_in_dim]
        # edge_type shape: [num_edges]

        # For each head, project node features to out_dim and correct NaNs.
        # Output shape: [num_nodes, num_heads, out_dim]
        node_emb = self.nfeat_drop(node_feat)
        node_emb = torch.matmul(node_emb, self.W_nfeat).view(
            -1, self.num_heads, self.out_dim
        )
        node_emb[torch.isnan(node_emb)] = 0.0

        # For each head, project edge features to out_dim and correct NaNs.
        # Output shape: [num_edges, num_heads, edge_in_dim]
        if edge_feat is not None and self.edge_in_dim is not None:
            edge_emb = self.efeat_drop(edge_feat)
            edge_emb = torch.matmul(edge_emb, self.W_efeat).view(
                -1, self.num_heads, self.edge_in_dim
            )
            edge_emb[torch.isnan(edge_emb)] = 0.0

        # For each edge type, get an embedding of dimension edge_type_dim for each head
        # Output shape: [num_edges, num_heads, edge_type_dim]
        edge_type_emb = self.W_etype(self.edge_type_emb[edge_type], edge_type).view(
            -1, self.num_heads, self.edge_type_dim
        )

        # Compute the attention scores (alpha) for all heads
        # Output shape: [num_edges, num_heads]
        row, col = edge_index[0, :], edge_index[1, :]
        h_l_term = (self.a_l * node_emb).sum(dim=-1)[row]
        h_r_term = (self.a_r * node_emb).sum(dim=-1)[col]
        h_etype_term = (self.a_etype * edge_type_emb).sum(dim=-1)

        h_efeat_term = (
            0
            if edge_feat is None or self.edge_in_dim is None
            else (self.a_efeat * edge_emb).sum(dim=-1)
        )
        alpha = self.leakyrelu(h_l_term + h_r_term + h_etype_term + h_efeat_term)
        alpha = softmax(alpha, row)

        # Propagate messages
        # Output shape: [num_nodes, num_heads, out_dim]
        out = self.propagate(
            edge_index, node_emb=node_emb, node_feat=node_feat, alpha=alpha
        )

        # Concatenate embeddings across heads
        # Output shape: [num_nodes, num_heads * out_dim]
        out = out.view(-1, self.num_heads * self.out_dim)

        # Add node residual
        # Output shape: [num_nodes, num_heads * out_dim]
        if self.residual:
            out += self.residual(node_feat)

        return out

    def message(self, node_emb_j, alpha):
        # Multiply embeddings for each head with attention scores
        # node_emb_j is shape [num_edges, num_heads, out_dim]
        # alpha is shape [num_edges, num_heads]
        return alpha.unsqueeze(-1) * node_emb_j
