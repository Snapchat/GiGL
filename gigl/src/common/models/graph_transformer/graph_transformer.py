"""Graph Transformer encoder for heterogeneous graphs.

Adapted from RelGT's LocalModule (https://github.com/snap-stanford/relgt).
Converts heterogeneous graph data into fixed-length sequences via
``heterodata_to_graph_transformer_input``, processes through a stack of pre-norm
transformer encoder layers, then produces per-node embeddings via
attention-weighted neighbor readout.

Conforms to the same forward interface as ``HGT`` and ``SimpleHGN`` in
``gigl.src.common.models.pyg.heterogeneous``, making it a drop-in
replacement as the encoder in ``LinkPredictionGNN``.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.hetero_data
from torch import Tensor

from gigl.src.common.models.layers.normalization import l2_normalize_embeddings
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.transforms.graph_transformer import heterodata_to_graph_transformer_input


class FeedForwardNetwork(nn.Module):
    """Two-layer feed-forward network with BatchNorm and GELU activation.

    Applies BatchNorm1d across the sequence dimension using permute.
    Adapted from RelGT's FeedForwardNetwork.

    Args:
        hidden_dim: Hidden (and output) dimension of the FFN.
        feedforward_dim: Inner dimension of the two-layer MLP.
        dropout_rate: Dropout probability applied after each linear layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        feedforward_dim: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self._batch_norm_in = nn.BatchNorm1d(hidden_dim)
        self._batch_norm_out = nn.BatchNorm1d(hidden_dim)
        self._ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters."""
        self._batch_norm_in.reset_parameters()
        self._batch_norm_out.reset_parameters()
        for layer in self._ffn:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq, hidden_dim)``.

        Returns:
            Output tensor of shape ``(batch, seq, hidden_dim)``.
        """
        # BatchNorm1d expects (batch, channels, seq), so permute
        x = x.permute(0, 2, 1)
        x = self._batch_norm_in(x)
        x = x.permute(0, 2, 1)

        x = self._ffn(x)

        x = x.permute(0, 2, 1)
        x = self._batch_norm_out(x)
        x = x.permute(0, 2, 1)

        return x


class GraphTransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with multi-head self-attention.

    Uses ``F.scaled_dot_product_attention`` which automatically selects the
    most efficient attention implementation (flash, memory-efficient, or
    math-based) based on input properties and hardware.

    Adapted from RelGT's EncoderLayer.

    Args:
        hidden_dim: Model dimension (d_model).
        num_heads: Number of attention heads. Must evenly divide hidden_dim.
        feedforward_dim: Inner dimension of the feed-forward network.
        dropout_rate: Dropout probability for feed-forward layers.
        attention_dropout_rate: Dropout probability for attention weights.

    Raises:
        ValueError: If hidden_dim is not divisible by num_heads.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self._num_heads = num_heads
        self._head_dim = hidden_dim // num_heads
        self._attention_dropout_rate = attention_dropout_rate

        self._attention_norm = nn.LayerNorm(hidden_dim)
        self._query_projection = nn.Linear(hidden_dim, hidden_dim)
        self._key_projection = nn.Linear(hidden_dim, hidden_dim)
        self._value_projection = nn.Linear(hidden_dim, hidden_dim)
        self._output_projection = nn.Linear(hidden_dim, hidden_dim)
        self._attention_dropout = nn.Dropout(dropout_rate)

        self._ffn_norm = nn.LayerNorm(hidden_dim)
        self._ffn = FeedForwardNetwork(hidden_dim, feedforward_dim, dropout_rate)

    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters."""
        self._attention_norm.reset_parameters()
        for projection in [
            self._query_projection,
            self._key_projection,
            self._value_projection,
            self._output_projection,
        ]:
            nn.init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
        self._ffn_norm.reset_parameters()
        self._ffn.reset_parameters()

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq, hidden_dim)``.
            attn_bias: Optional attention bias of shape
                ``(batch, num_heads, seq, seq)`` or broadcastable.
                Added as an additive mask to attention scores.

        Returns:
            Output tensor of shape ``(batch, seq, hidden_dim)``.
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Self-attention block (pre-norm)
        residual = x
        x_norm = self._attention_norm(x)

        query = self._query_projection(x_norm)
        key = self._key_projection(x_norm)
        value = self._value_projection(x_norm)

        # Reshape to (batch, num_heads, seq, head_dim)
        query = query.view(
            batch_size, seq_len, self._num_heads, self._head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, seq_len, self._num_heads, self._head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_size, seq_len, self._num_heads, self._head_dim
        ).transpose(1, 2)

        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_bias,
            dropout_p=self._attention_dropout_rate if self.training else 0.0,
            is_causal=False,
        )

        # Reshape back to (batch, seq, hidden_dim)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, seq_len, hidden_dim
        )
        attention_output = self._output_projection(attention_output)
        attention_output = self._attention_dropout(attention_output)

        x = residual + attention_output

        # Feed-forward block (pre-norm)
        residual = x
        x_norm = self._ffn_norm(x)
        ffn_output = self._ffn(x_norm)
        x = residual + ffn_output

        return x


class GraphTransformerEncoder(nn.Module):
    """Graph Transformer encoder for heterogeneous graphs.

    Converts heterogeneous graph data into fixed-length sequences via
    ``heterodata_to_graph_transformer_input``, processes through pre-norm
    transformer encoder layers, and produces per-node embeddings via
    attention-weighted neighbor readout (from RelGT's LocalModule).

    Conforms to the same forward interface as ``HGT`` and ``SimpleHGN``,
    making it a drop-in encoder for ``LinkPredictionGNN``.

    Args:
        node_type_to_feat_dim_map: Dictionary mapping node types to their
            input feature dimensions.
        edge_type_to_feat_dim_map: Dictionary mapping edge types to their
            feature dimensions. Accepted for interface conformance with
            ``HGT``/``SimpleHGN``; edge features are not used by the
            graph transformer.
        hid_dim: Hidden dimension for transformer layers. All node types
            are projected to this dimension before processing.
        out_dim: Output embedding dimension.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer. Must evenly divide
            ``hid_dim``.
        max_seq_len: Maximum sequence length for the graph-to-sequence
            transform. Neighborhoods are truncated to this length.
        hop_distance: Number of hops for neighborhood extraction in the
            graph-to-sequence transform.
        dropout_rate: Dropout probability for feed-forward layers.
        attention_dropout_rate: Dropout probability for attention weights.
        should_l2_normalize_embedding_layer_output: Whether to L2 normalize
            output embeddings.
        pe_attr_names: List of positional encoding attribute names to concatenate
            to node features. These should be node-level attributes stored by
            transforms like ``AddHeteroRandomWalkPE`` or ``AddHeteroRandomWalkSE``.
            Example: ``['random_walk_pe', 'random_walk_se']``.
            If None, no positional encodings are attached. (default: None)
        anchor_based_pe_attr_names: List of graph-level attribute names containing
            sparse (N x N) matrices for anchor-based positional encodings. For each
            node in the sequence, the value ``PE[anchor_idx, node_idx]`` is looked up
            and concatenated to node features.
            Example: ``['hop_distance']`` (from ``AddHeteroHopDistanceEncoding``).
            If None, no anchor-based PEs are attached. (default: None)
        feature_embedding_layer_dict: Optional ModuleDict mapping node types to
            feature embedding layers. If provided, these are applied to node
            features before node projection. (default: None)

    Example:
        >>> from gigl.src.common.models.graph_transformer.graph_transformer import (
        ...     GraphTransformerEncoder,
        ... )
        >>> encoder = GraphTransformerEncoder(
        ...     node_type_to_feat_dim_map={NodeType("user"): 64, NodeType("item"): 32},
        ...     edge_type_to_feat_dim_map={},
        ...     hid_dim=128,
        ...     out_dim=64,
        ...     num_layers=2,
        ...     num_heads=4,
        ... )
        >>> embeddings = encoder(data, output_node_types=[NodeType("user")], device=device)
    """

    def __init__(
        self,
        node_type_to_feat_dim_map: dict[NodeType, int],
        edge_type_to_feat_dim_map: dict[EdgeType, int],
        hid_dim: int,
        out_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        max_seq_len: int = 128,
        hop_distance: int = 2,
        dropout_rate: float = 0.3,
        attention_dropout_rate: float = 0.0,
        should_l2_normalize_embedding_layer_output: bool = False,
        pe_attr_names: Optional[list[str]] = None,
        anchor_based_pe_attr_names: Optional[list[str]] = None,
        feature_embedding_layer_dict: Optional[nn.ModuleDict] = None,
        **kwargs: object,
    ) -> None:
        super().__init__()

        self._hid_dim = hid_dim
        self._out_dim = out_dim
        self._max_seq_len = max_seq_len
        self._hop_distance = hop_distance
        self._should_l2_normalize_embedding_layer_output = (
            should_l2_normalize_embedding_layer_output
        )
        self._pe_attr_names = pe_attr_names
        self._anchor_based_pe_attr_names = anchor_based_pe_attr_names
        self._feature_embedding_layer_dict = feature_embedding_layer_dict

        # Per-node-type input projection to hid_dim (like HGT's lin_dict)
        self._node_projection_dict = nn.ModuleDict(
            {
                str(node_type): nn.Linear(feat_dim, hid_dim)
                for node_type, feat_dim in node_type_to_feat_dim_map.items()
            }
        )

        # Transformer encoder layers
        feedforward_dim = 2 * hid_dim
        self._encoder_layers = nn.ModuleList(
            [
                GraphTransformerEncoderLayer(
                    hidden_dim=hid_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        self._final_norm = nn.LayerNorm(hid_dim)

        # Readout attention: projects concatenated (anchor, neighbor) to score
        self._readout_attention = nn.Linear(2 * hid_dim, 1)

        # Output projection: hid_dim -> out_dim
        self._output_projection = nn.Linear(hid_dim, out_dim)

    def forward(
        self,
        data: torch_geometric.data.hetero_data.HeteroData,
        output_node_types: Optional[list[NodeType]] = None,
        device: Optional[torch.device] = None,
    ) -> dict[NodeType, torch.Tensor]:
        """Run the forward pass of the Graph Transformer encoder.

        Args:
            data: Input HeteroData object with node features (``x_dict``)
                and edge indices (``edge_index_dict``).
            output_node_types: List of node types for which to return output
                embeddings. If None, uses all node types in data.
            device: Torch device for output tensors. If None, inferred from data.

        Returns:
            Dictionary mapping each requested node type to its output
            embeddings of shape ``(num_seed_nodes, out_dim)``. When
            ``data`` comes from neighbor sampling, only seed/anchor node
            embeddings are produced (not sampled neighbors).
        """
        # Infer device from data if not provided
        if device is None:
            device = next(iter(data.x_dict.values())).device

        # Use all node types if not specified
        if output_node_types is None:
            output_node_types = list(data.node_types)

        # 0. Apply feature embedding if provided
        if self._feature_embedding_layer_dict is not None:
            for node_type, emb_layer in self._feature_embedding_layer_dict.items():
                data[node_type].x = emb_layer(data[node_type].x)

        # 1. Project all node features to hid_dim
        projected_x_dict = {
            node_type: self._node_projection_dict[str(node_type)](x.to(device))
            for node_type, x in data.x_dict.items()
        }

        # 2. Build projected HeteroData, preserving batch_size metadata
        # from neighbor sampling (first batch_size nodes are seed/anchor nodes).
        init_dict: dict = {
            edge_type: {"edge_index": data.edge_index_dict[edge_type]}
            for edge_type in data.edge_index_dict.keys()
        }
        init_dict.update(
            {node_type: {"x": x} for node_type, x in projected_x_dict.items()}
        )
        projected_data = torch_geometric.data.hetero_data.HeteroData(init_dict)

        # Preserve batch_size metadata so we only build sequences for seed nodes.
        for nt in data.node_types:
            if hasattr(data[nt], "batch_size"):
                projected_data[nt].batch_size = data[nt].batch_size

        # 3. For each output node type, run transform + transformer
        node_typed_embeddings: dict[NodeType, torch.Tensor] = {}

        for node_type in output_node_types:
            if node_type not in projected_data.node_types:
                node_typed_embeddings[node_type] = torch.FloatTensor([]).to(
                    device=device
                )
                continue

            # Use seed batch_size (from neighbor sampling) when available;
            # fall back to num_nodes for non-sampled data (e.g., full-batch).
            num_seed_nodes = getattr(
                projected_data[node_type], "batch_size", projected_data[node_type].num_nodes
            )
            sequences = heterodata_to_graph_transformer_input(
                data=projected_data,
                batch_size=num_seed_nodes,
                max_seq_len=self._max_seq_len,
                anchor_node_type=node_type,
                hop_distance=self._hop_distance,
                pe_attr_names=self._pe_attr_names,
                anchor_based_pe_attr_names=self._anchor_based_pe_attr_names,
            )

            embeddings = self._encode_and_readout(sequences)
            node_typed_embeddings[node_type] = self._output_projection(embeddings)

        if self._should_l2_normalize_embedding_layer_output:
            node_typed_embeddings = l2_normalize_embeddings(  # type: ignore[assignment]
                node_typed_embeddings=node_typed_embeddings
            )

        return node_typed_embeddings

    def _encode_and_readout(self, sequences: Tensor) -> Tensor:
        """Process sequences through transformer layers and attention readout.

        Args:
            sequences: Input tensor of shape ``(batch_size, max_seq_len, hid_dim)``.

        Returns:
            Output embeddings of shape ``(batch_size, hid_dim)``.
        """
        x = sequences

        for encoder_layer in self._encoder_layers:
            x = encoder_layer(x)

        x = self._final_norm(x)

        # Readout: anchor (position 0) + attention-weighted neighbor aggregation
        anchor = x[:, 0, :].unsqueeze(1)  # (batch, 1, hid_dim)
        neighbors = x[:, 1:, :]  # (batch, seq-1, hid_dim)
        seq_minus_one = neighbors.size(1)

        if seq_minus_one == 0:
            return anchor.squeeze(1)

        # Expand anchor to match neighbor dimension for concatenation
        anchor_expanded = anchor.expand(-1, seq_minus_one, -1)

        # Compute attention scores over neighbors
        readout_scores = self._readout_attention(
            torch.cat([anchor_expanded, neighbors], dim=-1)
        )  # (batch, seq-1, 1)
        readout_weights = F.softmax(readout_scores, dim=1)  # (batch, seq-1, 1)

        neighbor_aggregation = (neighbors * readout_weights).sum(
            dim=1, keepdim=True
        )  # (batch, 1, hid_dim)

        output = (anchor + neighbor_aggregation).squeeze(1)  # (batch, hid_dim)

        return output
