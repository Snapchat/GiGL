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

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.hetero_data
from torch import Tensor

from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.transforms.graph_transformer import heterodata_to_graph_transformer_input


def _get_node_type_positional_encodings(
    data: torch_geometric.data.hetero_data.HeteroData,
    node_type: NodeType,
    pe_attr_names: list[str],
    device: torch.device,
) -> Tensor:
    """Collect concatenated node-level PE for a single node type."""
    pe_parts = []
    sorted_node_types = sorted(data.node_types)
    node_store = data[node_type]

    for attr_name in pe_attr_names:
        if hasattr(node_store, attr_name):
            pe_parts.append(getattr(node_store, attr_name).to(device))
            continue

        attr_dim = None
        for other_node_type in sorted_node_types:
            other_store = data[other_node_type]
            if hasattr(other_store, attr_name):
                attr_dim = getattr(other_store, attr_name).size(-1)
                break
        if attr_dim is None:
            raise ValueError(
                f"Positional encoding '{attr_name}' not found in any node type."
            )
        pe_parts.append(torch.zeros(node_store.num_nodes, attr_dim, device=device))

    return torch.cat(pe_parts, dim=-1)


class FeedForwardNetwork(nn.Module):
    """Two-layer feed-forward network with LayerNorm and GELU activation.

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
        # Use LayerNorm instead of BatchNorm1d to avoid in-place updates
        # to running statistics during training (which breaks autograd when
        # model is called multiple times in the same forward-backward cycle)
        self._norm_in = nn.LayerNorm(hidden_dim)
        self._norm_out = nn.LayerNorm(hidden_dim)
        self._ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters."""
        self._norm_in.reset_parameters()
        self._norm_out.reset_parameters()
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
        # LayerNorm normalizes over the last dimension (hidden_dim)
        # No permute needed unlike BatchNorm1d
        x = self._norm_in(x)
        x = self._ffn(x)
        x = self._norm_out(x)
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
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq, hidden_dim)``.
            attn_bias: Optional attention bias of shape
                ``(batch, num_heads, seq, seq)`` or broadcastable.
                Added as an additive mask to attention scores.
            valid_mask: Optional boolean tensor of shape ``(batch, seq)`` used
                to zero out padded token states after each residual block.

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
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        # Feed-forward block (pre-norm)
        residual = x
        x_norm = self._ffn_norm(x)
        ffn_output = self._ffn(x_norm)
        x = residual + ffn_output
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)

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
        pe_attr_names: List of node-level positional encoding attribute names.
            In ``"concat"`` mode these are concatenated to sequence features.
            In ``"add"`` mode they are projected to ``hid_dim`` and added to
            node features before sequence construction.
        anchor_based_pe_attr_names: List of relative-encoding attribute names
            containing sparse (N x N) matrices for anchor-relative positional
            encodings.
            These are used as additive attention bias for sequence keys.
        pairwise_pe_attr_names: List of relative-encoding attribute names
            containing sparse (N x N) matrices for pairwise relative encodings
            between sequence nodes. These are used as additive attention bias
            and can be combined with anchor-relative bias in the same model.
        feature_embedding_layer_dict: Optional ModuleDict mapping node types to
            feature embedding layers. If provided, these are applied to node
            features before node projection. (default: None)
        pe_integration_mode: How to fuse positional encodings into the model
            input. ``"concat"`` preserves the current behavior by concatenating
            node-level PE to token features. ``"add"`` uses node-level additive
            PE before sequence construction and attention bias for relative
            encodings.

    Notes:
        This encoder uses ``nn.LazyLinear`` for node-level PE fusion. If you wrap
        it with ``DistributedDataParallel``, run one representative no-grad
        forward first, passing ``anchor_node_ids``/``anchor_node_type`` for the
        graph-transformer path, or load a checkpoint before DDP so all ranks see
        initialized weights.

        TODO: Pairwise relative bias is currently materialized densely for the selected
            sequence. That is fine for moderate ``max_seq_len``, but a chunked or
            sparse LPFormer-style path is still future work for larger sequences.

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
        >>> embeddings = encoder(data, anchor_node_type=NodeType("user"), device=device)
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
        pairwise_pe_attr_names: Optional[list[str]] = None,
        feature_embedding_layer_dict: Optional[nn.ModuleDict] = None,
        pe_integration_mode: Literal["concat", "add"] = "concat",
        **kwargs: object,
    ) -> None:
        super().__init__()
        del kwargs

        if pe_integration_mode not in {"concat", "add"}:
            raise ValueError(
                "pe_integration_mode must be one of {'concat', 'add'}, "
                f"got '{pe_integration_mode}'"
            )

        self._hid_dim = hid_dim
        self._out_dim = out_dim
        self._max_seq_len = max_seq_len
        self._hop_distance = hop_distance
        self._should_l2_normalize_embedding_layer_output = (
            should_l2_normalize_embedding_layer_output
        )
        self._pe_attr_names = pe_attr_names
        self._anchor_based_pe_attr_names = anchor_based_pe_attr_names
        self._pairwise_pe_attr_names = pairwise_pe_attr_names
        self._feature_embedding_layer_dict = feature_embedding_layer_dict
        self._pe_integration_mode = pe_integration_mode
        self._num_heads = num_heads

        # Per-node-type input projection to hid_dim (like HGT's lin_dict)
        self._node_projection_dict = nn.ModuleDict(
            {
                str(node_type): nn.Linear(feat_dim, hid_dim)
                for node_type, feat_dim in node_type_to_feat_dim_map.items()
            }
        )

        # PE fusion layer. In concat mode, infer the concatenated input width
        # from the per-node feature tensors before sequence construction.
        self._pe_projection: Optional[nn.Module] = None
        if pe_integration_mode == "concat" and pe_attr_names:
            self._pe_projection = nn.LazyLinear(hid_dim)

        self._node_pe_projection: Optional[nn.Module] = None
        if pe_integration_mode == "add" and pe_attr_names:
            self._node_pe_projection = nn.LazyLinear(hid_dim, bias=False)

        self._anchor_pe_attention_bias_projection: Optional[nn.Linear] = None
        if anchor_based_pe_attr_names:
            self._anchor_pe_attention_bias_projection = nn.Linear(
                len(anchor_based_pe_attr_names),
                num_heads,
                bias=False,
            )

        self._pairwise_pe_attention_bias_projection: Optional[nn.Linear] = None
        if pairwise_pe_attr_names:
            self._pairwise_pe_attention_bias_projection = nn.Linear(
                len(pairwise_pe_attr_names),
                num_heads,
                bias=False,
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
        anchor_node_type: Optional[NodeType] = None,
        anchor_node_ids: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Run the forward pass of the Graph Transformer encoder.

        Args:
            data: Input HeteroData object with node features (``x_dict``)
                and edge indices (``edge_index_dict``).
            anchor_node_type: Node type for which to compute embeddings.
                If None, uses the first node type in data.
            anchor_node_ids: Optional tensor of local node indices within
                anchor_node_type to use as anchors. If None, uses the first
                batch_size nodes (seed nodes from neighbor sampling).
            device: Torch device for output tensors. If None, inferred from data.

        Returns:
            Embeddings tensor of shape ``(num_anchor_nodes, out_dim)``.
        """
        # Infer device from data if not provided
        if device is None:
            device = next(iter(data.x_dict.values())).device

        # Use first node type if not specified
        if anchor_node_type is None:
            anchor_node_type = list(data.node_types)[0]

        # 0. Apply feature embedding if provided (without modifying original data)
        # 1. Project all node features to hid_dim
        # Build a new x_dict with processed features to avoid in-place modifications
        projected_x_dict: dict[NodeType, torch.Tensor] = {}
        for node_type, x in data.x_dict.items():
            x_processed = x.to(device)
            # Apply feature embedding if available for this node type
            if self._feature_embedding_layer_dict is not None:
                if node_type in self._feature_embedding_layer_dict:
                    x_processed = self._feature_embedding_layer_dict[node_type](
                        x_processed
                    )
            # Project to hid_dim
            x_projected = self._node_projection_dict[str(node_type)](x_processed)
            if self._pe_attr_names:
                node_pe = _get_node_type_positional_encodings(
                    data=data,
                    node_type=node_type,
                    pe_attr_names=self._pe_attr_names,
                    device=device,
                )
                if self._pe_integration_mode == "add":
                    if self._node_pe_projection is None:
                        raise ValueError("Node PE projection layer is not initialized.")
                    x_projected = x_projected + self._node_pe_projection(node_pe)
                else:
                    if self._pe_projection is None:
                        raise ValueError(
                            "Concat PE projection layer is not initialized."
                        )
                    x_projected = self._pe_projection(
                        torch.cat([x_projected, node_pe], dim=-1)
                    )
            projected_x_dict[node_type] = x_projected

        # Create a new HeteroData with projected features (avoiding in-place modification)
        projected_data = torch_geometric.data.HeteroData()
        for node_type in data.node_types:
            projected_data[node_type].x = projected_x_dict[node_type]
            # Copy batch_size if it exists
            if hasattr(data[node_type], "batch_size"):
                projected_data[node_type].batch_size = data[node_type].batch_size
        for edge_type in data.edge_types:
            projected_data[edge_type].edge_index = data[edge_type].edge_index
        # Copy relative-encoding attributes (e.g., hop_distance stored as sparse matrix)
        relative_pe_attr_names = set(self._anchor_based_pe_attr_names or [])
        relative_pe_attr_names.update(self._pairwise_pe_attr_names or [])
        if relative_pe_attr_names:
            for attr_name in sorted(relative_pe_attr_names):
                if hasattr(data, attr_name):
                    setattr(projected_data, attr_name, getattr(data, attr_name))

        # 2. Build sequences and run transformer
        # If anchor_node_ids provided, use those; otherwise use first batch_size nodes
        if anchor_node_ids is not None:
            num_anchor_nodes = anchor_node_ids.size(0)
        else:
            num_anchor_nodes = getattr(
                projected_data[anchor_node_type],
                "batch_size",
                projected_data[anchor_node_type].num_nodes,
            )

        (
            sequences,
            valid_mask,
            attention_bias_data,
        ) = heterodata_to_graph_transformer_input(
            data=projected_data,
            batch_size=num_anchor_nodes,
            max_seq_len=self._max_seq_len,
            anchor_node_type=anchor_node_type,
            anchor_node_ids=anchor_node_ids,
            hop_distance=self._hop_distance,
            anchor_based_pe_attr_names=self._anchor_based_pe_attr_names,
            pairwise_pe_attr_names=self._pairwise_pe_attr_names,
        )

        # Free memory after sequences are built
        del projected_data

        if sequences.size(-1) != self._hid_dim:
            raise ValueError(
                f"Expected sequence dim {self._hid_dim} after node projection, "
                f"got {sequences.size(-1)}."
            )

        attn_bias = self._build_attention_bias(
            valid_mask=valid_mask,
            sequences=sequences,
            attention_bias_data=attention_bias_data,
        )

        embeddings = self._encode_and_readout(
            sequences=sequences,
            valid_mask=valid_mask,
            attn_bias=attn_bias,
        )
        embeddings = self._output_projection(embeddings)

        if self._should_l2_normalize_embedding_layer_output:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def _build_attention_bias(
        self,
        valid_mask: Tensor,
        sequences: Tensor,
        attention_bias_data: dict[str, Optional[Tensor]],
    ) -> Tensor:
        """Build additive attention bias from padding and learned relative PE projections."""
        batch_size, seq_len = valid_mask.shape
        dtype = sequences.dtype
        device = sequences.device
        negative_inf = torch.finfo(dtype).min

        attn_bias = torch.zeros(
            (batch_size, 1, 1, seq_len),
            dtype=dtype,
            device=device,
        )
        attn_bias = attn_bias.masked_fill(
            ~valid_mask.unsqueeze(1).unsqueeze(2),
            negative_inf,
        )

        anchor_bias_features = attention_bias_data.get("anchor_bias")
        if anchor_bias_features is not None:
            if self._anchor_pe_attention_bias_projection is None:
                raise ValueError("Anchor attention-bias projection is not initialized.")
            anchor_bias = self._anchor_pe_attention_bias_projection(
                anchor_bias_features.to(dtype)
            )
            anchor_bias = anchor_bias.permute(0, 2, 1).unsqueeze(2)
            attn_bias = attn_bias + anchor_bias

        pairwise_bias_features = attention_bias_data.get("pairwise_bias")
        if pairwise_bias_features is not None:
            if self._pairwise_pe_attention_bias_projection is None:
                raise ValueError(
                    "Pairwise attention-bias projection is not initialized."
                )
            pairwise_bias = self._pairwise_pe_attention_bias_projection(
                pairwise_bias_features.to(dtype)
            )
            pairwise_bias = pairwise_bias.permute(0, 3, 1, 2)
            attn_bias = attn_bias + pairwise_bias

        return attn_bias

    def _encode_and_readout(
        self,
        sequences: Tensor,
        valid_mask: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Process sequences through transformer layers and attention readout.

        Args:
            sequences: Input tensor of shape ``(batch_size, max_seq_len, hid_dim)``.
            valid_mask: Boolean mask of shape ``(batch_size, max_seq_len)``.
            attn_bias: Optional additive attention bias broadcastable to
                ``(batch_size, num_heads, seq, seq)``.

        Returns:
            Output embeddings of shape ``(batch_size, hid_dim)``.
        """
        x = sequences * valid_mask.unsqueeze(-1).to(sequences.dtype)

        for encoder_layer in self._encoder_layers:
            x = encoder_layer(x, attn_bias=attn_bias, valid_mask=valid_mask)

        x = self._final_norm(x)
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        # Readout: anchor (position 0) + attention-weighted neighbor aggregation
        anchor = x[:, 0, :].unsqueeze(1)  # (batch, 1, hid_dim)
        neighbors = x[:, 1:, :]  # (batch, seq-1, hid_dim)
        neighbor_valid_mask = valid_mask[:, 1:]
        seq_minus_one = neighbors.size(1)

        if seq_minus_one == 0:
            return anchor.squeeze(1)

        # Expand anchor to match neighbor dimension for concatenation
        anchor_expanded = anchor.expand(-1, seq_minus_one, -1)

        # Compute attention scores over neighbors
        readout_scores = self._readout_attention(
            torch.cat([anchor_expanded, neighbors], dim=-1)
        )  # (batch, seq-1, 1)
        readout_scores = readout_scores.masked_fill(
            ~neighbor_valid_mask.unsqueeze(-1),
            torch.finfo(readout_scores.dtype).min,
        )
        readout_weights = F.softmax(readout_scores, dim=1)  # (batch, seq-1, 1)
        readout_weights = torch.nan_to_num(readout_weights, nan=0.0)
        readout_weights = readout_weights * neighbor_valid_mask.unsqueeze(-1).to(
            readout_weights.dtype
        )

        neighbor_aggregation = (neighbors * readout_weights).sum(
            dim=1, keepdim=True
        )  # (batch, 1, hid_dim)

        output = (anchor + neighbor_aggregation).squeeze(1)  # (batch, hid_dim)

        return output
