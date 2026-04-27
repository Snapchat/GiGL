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

import math
from typing import Callable, Literal, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.hetero_data
from torch import Tensor

from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.transforms.graph_transformer import (
    PPR_WEIGHT_FEATURE_NAME,
    SequenceAuxiliaryData,
    TokenInputData,
    heterodata_to_graph_transformer_input,
)


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


def _build_sinusoidal_sequence_position_table(
    max_seq_len: int,
    hid_dim: int,
) -> Tensor:
    """Build a standard sinusoidal absolute position table."""
    positions = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hid_dim, 2, dtype=torch.float) * (-math.log(10000.0) / hid_dim)
    )

    position_table = torch.zeros(max_seq_len, hid_dim, dtype=torch.float)
    position_table[:, 0::2] = torch.sin(positions * div_term)
    if hid_dim > 1:
        position_table[:, 1::2] = torch.cos(
            positions * div_term[: position_table[:, 1::2].shape[1]]
        )
    return position_table


# Supported activation functions for FeedForwardNetwork
_ACTIVATION_FNS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,  # Also known as Swish
    "tanh": nn.Tanh,
}

# XGLU activations use a gating mechanism: activation(xW) * xV
# where W and V are separate linear projections
_XGLU_BASE_ACTIVATIONS = {
    "geglu": F.gelu,
    "swiglu": F.silu,
    "reglu": F.relu,
}


class FeedForwardNetwork(nn.Module):
    """Two-layer feed-forward network with configurable activation.

    Supports standard activations (GELU, ReLU, SiLU) and XGLU family
    (SwiGLU, GeGLU, ReGLU) which use a gating mechanism.

    Note: This module does NOT include LayerNorm. Normalization should be
    applied externally (e.g., pre-norm in the transformer layer).

    Adapted from RelGT's FeedForwardNetwork.

    Args:
        model_dim: Model (input and output) dimension of the FFN.
        feedforward_dim: Inner dimension of the two-layer MLP.
        dropout_rate: Dropout probability applied after each linear layer.
        activation: Activation function name. Supported values:
            - Standard: "gelu" (default), "relu", "silu", "tanh"
            - XGLU family: "geglu", "swiglu", "reglu"
            XGLU activations use gating: activation(xW) * xV, which requires
            projecting to 2x feedforward_dim internally.
    """

    def __init__(
        self,
        model_dim: int,
        feedforward_dim: int,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self._activation_name = activation.lower()

        # Validate activation
        if (
            self._activation_name not in _ACTIVATION_FNS
            and self._activation_name not in _XGLU_BASE_ACTIVATIONS
        ):
            supported = sorted(
                set(_ACTIVATION_FNS.keys()) | set(_XGLU_BASE_ACTIVATIONS.keys())
            )
            raise ValueError(
                f"Unsupported activation '{activation}'. Supported: {supported}"
            )

        self._is_xglu = self._activation_name in _XGLU_BASE_ACTIVATIONS

        # Type declarations for optional attributes
        self._xglu_base_activation: Optional[Callable[..., Tensor]] = None
        self._linear_in: Optional[nn.Linear] = None
        self._dropout_in: Optional[nn.Dropout] = None
        self._linear_out: Optional[nn.Linear] = None
        self._dropout_out: Optional[nn.Dropout] = None
        self._ffn: Optional[nn.Sequential] = None

        if self._is_xglu:
            # XGLU: project to 2x feedforward_dim, split, apply gating
            self._xglu_base_activation = cast(
                Callable[..., Tensor], _XGLU_BASE_ACTIVATIONS[self._activation_name]
            )
            self._linear_in = nn.Linear(model_dim, feedforward_dim * 2)
            self._dropout_in = nn.Dropout(dropout_rate)
            self._linear_out = nn.Linear(feedforward_dim, model_dim)
            self._dropout_out = nn.Dropout(dropout_rate)
        else:
            # Standard activation
            activation_fn = _ACTIVATION_FNS[self._activation_name]
            self._ffn = nn.Sequential(
                nn.Linear(model_dim, feedforward_dim),
                activation_fn(),
                nn.Dropout(dropout_rate),
                nn.Linear(feedforward_dim, model_dim),
                nn.Dropout(dropout_rate),
            )

    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters."""
        if self._is_xglu:
            assert self._linear_in is not None
            assert self._linear_out is not None
            nn.init.xavier_uniform_(self._linear_in.weight)
            nn.init.zeros_(self._linear_in.bias)
            nn.init.xavier_uniform_(self._linear_out.weight)
            nn.init.zeros_(self._linear_out.bias)
        else:
            # Use xavier + zero bias for consistency with XGLU path and
            # GraphTransformerEncoderLayer (standard Transformer practice)
            assert self._ffn is not None
            for layer in self._ffn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq, model_dim)``.

        Returns:
            Output tensor of shape ``(batch, seq, model_dim)``.
        """
        if self._is_xglu:
            # XGLU gating: activation(x @ W1) * (x @ W2)
            # where W1 and W2 are the two halves of linear_in
            assert self._xglu_base_activation is not None
            assert self._linear_in is not None
            assert self._dropout_in is not None
            assert self._linear_out is not None
            assert self._dropout_out is not None
            x_proj = self._linear_in(x)  # (batch, seq, feedforward_dim * 2)
            x_gate, x_value = x_proj.chunk(
                2, dim=-1
            )  # Each: (batch, seq, feedforward_dim)
            x = self._xglu_base_activation(x_gate) * x_value
            x = self._dropout_in(x)
            x = self._linear_out(x)
            x = self._dropout_out(x)
        else:
            assert self._ffn is not None
            x = self._ffn(x)

        return x


class GraphTransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with multi-head self-attention.

    Uses ``F.scaled_dot_product_attention`` which automatically selects the
    most efficient attention implementation (flash, memory-efficient, or
    math-based) based on input properties and hardware.

    Adapted from RelGT's EncoderLayer.

    Args:
        model_dim: Model dimension (d_model).
        num_heads: Number of attention heads. Must evenly divide model_dim.
        feedforward_dim: Inner dimension of the feed-forward network.
        dropout_rate: Dropout probability for feed-forward layers.
        attention_dropout_rate: Dropout probability for attention weights.
        activation: Activation function for the feed-forward network.
            Supported values: "gelu" (default), "relu", "silu", "tanh",
            "geglu", "swiglu", "reglu".
        relation_attention_mode: Optional relation-aware augmentation strategy
            for attention scores. ``"none"`` preserves the default shared
            self-attention path. ``"edge_type_additive"`` adds a learned
            per-edge-type bilinear term for token pairs backed by sampled
            directed graph edges.
        num_relations: Number of relation channels expected in
            ``pairwise_relation_mask`` when
            ``relation_attention_mode="edge_type_additive"``.

    Raises:
        ValueError: If model_dim is not divisible by num_heads.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        activation: str = "gelu",
        relation_attention_mode: Literal["none", "edge_type_additive"] = "none",
        num_relations: int = 0,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            )
        if relation_attention_mode not in {"none", "edge_type_additive"}:
            raise ValueError(
                "relation_attention_mode must be one of "
                "{'none', 'edge_type_additive'}, "
                f"got '{relation_attention_mode}'"
            )
        if relation_attention_mode == "edge_type_additive" and num_relations <= 0:
            raise ValueError(
                "relation_attention_mode='edge_type_additive' requires "
                "num_relations > 0."
            )

        self._num_heads = num_heads
        self._head_dim = model_dim // num_heads
        self._attention_dropout_rate = attention_dropout_rate
        self._relation_attention_mode = relation_attention_mode
        self._num_relations = num_relations

        self._attention_norm = nn.LayerNorm(model_dim)
        self._query_projection = nn.Linear(model_dim, model_dim)
        self._key_projection = nn.Linear(model_dim, model_dim)
        self._value_projection = nn.Linear(model_dim, model_dim)
        self._output_projection = nn.Linear(model_dim, model_dim)
        self._dropout = nn.Dropout(dropout_rate)
        self._relation_attention_matrices: Optional[nn.Parameter] = None
        if relation_attention_mode == "edge_type_additive":
            self._relation_attention_matrices = nn.Parameter(
                torch.empty(num_relations, num_heads, self._head_dim, self._head_dim)
            )

        self._ffn_norm = nn.LayerNorm(model_dim)
        self._ffn = FeedForwardNetwork(
            model_dim, feedforward_dim, dropout_rate, activation=activation
        )

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
        if self._relation_attention_matrices is not None:
            for relation_matrices in self._relation_attention_matrices:
                for head_matrix in relation_matrices:
                    nn.init.xavier_uniform_(head_matrix)
        self._ffn_norm.reset_parameters()
        self._ffn.reset_parameters()

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
        pairwise_relation_mask: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq, model_dim)``.
            attn_bias: Optional attention bias of shape
                ``(batch, num_heads, seq, seq)`` or broadcastable.
                Added as an additive mask to attention scores.
            pairwise_relation_mask: Optional boolean multi-hot relation mask of shape
                ``(batch, seq, seq, num_relations)`` that marks which sampled
                directed edge types connect each token pair as ``key -> query``.
            valid_mask: Optional boolean tensor of shape ``(batch, seq)`` used
                to zero out padded token states after each residual block.

        Returns:
            Output tensor of shape ``(batch, seq, model_dim)``.
        """
        batch_size, seq_len, model_dim = x.shape

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

        if self._relation_attention_mode == "none":
            attention_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_bias,
                dropout_p=self._attention_dropout_rate if self.training else 0.0,
                is_causal=False,
            )
        else:
            attention_output = self._run_relation_aware_attention(
                query=query,
                key=key,
                value=value,
                attn_bias=attn_bias,
                pairwise_relation_mask=pairwise_relation_mask,
            )

        # Reshape back to (batch, seq, model_dim)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, seq_len, model_dim
        )
        attention_output = self._output_projection(attention_output)
        attention_output = self._dropout(attention_output)

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

    def _run_relation_aware_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias: Optional[Tensor],
        pairwise_relation_mask: Optional[Tensor],
    ) -> Tensor:
        relation_attention_bias = self._build_relation_attention_bias(
            query=query,
            key=key,
            pairwise_relation_mask=pairwise_relation_mask,
        )
        if relation_attention_bias is not None:
            attn_bias = (
                relation_attention_bias
                if attn_bias is None
                else attn_bias + relation_attention_bias
            )

        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_bias,
            dropout_p=self._attention_dropout_rate if self.training else 0.0,
            is_causal=False,
        )

    def _build_relation_attention_bias(
        self,
        query: Tensor,
        key: Tensor,
        pairwise_relation_mask: Optional[Tensor],
    ) -> Optional[Tensor]:
        if pairwise_relation_mask is None:
            raise ValueError(
                "pairwise_relation_mask is required when "
                "relation_attention_mode='edge_type_additive'."
            )
        if pairwise_relation_mask.size(-1) != self._num_relations:
            raise ValueError(
                "pairwise_relation_mask has unexpected relation dimension "
                f"{pairwise_relation_mask.size(-1)}; expected {self._num_relations}."
            )
        if self._relation_attention_matrices is None:
            raise ValueError("Relation attention matrices are not initialized.")
        if pairwise_relation_mask.size(1) != query.size(2) or pairwise_relation_mask.size(
            2
        ) != key.size(2):
            raise ValueError(
                "pairwise_relation_mask must align with the query/key sequence "
                "dimensions."
            )

        relation_mask = pairwise_relation_mask.to(
            device=query.device,
            dtype=torch.bool,
        )
        active_relation_positions = relation_mask.nonzero(as_tuple=False)
        if active_relation_positions.numel() == 0:
            return None

        relation_attention_bias = query.new_zeros(
            (query.size(0), query.size(2), key.size(2), self._num_heads)
        )
        query_by_position = query.transpose(1, 2)
        key_by_position = key.transpose(1, 2)
        relation_matrices = self._relation_attention_matrices.to(dtype=query.dtype)
        active_relation_ids = torch.unique(active_relation_positions[:, 3], sorted=True)

        for relation_idx_tensor in active_relation_ids:
            relation_idx = int(relation_idx_tensor.item())
            relation_positions = active_relation_positions[
                active_relation_positions[:, 3] == relation_idx
            ]
            batch_indices, query_indices, key_indices = relation_positions[
                :, :3
            ].unbind(dim=1)
            # Only materialize bilinear scores for token pairs backed by this relation.
            selected_query = query_by_position[batch_indices, query_indices]
            transformed_query = torch.einsum(
                "nhe,hde->nhd",
                selected_query,
                relation_matrices[relation_idx],
            )
            selected_key = key_by_position[batch_indices, key_indices]
            relation_scores = (selected_key * transformed_query).sum(dim=-1)
            relation_attention_bias.index_put_(
                (batch_indices, query_indices, key_indices),
                relation_scores / math.sqrt(self._head_dim),
                accumulate=True,
            )

        return relation_attention_bias.permute(0, 3, 1, 2)


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
            graph-to-sequence transform when using ``"khop"`` sequence construction.
        sequence_construction_method: Sequence builder used to create tokens for
            each anchor. ``"khop"`` expands the sampled graph by hop distance,
            while ``"ppr"`` consumes outgoing ``"ppr"`` edges sorted by weight.
        sequence_positional_encoding_type: Optional sequence-level positional
            encoding applied after sequence construction. Supported values are
            ``None`` and ``"sinusoidal"``. Lower-cost future extensions could
            add learned absolute position embeddings here, while attention-level
            options like RoPE or ALiBi would require changes inside the
            attention block.
        dropout_rate: Dropout probability for feed-forward layers.
        attention_dropout_rate: Dropout probability for attention weights.
        should_l2_normalize_embedding_layer_output: Whether to L2 normalize
            output embeddings.
        pe_attr_names: List of node-level positional encoding attribute names.
            In ``"concat"`` mode these are concatenated to sequence features.
            In ``"add"`` mode they are projected to ``hid_dim`` and added to
            node features before sequence construction.
        anchor_based_attention_bias_attr_names: List of anchor-relative feature
            names used as additive attention bias for sequence keys. Sparse
            graph-level attributes are looked up from ``data`` and the reserved
            name ``"ppr_weight"`` resolves to PPR edge weights in PPR mode.
            Example: ``['hop_distance', 'ppr_weight']`` where ``hop_distance``
            is a sparse matrix attribute on ``data`` and ``ppr_weight`` is
            extracted from PPR edge weights.
        anchor_based_input_attr_names: List of anchor-relative attribute names
            used as token-aligned input features. Sparse graph-level attributes
            are looked up from ``data`` and ``"ppr_weight"`` resolves to PPR
            edge weights in PPR mode. These are projected to ``hid_dim`` and
            added to the sequence tokens after sequence construction.
            Example: ``['hop_distance', 'ppr_weight']`` for continuous features,
            or ``['hop_distance']`` when ``hop_distance`` will be embedded via
            ``anchor_based_input_embedding_dict``.
        anchor_based_input_embedding_dict: Optional ModuleDict mapping a subset
            of ``anchor_based_input_attr_names`` to per-attribute embedding
            layers. These attributes are treated as discrete indices and their
            embedded contributions are added to the sequence tokens. Padding is
            masked out using the sequence valid mask.
            Example: ``nn.ModuleDict({'hop_distance': nn.Embedding(10, hid_dim)})``
            to embed hop distances 0-9 into ``hid_dim``-dimensional vectors.
            The embedding output dimension must match ``hid_dim``.
        pairwise_attention_bias_attr_names: List of pairwise feature names used
            as additive attention bias. These must correspond to sparse
            graph-level attributes on ``data``.
        feature_embedding_layer_dict: Optional ModuleDict mapping node types to
            feature embedding layers. If provided, these are applied to node
            features before node projection. (default: None)
        pe_integration_mode: How to fuse positional encodings into the model
            input. ``"concat"`` preserves the current behavior by concatenating
            node-level PE to token features. ``"add"`` uses node-level additive
            PE before sequence construction and attention bias for relative
            encodings.
        activation: Activation function for the feed-forward network in each
            transformer layer. Supported values:
            - Standard: "gelu" (default), "relu", "silu", "tanh"
            - XGLU family: "geglu", "swiglu", "reglu"
            XGLU activations use gating: activation(xW) * xV.
        feedforward_ratio: Ratio of feedforward dimension to hidden dimension
            (feedforward_dim = hid_dim * feedforward_ratio). If None (default),
            uses 4.0 for standard activations and 8/3 (~2.67) for XGLU variants,
            following the convention that XGLU's gating doubles the effective
            parameters, so a smaller ratio maintains similar parameter count.
        relation_attention_mode: Optional relation-aware augmentation for
            attention scores. ``"none"`` preserves the current dense transformer
            path. ``"edge_type_additive"`` adds a learned per-edge-type
            bilinear score term for sampled directed edges in ``"khop"`` mode.

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
        sequence_construction_method: Literal["khop", "ppr"] = "khop",
        sequence_positional_encoding_type: Optional[str] = None,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        should_l2_normalize_embedding_layer_output: bool = False,
        pe_attr_names: Optional[list[str]] = None,
        anchor_based_attention_bias_attr_names: Optional[list[str]] = None,
        anchor_based_input_attr_names: Optional[list[str]] = None,
        anchor_based_input_embedding_dict: Optional[nn.ModuleDict] = None,
        pairwise_attention_bias_attr_names: Optional[list[str]] = None,
        feature_embedding_layer_dict: Optional[nn.ModuleDict] = None,
        pe_integration_mode: Literal["concat", "add"] = "concat",
        activation: str = "gelu",
        feedforward_ratio: Optional[float] = None,
        relation_attention_mode: Literal["none", "edge_type_additive"] = "none",
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
        if sequence_construction_method not in {"khop", "ppr"}:
            raise ValueError(
                "sequence_construction_method must be one of {'khop', 'ppr'}, "
                f"got '{sequence_construction_method}'"
            )
        if sequence_positional_encoding_type is not None:
            sequence_positional_encoding_type = (
                sequence_positional_encoding_type.lower()
            )
            if sequence_positional_encoding_type == "none":
                sequence_positional_encoding_type = None
        if sequence_positional_encoding_type not in {None, "sinusoidal"}:
            raise ValueError(
                "sequence_positional_encoding_type must be one of "
                "{None, 'sinusoidal'}, "
                f"got '{sequence_positional_encoding_type}'"
            )
        if (
            sequence_construction_method == "khop"
            and sequence_positional_encoding_type is not None
        ):
            raise ValueError(
                "sequence_positional_encoding_type requires "
                "sequence_construction_method='ppr' because khop sequences do not "
                "enforce a stable token order."
            )
        if relation_attention_mode not in {"none", "edge_type_additive"}:
            raise ValueError(
                "relation_attention_mode must be one of "
                "{'none', 'edge_type_additive'}, "
                f"got '{relation_attention_mode}'"
            )
        if (
            relation_attention_mode == "edge_type_additive"
            and sequence_construction_method != "khop"
        ):
            raise ValueError(
                "relation_attention_mode='edge_type_additive' requires "
                "sequence_construction_method='khop'."
            )
        anchor_bias_attr_names = anchor_based_attention_bias_attr_names or []
        anchor_input_attr_names = anchor_based_input_attr_names or []
        pairwise_bias_attr_names = pairwise_attention_bias_attr_names or []
        if PPR_WEIGHT_FEATURE_NAME in pairwise_bias_attr_names:
            raise ValueError(
                f"'{PPR_WEIGHT_FEATURE_NAME}' is an anchor-relative feature and "
                "cannot be used as pairwise attention bias."
            )
        if (
            PPR_WEIGHT_FEATURE_NAME in anchor_bias_attr_names + anchor_input_attr_names
            and sequence_construction_method != "ppr"
        ):
            raise ValueError(
                "The reserved anchor-relative feature 'ppr_weight' requires "
                "sequence_construction_method='ppr'."
            )
        self._sequence_construction_method = sequence_construction_method
        self._sequence_positional_encoding_type = sequence_positional_encoding_type
        self._should_l2_normalize_embedding_layer_output = (
            should_l2_normalize_embedding_layer_output
        )
        self._pe_attr_names = pe_attr_names
        self._anchor_based_attention_bias_attr_names = (
            anchor_based_attention_bias_attr_names
        )
        self._anchor_based_input_attr_names = anchor_based_input_attr_names
        self._anchor_based_input_embedding_dict = anchor_based_input_embedding_dict
        self._pairwise_attention_bias_attr_names = pairwise_attention_bias_attr_names
        self._feature_embedding_layer_dict = feature_embedding_layer_dict
        self._pe_integration_mode = pe_integration_mode
        self._num_heads = num_heads
        self._relation_attention_mode = relation_attention_mode
        self._relation_attention_edge_types = (
            sorted(edge_type_to_feat_dim_map.keys())
            if relation_attention_mode == "edge_type_additive"
            else []
        )
        anchor_input_embedding_attr_names = (
            set(anchor_based_input_embedding_dict.keys())
            if anchor_based_input_embedding_dict is not None
            else set()
        )
        invalid_anchor_input_embedding_attr_names = (
            anchor_input_embedding_attr_names - set(anchor_input_attr_names)
        )
        if invalid_anchor_input_embedding_attr_names:
            raise ValueError(
                "anchor_based_input_embedding_dict keys must be a subset of "
                "anchor_based_input_attr_names, got unexpected keys "
                f"{sorted(invalid_anchor_input_embedding_attr_names)}."
            )
        self._continuous_anchor_input_attr_names = [
            attr_name
            for attr_name in anchor_input_attr_names
            if attr_name not in anchor_input_embedding_attr_names
        ]
        if self._sequence_positional_encoding_type == "sinusoidal":
            self.register_buffer(
                "_sequence_positional_encoding_table",
                _build_sinusoidal_sequence_position_table(
                    max_seq_len=max_seq_len,
                    hid_dim=hid_dim,
                ),
                persistent=False,
            )
        else:
            self.register_buffer(
                "_sequence_positional_encoding_table",
                None,
                persistent=False,
            )

        # Per-node-type input projection to hid_dim (like HGT's lin_dict)
        self._node_projection_dict = nn.ModuleDict(
            {
                str(node_type): nn.Linear(feat_dim, hid_dim)
                for node_type, feat_dim in node_type_to_feat_dim_map.items()
            }
        )

        # PE fusion layers for node-level positional encodings.
        # In "concat" mode: projects [node_features || PE] → hid_dim
        # In "add" mode: projects PE → hid_dim, then adds to node features
        self._concat_pe_fusion_projection: Optional[nn.Module] = None
        has_node_level_pe = bool(pe_attr_names)
        if pe_integration_mode == "concat" and has_node_level_pe:
            self._concat_pe_fusion_projection = nn.LazyLinear(hid_dim)

        self._pe_projection: Optional[nn.Module] = None
        if pe_integration_mode == "add" and has_node_level_pe:
            self._pe_projection = nn.LazyLinear(hid_dim, bias=False)

        self._token_input_projection: Optional[nn.Module] = None
        if self._continuous_anchor_input_attr_names:
            self._token_input_projection = nn.LazyLinear(hid_dim, bias=False)

        self._anchor_pe_attention_bias_projection: Optional[nn.Linear] = None
        num_anchor_bias_attrs = len(self._anchor_based_attention_bias_attr_names or [])
        if num_anchor_bias_attrs > 0:
            self._anchor_pe_attention_bias_projection = nn.Linear(
                num_anchor_bias_attrs,
                num_heads,
                bias=False,
            )

        self._pairwise_pe_attention_bias_projection: Optional[nn.Linear] = None
        if self._pairwise_attention_bias_attr_names:
            self._pairwise_pe_attention_bias_projection = nn.Linear(
                len(self._pairwise_attention_bias_attr_names),
                num_heads,
                bias=False,
            )

        # Transformer encoder layers
        # Default feedforward ratio: 4.0 for standard activations, 8/3 for XGLU
        # XGLU's gating mechanism doubles effective parameters, so smaller ratio
        # maintains similar parameter count to standard activations with ratio 4.
        is_xglu = activation.lower() in _XGLU_BASE_ACTIVATIONS
        if feedforward_ratio is None:
            feedforward_ratio = 8.0 / 3.0 if is_xglu else 4.0
        feedforward_dim = int(hid_dim * feedforward_ratio)
        self._encoder_layers = nn.ModuleList(
            [
                GraphTransformerEncoderLayer(
                    model_dim=hid_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    activation=activation,
                    relation_attention_mode=relation_attention_mode,
                    num_relations=len(self._relation_attention_edge_types),
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
            feature_embedding_layer = None
            if (
                self._feature_embedding_layer_dict is not None
                and node_type in self._feature_embedding_layer_dict
            ):
                feature_embedding_layer = self._feature_embedding_layer_dict[node_type]
            # Apply feature embedding if available for this node type
            if feature_embedding_layer is not None:
                x_processed = feature_embedding_layer(x_processed)
            # Project to hid_dim
            x_projected = self._node_projection_dict[str(node_type)](x_processed)
            node_pe_parts = []
            if self._pe_attr_names:
                node_pe_parts.append(
                    _get_node_type_positional_encodings(
                        data=data,
                        node_type=node_type,
                        pe_attr_names=self._pe_attr_names,
                        device=device,
                    )
                )
            if node_pe_parts:
                node_pe = torch.cat(node_pe_parts, dim=-1)
                if self._pe_integration_mode == "add":
                    if self._pe_projection is None:
                        raise ValueError("PE projection layer is not initialized.")
                    x_projected = x_projected + self._pe_projection(node_pe)
                else:
                    if self._concat_pe_fusion_projection is None:
                        raise ValueError(
                            "Concat PE fusion projection layer is not initialized."
                        )
                    x_projected = self._concat_pe_fusion_projection(
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
            if hasattr(data[edge_type], "edge_attr"):
                projected_data[edge_type].edge_attr = data[edge_type].edge_attr
        # Copy relative-encoding attributes (e.g., hop_distance stored as sparse matrix)
        relative_pe_attr_names = {
            attr_name
            for attr_name in (self._anchor_based_attention_bias_attr_names or [])
            if attr_name != PPR_WEIGHT_FEATURE_NAME
        }
        relative_pe_attr_names.update(self._anchor_based_input_attr_names or [])
        relative_pe_attr_names.update(self._pairwise_attention_bias_attr_names or [])
        relative_pe_attr_names.discard(PPR_WEIGHT_FEATURE_NAME)
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
            sequence_auxiliary_data,
        ) = heterodata_to_graph_transformer_input(
            data=projected_data,
            batch_size=num_anchor_nodes,
            max_seq_len=self._max_seq_len,
            anchor_node_type=anchor_node_type,
            anchor_node_ids=anchor_node_ids,
            hop_distance=self._hop_distance,
            sequence_construction_method=self._sequence_construction_method,
            anchor_based_attention_bias_attr_names=self._anchor_based_attention_bias_attr_names,
            anchor_based_input_attr_names=self._anchor_based_input_attr_names,
            pairwise_attention_bias_attr_names=self._pairwise_attention_bias_attr_names,
            relation_edge_types=self._relation_attention_edge_types,
        )

        # Free memory after sequences are built
        del projected_data

        if sequences.size(-1) != self._hid_dim:
            raise ValueError(
                f"Expected sequence dim {self._hid_dim} after node projection, "
                f"got {sequences.size(-1)}."
            )

        token_input_features = sequence_auxiliary_data["token_input"]
        if token_input_features is not None:
            sequences = sequences + self._build_token_input_contribution(
                token_input_features=token_input_features,
                sequences=sequences,
                valid_mask=valid_mask,
            )

        sequence_positional_encoding = self._get_sequence_positional_encoding(
            valid_mask=valid_mask,
            sequences=sequences,
        )
        if sequence_positional_encoding is not None:
            sequences = sequences + sequence_positional_encoding

        attn_bias = self._build_attention_bias(
            valid_mask=valid_mask,
            sequences=sequences,
            attention_bias_data=sequence_auxiliary_data,
        )

        embeddings = self._encode_and_readout(
            sequences=sequences,
            valid_mask=valid_mask,
            attn_bias=attn_bias,
            pairwise_relation_mask=sequence_auxiliary_data.get(
                "pairwise_relation_mask"
            ),
        )
        embeddings = self._output_projection(embeddings)

        if self._should_l2_normalize_embedding_layer_output:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def _get_sequence_positional_encoding(
        self,
        valid_mask: Tensor,
        sequences: Tensor,
    ) -> Optional[Tensor]:
        if self._sequence_positional_encoding_type is None:
            return None
        if self._sequence_positional_encoding_type != "sinusoidal":
            raise ValueError(
                "Unsupported sequence_positional_encoding_type "
                f"'{self._sequence_positional_encoding_type}'."
            )
        if self._sequence_positional_encoding_table is None:
            raise ValueError("Sequence positional encoding table is not initialized.")
        position_table = cast(Tensor, self._sequence_positional_encoding_table)

        seq_len = sequences.size(1)
        if seq_len > position_table.size(0):
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured max_seq_len "
                f"{position_table.size(0)}."
            )

        position_encoding = position_table[:seq_len]
        position_encoding = position_encoding.to(
            device=sequences.device,
            dtype=sequences.dtype,
        )
        position_encoding = position_encoding.unsqueeze(0).expand(
            sequences.size(0), -1, -1
        )
        return position_encoding * valid_mask.unsqueeze(-1).to(sequences.dtype)

    def _build_token_input_contribution(
        self,
        token_input_features: TokenInputData,
        sequences: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        token_contribution = torch.zeros_like(sequences)
        valid_token_mask = valid_mask.unsqueeze(-1).to(sequences.dtype)

        if self._anchor_based_input_embedding_dict is not None:
            for (
                attr_name,
                embedding_layer,
            ) in self._anchor_based_input_embedding_dict.items():
                if attr_name not in token_input_features:
                    raise ValueError(
                        f"Token-input feature '{attr_name}' is missing from the "
                        "sequence auxiliary data."
                    )
                indices = token_input_features[attr_name]
                if indices.size(-1) != 1:
                    raise ValueError(
                        f"Embedded token-input feature '{attr_name}' must have "
                        f"shape (batch, seq, 1), got {indices.shape}."
                    )
                embedded_attr = embedding_layer(indices.squeeze(-1).long())
                if embedded_attr.shape != sequences.shape:
                    raise ValueError(
                        f"Embedded token-input feature '{attr_name}' must produce "
                        f"shape {sequences.shape}, got {embedded_attr.shape}."
                    )
                token_contribution = token_contribution + (
                    embedded_attr.to(sequences.dtype) * valid_token_mask
                )

        if self._continuous_anchor_input_attr_names:
            if self._token_input_projection is None:
                raise ValueError("Token-input projection is not initialized.")
            continuous_feature_parts: list[Tensor] = []
            for attr_name in self._continuous_anchor_input_attr_names:
                if attr_name not in token_input_features:
                    raise ValueError(
                        f"Token-input feature '{attr_name}' is missing from the "
                        "sequence auxiliary data."
                    )
                continuous_feature_parts.append(token_input_features[attr_name])
            token_contribution = token_contribution + (
                self._token_input_projection(
                    torch.cat(continuous_feature_parts, dim=-1).to(sequences.dtype)
                )
                * valid_token_mask
            )

        return token_contribution

    def _build_attention_bias(
        self,
        valid_mask: Tensor,
        sequences: Tensor,
        attention_bias_data: SequenceAuxiliaryData,
    ) -> Tensor:
        """Build additive attention bias from padding mask and learned relative PE projections.

        This function constructs a combined attention bias tensor that is added to
        attention scores before softmax. The bias has three components:

        1. **Padding mask bias**: Sets padded positions to -inf so they receive zero
           attention weight after softmax. Shape: (batch, 1, 1, seq) broadcasts to
           (batch, num_heads, seq, seq) for key masking.

        2. **Anchor-relative bias** (optional): For each sequence position, looks up
           the PE value relative to the anchor (e.g., hop distance from anchor).
           Input shape: (batch, seq, num_anchor_attrs)
           After projection: (batch, num_heads, 1, seq) - same bias for all query positions.

        3. **Pairwise bias** (optional): For each (query, key) pair, looks up the PE
           value between those two nodes (e.g., random walk structural encoding).
           Input shape: (batch, seq, seq, num_pairwise_attrs)
           After projection: (batch, num_heads, seq, seq) - unique bias per query-key pair.

        Args:
            valid_mask: Boolean mask of shape (batch_size, seq_len) indicating
                valid (non-padding) positions.
            sequences: Input sequences of shape (batch_size, seq_len, hid_dim),
                used only to infer dtype and device.
            attention_bias_data: Dictionary containing optional PE tensors:
                - "anchor_bias": (batch, seq, num_anchor_attrs) or None
                - "pairwise_bias": (batch, seq, seq, num_pairwise_attrs) or None

        Returns:
            Combined attention bias tensor of shape (batch_size, num_heads, seq_len, seq_len)
            or broadcastable shape. Added to attention scores before softmax.

        Example:
            # With batch_size=2, seq_len=4, num_heads=8
            # valid_mask = [[T, T, T, F], [T, T, F, F]]
            #
            # Output attn_bias shape: (2, 8, 4, 4)
            # - Positions where valid_mask is False get -inf
            # - Anchor bias adds per-key bias (same for all queries)
            # - Pairwise bias adds unique bias for each (query, key) pair
        """
        batch_size, seq_len = valid_mask.shape
        dtype = sequences.dtype
        device = sequences.device
        negative_inf = torch.finfo(dtype).min

        # Step 1: Initialize with padding mask bias
        # Shape: (batch, 1, 1, seq) - broadcasts to mask invalid keys for all queries/heads
        attn_bias = torch.zeros(
            (batch_size, 1, 1, seq_len),
            dtype=dtype,
            device=device,
        )
        attn_bias = attn_bias.masked_fill(
            ~valid_mask.unsqueeze(1).unsqueeze(2),  # (batch, 1, 1, seq)
            negative_inf,
        )

        # Step 2: Add anchor-relative bias (optional)
        # Projects (batch, seq, num_attrs) → (batch, seq, num_heads)
        # Then reshapes to (batch, num_heads, 1, seq) for key-side bias
        anchor_bias_features = attention_bias_data.get("anchor_bias")
        if anchor_bias_features is not None:
            if self._anchor_pe_attention_bias_projection is None:
                raise ValueError("Anchor attention-bias projection is not initialized.")
            anchor_bias = self._anchor_pe_attention_bias_projection(
                anchor_bias_features.to(dtype)
            )  # (batch, seq, num_heads)
            anchor_bias = anchor_bias.permute(0, 2, 1).unsqueeze(
                2
            )  # (batch, num_heads, 1, seq)
            attn_bias = attn_bias + anchor_bias

        # Step 3: Add pairwise bias (optional)
        # Projects (batch, seq, seq, num_attrs) → (batch, seq, seq, num_heads)
        # Then reshapes to (batch, num_heads, seq, seq)
        pairwise_bias_features = attention_bias_data.get("pairwise_bias")
        if pairwise_bias_features is not None:
            if self._pairwise_pe_attention_bias_projection is None:
                raise ValueError(
                    "Pairwise attention-bias projection is not initialized."
                )
            pairwise_bias = self._pairwise_pe_attention_bias_projection(
                pairwise_bias_features.to(dtype)
            )  # (batch, seq, seq, num_heads)
            pairwise_bias = pairwise_bias.permute(
                0, 3, 1, 2
            )  # (batch, num_heads, seq, seq)
            attn_bias = attn_bias + pairwise_bias

        return attn_bias

    def _encode_and_readout(
        self,
        sequences: Tensor,
        valid_mask: Tensor,
        attn_bias: Optional[Tensor] = None,
        pairwise_relation_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Process sequences through transformer layers and attention readout.

        Args:
            sequences: Input tensor of shape ``(batch_size, max_seq_len, hid_dim)``.
            valid_mask: Boolean mask of shape ``(batch_size, max_seq_len)``.
            attn_bias: Optional additive attention bias broadcastable to
                ``(batch_size, num_heads, seq, seq)``.
            pairwise_relation_mask: Optional boolean relation mask shaped
                ``(batch_size, seq, seq, num_relations)``.

        Returns:
            Output embeddings of shape ``(batch_size, hid_dim)``.
        """
        x = sequences * valid_mask.unsqueeze(-1).to(sequences.dtype)

        for encoder_layer in self._encoder_layers:
            x = encoder_layer(
                x,
                attn_bias=attn_bias,
                pairwise_relation_mask=pairwise_relation_mask,
                valid_mask=valid_mask,
            )

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
