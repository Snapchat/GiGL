"""Graph Transformer encoder for heterogeneous graphs.

Adapted from RelGT's LocalModule (https://github.com/snap-stanford/relgt).
Converts heterogeneous graph data into fixed-length sequences via
``heterodata_to_graph_transformer_input``, processes through a stack of pre-norm
transformer encoder layers, then produces per-node embeddings via configurable
anchor or anchor-neighbor readout.

Conforms to the same forward interface as ``HGT`` and ``SimpleHGN`` in
``gigl.src.common.models.pyg.heterogeneous``, making it a drop-in
replacement as the encoder in ``LinkPredictionGNN``.
"""

import math
from typing import Callable, Literal, Optional, Sequence, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.hetero_data
from jaxtyping import Bool, Float, Int64
from torch import Tensor
from torch_geometric.utils import softmax

from gigl.common.logger import Logger
from gigl.nn.packed_attention import PackedAttentionBias, packed_attention
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.transforms.graph_transformer import (
    HOP_DISTANCE_FEATURE_NAME,
    PPR_EXPANDED_FEATURE_NAME,
    PPR_RELATION_FEATURES_NAME,
    PPR_WEIGHT_FEATURE_NAME,
    SequenceAuxiliaryData,
    TokenInputData,
    _lookup_pairwise_relation_indices,
    _validate_reserved_anchor_feature_usage,
    concat_tensors,
    get_node_type_offsets,
    tensors_to_graph_transformer_input,
    to_homogeneous_edge_index,
)
from gigl.transforms.static_khop import (
    KhopLayoutTooLargeError,
    PackedLayout,
    PackedRelationEdges,
    pack_khop_layout,
)

logger = Logger()


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

    def forward(
        self, x: Float[Tensor, "batch sequence model_dim"]
    ) -> Float[Tensor, "batch sequence model_dim"]:
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


def _get_relation_segments_from_bounds(
    bounds: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    """Returns a ``(relation, start, end)`` segment for every relation.

    ``bounds[r]`` to ``bounds[r + 1]`` is relation ``r``'s edge range. A relation
    without edges keeps an empty segment, so the number of segments is the number
    of relations whatever the batch; no segments at all means no edges.
    """
    if bounds[-1] == 0:
        return ()
    return tuple(
        (relation, bounds[relation], bounds[relation + 1])
        for relation in range(len(bounds) - 1)
    )


def _get_relation_segments(
    relation_indices: Tensor,
) -> tuple[tuple[int, int, int], ...]:
    """Returns the ``(relation, start, end)`` runs of equal consecutive relations.

    Costs one host read.
    """
    relations, counts = torch.unique_consecutive(relation_indices, return_counts=True)
    segments: list[tuple[int, int, int]] = []
    start = 0
    for relation, count in torch.stack([relations, counts]).t().tolist():
        segments.append((relation, start, start + count))
        start += count
    return tuple(segments)


def _anchor_relation_edges(
    edges: PackedRelationEdges, layout: PackedLayout
) -> PackedRelationEdges:
    """The edges into each sequence's anchor token, as they feed a last layer that
    computes only the anchors.

    ``target`` becomes the sequence index. The edges keep their relation order, and
    ``all_filtered`` is set when ``edges`` has edges but none reach an anchor.
    """
    into_anchor = layout.position[edges.target] == 0
    relations = edges.relation[into_anchor]
    num_relations = 0
    if edges.segments:
        num_relations = edges.segments[-1][0] + 1
    bounds = torch.searchsorted(
        relations, torch.arange(num_relations + 1, device=relations.device)
    ).tolist()
    return PackedRelationEdges(
        target=layout.sequence_id[edges.target[into_anchor]],
        source=edges.source[into_anchor],
        relation=relations,
        segments=_get_relation_segments_from_bounds(bounds),
        all_filtered=edges.target.numel() > 0 and bounds[-1] == 0,
    )


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
            self-attention path. ``"edge_type_bilinear"`` adds a learned
            per-edge-type bilinear term for sampled directed graph edges. This
            changes attention weights, not value/message content.
            ``"edge_type_hgt"`` replaces the base query/key score on relation
            edges with an HGT-style relation transform and relation prior.
        relation_message_mode: Optional relation-aware message channel run in
            parallel with self-attention. ``"none"`` preserves the default
            layer. ``"edge_type_linear"`` adds, for every sampled directed graph
            edge, a per-edge-type linear transform of the source token
            (mean-aggregated per target and relation) into the attention
            residual — the message analog of HGT's per-relation value transform
            (``v_rel``) and per-edge-type GATv2 weights. ``"edge_type_attention"``
            replaces the mean with a per-head softmax over each target's
            relation neighbors, scored HGT-style from the layer's own query/key
            projections with per-(relation, head) transforms and priors —
            coupling attention weighting with relation-transformed content.
            Either way compute scales with the number of relation edges, not
            seq^2, and message weights are zero-initialized so startup behavior
            matches the plain layer.
        num_relations: Number of relation channels expected in
            ``pairwise_relation_indices`` when relation-aware attention or
            messages are enabled.

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
        relation_attention_mode: Literal[
            "none",
            "edge_type_bilinear",
            "edge_type_hgt",
        ] = "none",
        relation_message_mode: Literal[
            "none",
            "edge_type_linear",
            "edge_type_attention",
        ] = "none",
        num_relations: int = 0,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            )
        if relation_attention_mode not in {
            "none",
            "edge_type_bilinear",
            "edge_type_hgt",
        }:
            raise ValueError(
                "relation_attention_mode must be one of "
                "{'none', 'edge_type_bilinear', 'edge_type_hgt'}, "
                f"got '{relation_attention_mode}'"
            )
        if relation_attention_mode != "none" and num_relations <= 0:
            raise ValueError(
                "Relation-aware attention mode requires num_relations > 0."
            )
        if relation_message_mode not in {
            "none",
            "edge_type_linear",
            "edge_type_attention",
        }:
            raise ValueError(
                "relation_message_mode must be one of "
                "{'none', 'edge_type_linear', 'edge_type_attention'}, "
                f"got '{relation_message_mode}'"
            )
        if relation_message_mode != "none" and num_relations <= 0:
            raise ValueError("Relation-aware message mode requires num_relations > 0.")

        self._num_heads = num_heads
        self._head_dim = model_dim // num_heads
        self._attention_dropout_rate = attention_dropout_rate
        self._relation_attention_mode = relation_attention_mode
        self._relation_message_mode = relation_message_mode
        self._num_relations = num_relations

        self._attention_norm = nn.LayerNorm(model_dim)
        self._query_projection = nn.Linear(model_dim, model_dim)
        self._key_projection = nn.Linear(model_dim, model_dim)
        self._value_projection = nn.Linear(model_dim, model_dim)
        self._output_projection = nn.Linear(model_dim, model_dim)
        self._dropout = nn.Dropout(dropout_rate)
        self._relation_attention_matrices: Optional[nn.Parameter] = None
        if relation_attention_mode == "edge_type_bilinear":
            # score(target, source, relation) += q_target^T W_relation k_source
            # Zero init keeps startup behavior identical to shared attention.
            self._relation_attention_matrices = nn.Parameter(
                torch.zeros(num_relations, num_heads, self._head_dim, self._head_dim)
            )
        self._relation_hgt_attention_matrices: Optional[nn.Parameter] = None
        self._relation_hgt_attention_priors: Optional[nn.Parameter] = None
        if relation_attention_mode == "edge_type_hgt":
            # Replaces the base score on relation edges with
            # mu_relation * q_target^T W_relation k_source. Identity/one init
            # preserves the base query/key score at startup.
            self._relation_hgt_attention_matrices = nn.Parameter(
                torch.empty(num_relations, self._head_dim, self._head_dim)
            )
            self._relation_hgt_attention_priors = nn.Parameter(
                torch.empty(num_relations)
            )
            self._reset_hgt_relation_attention_parameters()
        self._relation_message_matrices: Optional[nn.Parameter] = None
        if relation_message_mode in {"edge_type_linear", "edge_type_attention"}:
            # message(target <- source, relation) = W_relation @ x_norm[source],
            # aggregated per (target, relation) and added to the attention
            # residual. This is the per-relation value/message transform of HGT
            # (``v_rel``) / per-edge-type GATv2 weights, applied sparsely over
            # sampled graph edges. Zero init keeps startup behavior identical to
            # the plain layer regardless of the aggregation weights.
            self._relation_message_matrices = nn.Parameter(
                torch.zeros(num_relations, model_dim, model_dim)
            )
        self._relation_message_attention_matrices: Optional[nn.Parameter] = None
        self._relation_message_attention_priors: Optional[nn.Parameter] = None
        if relation_message_mode == "edge_type_attention":
            # HGT-style per-(relation, head) score transform and per-head
            # relation priors for the message-channel softmax:
            # score = prior * <q_target, W_relation k_source> / sqrt(head_dim).
            # Identity/one init makes the initial scores the plain query/key
            # dot product.
            self._relation_message_attention_matrices = nn.Parameter(
                torch.empty(num_relations, num_heads, self._head_dim, self._head_dim)
            )
            self._relation_message_attention_priors = nn.Parameter(
                torch.empty(num_relations, num_heads)
            )
            self._reset_relation_message_attention_parameters()

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
            nn.init.zeros_(self._relation_attention_matrices)
        self._reset_hgt_relation_attention_parameters()
        if self._relation_message_matrices is not None:
            nn.init.zeros_(self._relation_message_matrices)
        self._reset_relation_message_attention_parameters()
        self._ffn_norm.reset_parameters()
        self._ffn.reset_parameters()

    def _reset_hgt_relation_attention_parameters(self) -> None:
        if self._relation_hgt_attention_matrices is None:
            return
        if self._relation_hgt_attention_priors is None:
            raise ValueError("Relation HGT attention priors are not initialized.")

        with torch.no_grad():
            relation_identity = torch.eye(
                self._head_dim,
                dtype=self._relation_hgt_attention_matrices.dtype,
                device=self._relation_hgt_attention_matrices.device,
            )
            self._relation_hgt_attention_matrices.copy_(
                relation_identity.expand(self._num_relations, -1, -1)
            )
            self._relation_hgt_attention_priors.fill_(1.0)

    def _reset_relation_message_attention_parameters(self) -> None:
        if self._relation_message_attention_matrices is None:
            return
        if self._relation_message_attention_priors is None:
            raise ValueError("Relation message attention priors are not initialized.")

        with torch.no_grad():
            relation_identity = torch.eye(
                self._head_dim,
                dtype=self._relation_message_attention_matrices.dtype,
                device=self._relation_message_attention_matrices.device,
            )
            self._relation_message_attention_matrices.copy_(
                relation_identity.expand(self._num_relations, self._num_heads, -1, -1)
            )
            self._relation_message_attention_priors.fill_(1.0)

    def forward(
        self,
        x: Float[Tensor, "batch sequence model_dim"],
        attn_bias: Optional[
            Union[Float[Tensor, ""], Float[Tensor, "... #sequence"]]
        ] = None,
        valid_mask: Optional[Bool[Tensor, "batch sequence"]] = None,
        pairwise_relation_indices: Optional[Int64[Tensor, "relation_edges 4"]] = None,
        query_seq_len: Optional[int] = None,
    ) -> Float[Tensor, "batch query_sequence model_dim"]:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq, model_dim)``.
            attn_bias: Optional attention bias of shape
                ``(batch, num_heads, seq, seq)`` or broadcastable.
                Added as an additive mask to attention scores.
            valid_mask: Optional boolean tensor of shape ``(batch, seq)`` used
                to zero out padded token states after each residual block.
            pairwise_relation_indices: Optional long tensor of shape
                ``(num_relation_edges, 4)`` with sparse
                ``(batch_idx, query_pos, key_pos, relation_idx)`` coordinates.
            query_seq_len: Number of leading query positions to compute. Defaults
                to the full input sequence length.

        Returns:
            Output tensor of shape ``(batch, query_seq_len, model_dim)``.
        """
        return self._forward_query_prefix(
            x=x,
            seq_len=x.size(1) if query_seq_len is None else query_seq_len,
            attn_bias=attn_bias,
            valid_mask=valid_mask,
            pairwise_relation_indices=pairwise_relation_indices,
        )

    def forward_packed(
        self,
        x: Float[Tensor, "tokens model_dim"],
        layout: PackedLayout,
        first_token_output: bool = False,
        bias: Optional[PackedAttentionBias] = None,
    ) -> Float[Tensor, "outputs model_dim"]:
        """Runs the layer on packed tokens (see ``gigl.nn.packed_attention``).

        The tokens of each sequence sit back to back in ``x``. Each token attends
        only to tokens of its own sequence. Requires ``relation_attention_mode``
        to be ``"none"``, and attention dropout to be off in training.

        Args:
            x: ``(tokens, model_dim)`` packed token states.
            layout: Layout of the sequences in ``x``. With a relation message mode,
                ``layout.relation_edges`` is required. Its ``target`` indexes the
                output rows, so with ``first_token_output`` it is the sequence
                index and the edges into later tokens are left out.
            first_token_output: Compute only each sequence's first token.
            bias: Optional additive attention bias.

        Returns:
            ``(tokens, model_dim)``, or ``(num_sequences, model_dim)`` with
            ``first_token_output``.

        Raises:
            ValueError: If ``relation_attention_mode`` is not ``"none"``, if
                ``layout.relation_edges`` is missing with a relation message mode,
                or in training with attention dropout.
        """
        if self._relation_attention_mode != "none":
            raise ValueError(
                "forward_packed supports relation_attention_mode 'none' only."
            )
        relation_edges = layout.relation_edges
        if self._relation_message_mode != "none" and relation_edges is None:
            raise ValueError("layout.relation_edges is required for relation messages.")
        if self.training and self._attention_dropout_rate > 0:
            raise ValueError(
                "Packed attention does not support attention dropout in training; "
                "set attention_dropout_rate=0 or use forward."
            )
        x_norm = self._attention_norm(x)
        key = self._key_projection(x_norm).view(-1, self._num_heads, self._head_dim)
        value = self._value_projection(x_norm).view(-1, self._num_heads, self._head_dim)
        if first_token_output:
            first_tokens = layout.cu_seqlens[:-1].long()
            residual = x[first_tokens]
            query_input = x_norm[first_tokens]
        else:
            residual = x
            query_input = x_norm
        query = self._query_projection(query_input).view(
            -1, self._num_heads, self._head_dim
        )
        attention_output = packed_attention(
            query,
            key,
            value,
            layout,
            query_is_first_token=first_token_output,
            bias=bias,
        ).reshape(-1, self._num_heads * self._head_dim)
        x = residual + self._dropout(self._output_projection(attention_output))
        if relation_edges is not None:
            if relation_edges.segments:
                uses_attention = self._relation_message_mode == "edge_type_attention"
                messages = self._aggregate_relation_messages(
                    source_features=x_norm[relation_edges.source],
                    edge_queries=(
                        query[relation_edges.target] if uses_attention else None
                    ),
                    edge_keys=key[relation_edges.source] if uses_attention else None,
                    relation_indices=relation_edges.relation,
                    relation_segments=relation_edges.segments,
                    target_rows=relation_edges.target,
                    num_target_rows=x.size(0),
                )
                x = x + self._dropout(messages)
            elif self.training and relation_edges.all_filtered:
                # Keep the relation parameters in the graph, as the padded path does
                # when anchor filtering drops every edge.
                x = x + self._zero_relation_parameter_dependency(x)
        return x + self._ffn(self._ffn_norm(x))

    def _forward_query_prefix(
        self,
        x: Tensor,
        seq_len: int,
        attn_bias: Optional[Tensor],
        valid_mask: Optional[Tensor],
        pairwise_relation_indices: Optional[Tensor],
    ) -> Tensor:
        """Compute layer outputs for a prefix of query positions.

        Keys and values retain the full sequence so every token can contribute
        to each requested query output.

        Args:
            x: Input tensor of shape ``(batch, seq, model_dim)``.
            seq_len: Number of leading sequence positions to compute.
            attn_bias: Optional attention bias broadcastable to
                ``(batch, num_heads, seq_len, input_seq_len)``.
            valid_mask: Optional boolean tensor of shape ``(batch, seq)``.
            pairwise_relation_indices: Optional sparse relation coordinates
                shaped ``(num_relation_edges, 4)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, model_dim)``.

        Raises:
            ValueError: If ``seq_len`` is outside ``[1, input_seq_len]`` or relation
                coordinates are invalid.
        """
        batch_size, input_seq_len, model_dim = x.shape
        if seq_len < 1 or seq_len > input_seq_len:
            raise ValueError(f"seq_len must be in [1, {input_seq_len}], got {seq_len}.")

        valid_mask = (
            valid_mask[:, :seq_len] if valid_mask is not None else None
        )  # [batch, seq_len] or None
        residual = x[:, :seq_len, :]  # [batch, seq_len, model_dim]

        selected_relation_indices = pairwise_relation_indices
        all_relation_edges_filtered = False
        if (
            self._relation_attention_mode != "none"
            or self._relation_message_mode != "none"
        ):
            if pairwise_relation_indices is None:
                raise ValueError(
                    "pairwise_relation_indices is required for relation-aware "
                    "attention or messages."
                )
            if (
                pairwise_relation_indices.dim() != 2
                or pairwise_relation_indices.size(-1) != 4
            ):
                raise ValueError(
                    "pairwise_relation_indices must have shape (num_relation_edges, 4)."
                )
            selected_relation_indices = pairwise_relation_indices.to(
                device=x.device,
                dtype=torch.long,
            )  # [num_relation_edges, 4]
            if selected_relation_indices.numel() > 0:
                relation_indices = selected_relation_indices[:, 3]
                if (
                    relation_indices.min().item() < 0
                    or relation_indices.max().item() >= self._num_relations
                ):
                    raise ValueError(
                        "pairwise_relation_indices contains relation ids outside "
                        f"[0, {self._num_relations})."
                    )
            if seq_len < input_seq_len:
                selected_relation_indices = selected_relation_indices[
                    selected_relation_indices[:, 1] < seq_len
                ]  # [num_selected_relation_edges, 4]
            all_relation_edges_filtered = (
                pairwise_relation_indices.numel() > 0
                and selected_relation_indices.numel() == 0
            )

        x_norm = self._attention_norm(x)  # [batch, input_seq_len, model_dim]
        query = self._query_projection(
            x_norm[:, :seq_len, :]
        )  # [batch, seq_len, model_dim]
        key = self._key_projection(x_norm)  # [batch, input_seq_len, model_dim]
        value = self._value_projection(x_norm)  # [batch, input_seq_len, model_dim]

        query = query.view(
            batch_size, seq_len, self._num_heads, self._head_dim
        ).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        key = key.view(
            batch_size,
            input_seq_len,
            self._num_heads,
            self._head_dim,
        ).transpose(1, 2)  # [batch, num_heads, input_seq_len, head_dim]
        value = value.view(
            batch_size,
            input_seq_len,
            self._num_heads,
            self._head_dim,
        ).transpose(1, 2)  # [batch, num_heads, input_seq_len, head_dim]

        if attn_bias is not None and attn_bias.dim() >= 2:
            attn_bias = attn_bias[..., :seq_len, :]  # [..., seq_len, input_seq_len]
        attention_output = self._run_attention(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
            pairwise_relation_indices=selected_relation_indices,
        )  # [batch, num_heads, seq_len, head_dim]
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, seq_len, model_dim
        )  # [batch, seq_len, model_dim]
        attention_output = self._dropout(
            self._output_projection(attention_output)
        )  # [batch, seq_len, model_dim]
        x = residual + attention_output  # [batch, seq_len, model_dim]

        if self._relation_message_mode == "edge_type_attention":
            x = x + self._dropout(
                self._compute_relation_attention_messages(
                    x_norm=x_norm,
                    query=query,
                    key=key,
                    pairwise_relation_indices=cast(
                        Tensor,
                        selected_relation_indices,
                    ),
                    batch_size=batch_size,
                    seq_len=seq_len,
                )
            )
        elif self._relation_message_mode != "none":
            x = x + self._dropout(
                self._compute_relation_messages(
                    x_norm=x_norm,
                    pairwise_relation_indices=cast(
                        Tensor,
                        selected_relation_indices,
                    ),
                    batch_size=batch_size,
                    seq_len=seq_len,
                )
            )
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)  # [batch, seq_len, model_dim]

        # The full-sequence path still touches relation parameters for edges outside
        # this prefix, producing zero gradients. Filtering all such edges would leave
        # those parameters with grad=None instead, changing DDP/optimizer behavior;
        # retain a zero-valued dependency to preserve the original training semantics.
        if self.training and all_relation_edges_filtered:
            x = x + self._zero_relation_parameter_dependency(x)

        residual = x
        x = residual + self._ffn(self._ffn_norm(x))  # [batch, seq_len, model_dim]
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)  # [batch, seq_len, model_dim]

        return x

    def _zero_relation_parameter_dependency(self, reference: Tensor) -> Tensor:
        """Keep relation parameters in the training graph after anchor filtering."""
        zero_dependency = reference.new_zeros(())
        relation_parameters = [
            self._relation_attention_matrices,
            self._relation_hgt_attention_matrices,
            self._relation_hgt_attention_priors,
            self._relation_message_matrices,
            self._relation_message_attention_matrices,
            self._relation_message_attention_priors,
        ]
        for parameter in relation_parameters:
            if parameter is not None:
                zero_dependency = zero_dependency + (
                    parameter.reshape(-1)[0].to(dtype=reference.dtype) * 0.0
                )
        return zero_dependency

    def _run_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias: Optional[Tensor],
        pairwise_relation_indices: Optional[Tensor],
    ) -> Tensor:
        if self._relation_attention_mode in {"edge_type_bilinear", "edge_type_hgt"}:
            attn_bias = self._add_relation_attention_bias(
                attn_bias=attn_bias,
                query=query,
                key=key,
                pairwise_relation_indices=cast(
                    Tensor,
                    pairwise_relation_indices,
                ),
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
        pairwise_relation_indices: Tensor,
    ) -> Optional[Tensor]:
        if pairwise_relation_indices.numel() == 0:
            return None

        batch_size, _, query_len, _ = query.shape
        key_len = key.size(2)
        empty_bias = query.new_zeros((batch_size, self._num_heads, query_len, key_len))
        return self._add_relation_attention_bias(
            attn_bias=empty_bias,
            query=query,
            key=key,
            pairwise_relation_indices=pairwise_relation_indices,
        )

    def _add_relation_attention_bias(
        self,
        attn_bias: Optional[Tensor],
        query: Tensor,
        key: Tensor,
        pairwise_relation_indices: Tensor,
    ) -> Optional[Tensor]:
        batch_size, _, query_len, _ = query.shape
        key_len = key.size(2)
        if pairwise_relation_indices.numel() == 0:
            return attn_bias
        batch_indices = pairwise_relation_indices[:, 0]
        query_indices = pairwise_relation_indices[:, 1]
        key_indices = pairwise_relation_indices[:, 2]
        relation_indices = pairwise_relation_indices[:, 3]

        bias_shape = (self._num_heads, query_len, key_len)
        if attn_bias is None:
            attn_bias = query.new_zeros((batch_size, *bias_shape))
        elif attn_bias.shape[1:] != bias_shape:
            attn_bias = attn_bias.expand(
                batch_size,
                *bias_shape,
            ).clone()
        else:
            attn_bias = attn_bias.clone()

        attn_bias_by_position = attn_bias.permute(0, 2, 3, 1)
        query_by_position = query.transpose(1, 2)
        key_by_position = key.transpose(1, 2)

        unique_relation_indices, relation_counts = torch.unique_consecutive(
            relation_indices,
            return_counts=True,
        )
        relation_start = 0
        for relation_idx_tensor, relation_count_tensor in zip(
            unique_relation_indices,
            relation_counts,
        ):
            relation_idx = int(relation_idx_tensor.item())
            relation_end = relation_start + int(relation_count_tensor.item())
            relation_batch_indices = batch_indices[relation_start:relation_end]
            relation_query_indices = query_indices[relation_start:relation_end]
            relation_key_indices = key_indices[relation_start:relation_end]

            selected_query = query_by_position[
                relation_batch_indices,
                relation_query_indices,
            ]
            selected_key = key_by_position[
                relation_batch_indices,
                relation_key_indices,
            ]
            relation_scores = self._compute_relation_attention_scores(
                selected_query=selected_query,
                selected_key=selected_key,
                relation_idx=relation_idx,
            )
            attn_bias_by_position.index_put_(
                (
                    relation_batch_indices,
                    relation_query_indices,
                    relation_key_indices,
                ),
                relation_scores.to(dtype=attn_bias.dtype),
                accumulate=True,
            )
            relation_start = relation_end

        return attn_bias

    def _compute_relation_attention_scores(
        self,
        selected_query: Tensor,
        selected_key: Tensor,
        relation_idx: int,
    ) -> Tensor:
        if self._relation_attention_mode == "edge_type_bilinear":
            if self._relation_attention_matrices is None:
                raise ValueError("Relation attention matrices are not initialized.")
            relation_matrices = self._relation_attention_matrices.to(
                dtype=selected_query.dtype
            )
            relation_scores = torch.einsum(
                "ehd,hdf,ehf->eh",
                selected_query,
                relation_matrices[relation_idx],
                selected_key,
            )
        elif self._relation_attention_mode == "edge_type_hgt":
            if self._relation_hgt_attention_matrices is None:
                raise ValueError("Relation HGT attention matrices are not initialized.")
            if self._relation_hgt_attention_priors is None:
                raise ValueError("Relation HGT attention priors are not initialized.")
            relation_matrices = self._relation_hgt_attention_matrices.to(
                dtype=selected_query.dtype
            )
            relation_priors = self._relation_hgt_attention_priors.to(
                dtype=selected_query.dtype
            )
            hgt_scores = torch.einsum(
                "ehd,df,ehf->eh",
                selected_query,
                relation_matrices[relation_idx],
                selected_key,
            )
            base_scores = (selected_query * selected_key).sum(dim=-1)
            relation_scores = relation_priors[relation_idx] * hgt_scores - base_scores
        else:
            raise ValueError(
                "Relation attention scores requested when relation_attention_mode "
                f"is '{self._relation_attention_mode}'."
            )
        return relation_scores / math.sqrt(self._head_dim)

    def _compute_relation_messages(
        self,
        x_norm: Tensor,
        pairwise_relation_indices: Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tensor:
        """Aggregate per-relation linear messages over sampled directed edges.

        For every relation edge ``(batch_idx, query_pos=target, key_pos=source,
        relation_idx)`` the target token receives
        ``W_relation @ x_norm[source]``, mean-aggregated per (target, relation)
        and summed across relations.

        Compute scales with the number of relation edges (not seq^2) and never
        materializes any (seq, seq) tensor, so the scaled-dot-product attention
        path and its memory profile are unaffected.

        Returns:
            Message tensor of shape ``(batch_size, seq_len, model_dim)``.
        """
        if self._relation_message_matrices is None:
            raise ValueError("Relation message matrices are not initialized.")
        if pairwise_relation_indices.numel() == 0:
            return x_norm.new_zeros((batch_size, seq_len, x_norm.size(-1)))
        batch_indices, target_indices, source_indices, relation_indices = (
            pairwise_relation_indices.unbind(dim=1)
        )
        return self._aggregate_relation_messages(
            source_features=x_norm[batch_indices, source_indices],
            edge_queries=None,
            edge_keys=None,
            relation_indices=relation_indices,
            relation_segments=_get_relation_segments(relation_indices),
            target_rows=batch_indices * seq_len + target_indices,
            num_target_rows=batch_size * seq_len,
        ).view(batch_size, seq_len, -1)

    def _compute_relation_attention_messages(
        self,
        x_norm: Tensor,
        query: Tensor,
        key: Tensor,
        pairwise_relation_indices: Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tensor:
        """Aggregate per-relation messages weighted by renormalized attention.

        Like ``_compute_relation_messages`` but each edge is weighted by a
        per-head softmax over the target's relation-``r`` neighbors instead of a
        uniform mean, with HGT-style scores
        ``prior * <q_target, W_relation k_source> / sqrt(head_dim)`` computed
        from the layer's own query/key projections. This couples attention
        weighting with relation-transformed message content (HGT's ``v_rel``
        semantics) while keeping every tensor edge-sparse: no (seq, seq)
        buffer is ever materialized.

        Args:
            x_norm: Pre-norm token states of shape
                ``(batch_size, seq_len, model_dim)``.
            query: Projected queries of shape
                ``(batch_size, num_heads, seq_len, head_dim)``.
            key: Projected keys of the same shape as ``query``.
            pairwise_relation_indices: Sparse relation coordinates shaped
                ``(num_relation_edges, 4)`` storing
                ``(batch_idx, query_pos=target, key_pos=source, relation_idx)``.
            batch_size: Number of sequences in the batch.
            seq_len: Sequence length.

        Returns:
            Message tensor of shape ``(batch_size, seq_len, model_dim)``.
        """
        if pairwise_relation_indices.numel() == 0:
            return x_norm.new_zeros((batch_size, seq_len, x_norm.size(-1)))
        batch_indices, target_indices, source_indices, relation_indices = (
            pairwise_relation_indices.unbind(dim=1)
        )
        return self._aggregate_relation_messages(
            source_features=x_norm[batch_indices, source_indices],
            edge_queries=query[batch_indices, :, target_indices],
            edge_keys=key[batch_indices, :, source_indices],
            relation_indices=relation_indices,
            relation_segments=_get_relation_segments(relation_indices),
            target_rows=batch_indices * seq_len + target_indices,
            num_target_rows=batch_size * seq_len,
        ).view(batch_size, seq_len, -1)

    def _aggregate_relation_messages(
        self,
        source_features: Tensor,
        edge_queries: Optional[Tensor],
        edge_keys: Optional[Tensor],
        relation_indices: Tensor,
        relation_segments: Sequence[tuple[int, int, int]],
        target_rows: Tensor,
        num_target_rows: int,
    ) -> Tensor:
        """Sums each target's per-relation messages from its incoming edges.

        This is the layout-independent core of ``_compute_relation_messages`` and
        ``_compute_relation_attention_messages`` and of the packed path. Edge
        ``e`` sends ``W_relation[e] @ source_features[e]``, weighted per head by
        a softmax of the relation-transformed query/key scores over the target's
        edges of that relation (``edge_queries`` and ``edge_keys`` given) or by
        the mean over them (both ``None``).

        Args:
            source_features: ``(num_edges, model_dim)`` source token states.
            edge_queries: ``(num_edges, num_heads, head_dim)`` target queries,
                or ``None`` for mean aggregation.
            edge_keys: ``(num_edges, num_heads, head_dim)`` source keys, or
                ``None`` for mean aggregation.
            relation_indices: ``(num_edges,)`` relation of each edge.
            relation_segments: ``(relation, start, end)`` edge ranges covering
                ``[0, num_edges)`` in order, empty ranges allowed; the
                per-relation transforms run on these slices.
            target_rows: ``(num_edges,)`` output row of each edge's target.
            num_target_rows: Number of output rows.

        Returns:
            ``(num_target_rows, model_dim)`` messages; zero rows for targets
            without edges.
        """
        if self._relation_message_matrices is None:
            raise ValueError("Relation message matrices are not initialized.")
        num_edges, model_dim = source_features.shape
        # One group per (relation, target) and head.
        group_indices = relation_indices * num_target_rows + target_rows
        head_group_indices = group_indices.unsqueeze(-1).expand(-1, self._num_heads)
        num_groups = self._num_relations * num_target_rows

        if edge_queries is None or edge_keys is None:
            group_degrees = source_features.new_zeros(num_groups).index_add_(
                0, group_indices, source_features.new_ones(num_edges)
            )
            edge_weights = (1.0 / group_degrees[group_indices]).unsqueeze(-1)
        else:
            if self._relation_message_attention_matrices is None:
                raise ValueError(
                    "Relation message attention matrices are not initialized."
                )
            if self._relation_message_attention_priors is None:
                raise ValueError(
                    "Relation message attention priors are not initialized."
                )
            relation_attention_matrices = self._relation_message_attention_matrices.to(
                dtype=edge_queries.dtype
            )
            relation_attention_priors = self._relation_message_attention_priors.to(
                dtype=edge_queries.dtype
            )
            scores = torch.cat(
                [
                    (
                        edge_queries[start:end]
                        * torch.einsum(
                            "ehd,hdf->ehf",
                            edge_keys[start:end],
                            relation_attention_matrices[relation_idx],
                        )
                    ).sum(dim=-1)
                    * relation_attention_priors[relation_idx]
                    / math.sqrt(self._head_dim)
                    for relation_idx, start, end in relation_segments
                ],
                dim=0,
            )  # (num_edges, num_heads)

            # Numerically stable scatter-softmax per group and head. Group buffers
            # are (num_relations * num_target_rows, num_heads) -- tiny compared
            # to any (seq, seq) tensor.
            group_score_max = scores.new_full(
                (num_groups, self._num_heads), torch.finfo(scores.dtype).min
            )
            group_score_max.scatter_reduce_(
                0, head_group_indices, scores.detach(), reduce="amax", include_self=True
            )
            normalized_scores = torch.exp(
                scores - group_score_max.gather(0, head_group_indices)
            )
            group_denominators = scores.new_zeros(
                (num_groups, self._num_heads)
            ).scatter_add(0, head_group_indices, normalized_scores)
            # Sums to 1 per (relation, target) and head.
            edge_weights = normalized_scores / group_denominators.gather(
                0, head_group_indices
            )

        relation_matrices = self._relation_message_matrices.to(
            dtype=source_features.dtype
        )
        edge_messages = torch.cat(
            [
                source_features[start:end] @ relation_matrices[relation_idx].t()
                for relation_idx, start, end in relation_segments
            ],
            dim=0,
        )  # (num_edges, model_dim)
        weighted_messages = (
            edge_messages.view(num_edges, -1, model_dim // self._num_heads)
            * edge_weights.unsqueeze(-1)
        ).reshape(num_edges, model_dim)
        messages = source_features.new_zeros((num_target_rows, model_dim))
        return messages.index_put_(
            (target_rows,), weighted_messages.to(dtype=messages.dtype), accumulate=True
        )


class GraphTransformerEncoder(nn.Module):
    """Graph Transformer encoder for heterogeneous graphs.

    Converts heterogeneous graph data into fixed-length sequences via
    ``heterodata_to_graph_transformer_input``, processes through pre-norm
    transformer encoder layers, and produces per-node embeddings via configurable
    anchor or anchor-neighbor readout.

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
        readout_mode: Strategy used to turn encoded sequence tokens into one
            embedding per anchor. ``"anchor_neighbor_attention"`` preserves the
            existing behavior by adding the encoded anchor token to an
            attention-weighted neighbor aggregation. ``"anchor_only"`` returns
            only the encoded anchor token. Both run on the packed backend, but
            ``"anchor_only"`` lets the last layer skip the neighbour tokens, so it
            is cheaper there.
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
            The padded path reads the ``hop_distance`` attribute that
            ``AddHeteroHopDistanceEncoding`` stored on ``data`` and raises if it
            is missing. The packed path computes the hop from the sequence layout
            instead, so it runs without the attribute. A batch whose k-hop layout
            is too large falls back to the padded path, though, and that batch
            raises if the attribute is missing. Add the transform to the pipeline
            (``is_undirected=False``, ``sampling_direction`` equal to this
            encoder's, ``h_max`` at least this encoder's ``hop_distance``) unless
            every batch is sure to fit the layout budget. When the attribute is
            present, it must match this encoder's ``hop_distance`` and direction,
            or the two paths disagree. An embedding for ``hop_distance`` needs more
            than ``hop_distance`` entries on a config that can run packed.
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
        sampling_direction: Direction used for sequence token construction.
            ``"out"`` preserves the existing k-hop reachability expansion.
            ``"in"`` expands over reversed edges and is supported only
            when ``sequence_construction_method="khop"``. Directed relative
            encodings such as ``"hop_distance"`` should be computed with the
            same direction.
        prioritize_hop_order: If True, k-hop sequence construction fills tokens
            by hop depth before truncating to ``max_seq_len``. The default False
            preserves existing homogeneous-node-id ordering.
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
            attention scores. ``"none"`` preserves the current transformer path.
            ``"edge_type_bilinear"`` adds a learned per-edge-type bilinear score
            term for sampled directed edges. ``"edge_type_hgt"`` replaces base
            query/key scores on relation edges with an HGT-style relation
            transform and relation prior. Both change attention weights only,
            not value/message content.
        relation_message_mode: Optional relation-aware message channel run in
            parallel with self-attention in every layer. ``"none"`` preserves
            the current path. ``"edge_type_linear"`` adds, for every sampled
            directed graph edge, a per-edge-type linear transform of the source
            token (mean-aggregated per target and relation) into the attention
            residual — the message analog of HGT's per-relation value transform
            (``v_rel``) and of per-edge-type GATv2 weights.
            ``"edge_type_attention"`` replaces the mean with a per-head softmax
            over each target's relation neighbors, scored HGT-style from each
            layer's own query/key projections with per-(relation, head)
            transforms and priors — the full coupled HGT message semantics.
            Either way compute scales with the number of relation edges (never
            seq^2) and message weights are zero-initialized, so startup behavior
            matches ``"none"``. Composes with ``relation_attention_mode``, and
            with ``relation_attention_mode="none"`` keeps the dense attention
            bias broadcastable (no per-layer (seq, seq) bias materialization).

    Notes:
        Packed backend: on CUDA, the encoder runs on packed tokens (one
        ``[tokens, hid_dim]`` buffer and a varlen attention kernel, so no FLOPs go
        to padding) when the config meets all of these:
          - ``sequence_construction_method="khop"``
          - ``relation_attention_mode`` and ``relation_message_mode`` are ``"none"``
          - no attention biases
          - ``"hop_distance"`` is the only anchor-based input, if any
          - ``hid_dim // num_heads`` is divisible by 8
          - ``hop_distance <= 4``
          - ``attention_dropout_rate == 0``

        Every other config, and every CPU run, uses padded sequences. A batch whose
        k-hop candidate set is too large for the packed layout also runs padded;
        these batches are counted, and a warning is logged on the 1st, 2nd, 4th,
        ... occurrence. The reasons a config cannot be packed are logged at
        construction.

        Both paths give the same embeddings and gradients up to float error. With
        ``dropout_rate > 0`` the train-mode dropout masks differ between paths, but
        follow the same distribution.

        The packed path computes ``hop_distance`` from the layout (the first-reach
        directed hop in ``sampling_direction``) and ignores the batch attribute.
        That equals ``AddHeteroHopDistanceEncoding(is_undirected=False,
        sampling_direction=<same>, h_max >= hop_distance)``. The packed path costs
        two host reads per batch.

        This encoder uses ``nn.LazyLinear`` for node-level PE fusion. If you wrap
        it with ``DistributedDataParallel``, run one representative no-grad
        forward first, passing ``anchor_node_ids``/``anchor_node_type`` for the
        graph-transformer path, or load a checkpoint before DDP so all ranks see
        initialized weights.

        TODO: Pairwise relative bias is currently materialized densely for the selected
            sequence. That is fine for moderate ``max_seq_len``, but a chunked or
            sparse LPFormer-style path is still future work for larger sequences.

    Example:
        >>> from gigl.nn.graph_transformer import GraphTransformerEncoder
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
        readout_mode: Literal[
            "anchor_neighbor_attention",
            "anchor_only",
        ] = "anchor_neighbor_attention",
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        should_l2_normalize_embedding_layer_output: bool = False,
        pe_attr_names: Optional[list[str]] = None,
        anchor_based_attention_bias_attr_names: Optional[list[str]] = None,
        anchor_based_input_attr_names: Optional[list[str]] = None,
        anchor_based_input_embedding_dict: Optional[nn.ModuleDict] = None,
        pairwise_attention_bias_attr_names: Optional[list[str]] = None,
        sampling_direction: Literal["in", "out"] = "out",
        prioritize_hop_order: bool = False,
        feature_embedding_layer_dict: Optional[nn.ModuleDict] = None,
        pe_integration_mode: Literal["concat", "add"] = "concat",
        activation: str = "gelu",
        feedforward_ratio: Optional[float] = None,
        relation_attention_mode: Literal[
            "none",
            "edge_type_bilinear",
            "edge_type_hgt",
        ] = "none",
        relation_message_mode: Literal[
            "none",
            "edge_type_linear",
            "edge_type_attention",
        ] = "none",
        **kwargs: object,
    ) -> None:
        super().__init__()
        del kwargs

        if pe_integration_mode not in {"concat", "add"}:
            raise ValueError(
                "pe_integration_mode must be one of {'concat', 'add'}, "
                f"got '{pe_integration_mode}'"
            )
        if readout_mode not in {"anchor_neighbor_attention", "anchor_only"}:
            raise ValueError(
                "readout_mode must be one of "
                "{'anchor_neighbor_attention', 'anchor_only'}, "
                f"got '{readout_mode}'"
            )

        if max_seq_len < 1:
            raise ValueError(f"max_seq_len must be at least 1, got {max_seq_len}")
        if hop_distance < 0:
            raise ValueError(f"hop_distance must be non-negative, got {hop_distance}")

        self._hid_dim = hid_dim
        self._out_dim = out_dim
        self._max_seq_len = max_seq_len
        self._hop_distance = hop_distance
        self._readout_mode = readout_mode
        if sequence_construction_method not in {"khop", "ppr"}:
            raise ValueError(
                "sequence_construction_method must be one of {'khop', 'ppr'}, "
                f"got '{sequence_construction_method}'"
            )
        if sampling_direction not in {"in", "out"}:
            raise ValueError(
                "sampling_direction must be one of {'in', 'out'}, "
                f"got '{sampling_direction}'"
            )
        if sequence_construction_method == "ppr" and sampling_direction != "out":
            raise ValueError(
                "sequence_construction_method='ppr' supports only "
                "sampling_direction='out'."
            )
        if prioritize_hop_order and sequence_construction_method != "khop":
            raise ValueError(
                "prioritize_hop_order=True is supported only with "
                "sequence_construction_method='khop'."
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
        if relation_attention_mode not in {
            "none",
            "edge_type_bilinear",
            "edge_type_hgt",
        }:
            raise ValueError(
                "relation_attention_mode must be one of "
                "{'none', 'edge_type_bilinear', 'edge_type_hgt'}, "
                f"got '{relation_attention_mode}'"
            )
        if relation_message_mode not in {
            "none",
            "edge_type_linear",
            "edge_type_attention",
        }:
            raise ValueError(
                "relation_message_mode must be one of "
                "{'none', 'edge_type_linear', 'edge_type_attention'}, "
                f"got '{relation_message_mode}'"
            )
        # Config loaders may pass OmegaConf ListConfig objects, which TorchDynamo cannot
        # trace. Plain lists iterate identically and keep the forward path traceable.
        pe_attr_names = list(pe_attr_names or ())
        anchor_based_attention_bias_attr_names = list(
            anchor_based_attention_bias_attr_names or ()
        )
        anchor_based_input_attr_names = list(anchor_based_input_attr_names or ())
        pairwise_attention_bias_attr_names = list(
            pairwise_attention_bias_attr_names or ()
        )
        anchor_bias_attr_names = anchor_based_attention_bias_attr_names
        anchor_input_attr_names = anchor_based_input_attr_names
        pairwise_bias_attr_names = pairwise_attention_bias_attr_names
        _validate_reserved_anchor_feature_usage(
            pairwise_bias_attr_names=pairwise_bias_attr_names,
            anchor_bias_attr_names=anchor_bias_attr_names,
            anchor_input_attr_names=anchor_input_attr_names,
            sequence_construction_method=sequence_construction_method,
        )
        self._sequence_construction_method = sequence_construction_method
        self._sampling_direction = sampling_direction
        self._prioritize_hop_order = prioritize_hop_order
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
        self._relation_message_mode = relation_message_mode
        self._edge_type_to_feat_dim_map = {
            edge_type: edge_type_to_feat_dim_map[edge_type]
            for edge_type in sorted(edge_type_to_feat_dim_map.keys())
        }
        # Relation vocabulary shared by relation-aware attention and messages;
        # each output relation index corresponds to one edge type in this list.
        self._relation_attention_edge_types = (
            list(self._edge_type_to_feat_dim_map.keys())
            if relation_attention_mode in {"edge_type_bilinear", "edge_type_hgt"}
            or relation_message_mode != "none"
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
        self._pairwise_nonmissing_attention_bias: Optional[nn.Parameter] = None
        if self._pairwise_attention_bias_attr_names:
            self._pairwise_pe_attention_bias_projection = nn.Linear(
                len(self._pairwise_attention_bias_attr_names),
                num_heads,
                bias=False,
            )
            self._pairwise_nonmissing_attention_bias = nn.Parameter(
                torch.zeros(num_heads)
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
                    relation_message_mode=relation_message_mode,
                    num_relations=len(self._relation_attention_edge_types),
                )
                for _ in range(num_layers)
            ]
        )

        self._final_norm = nn.LayerNorm(hid_dim)

        # Readout attention: projects concatenated (anchor, neighbor) to score
        self._readout_attention: Optional[nn.Linear] = None
        if self._readout_mode == "anchor_neighbor_attention":
            self._readout_attention = nn.Linear(2 * hid_dim, 1)

        # Output projection: hid_dim -> out_dim
        self._output_projection = nn.Linear(hid_dim, out_dim)

        self._attention_dropout_rate = attention_dropout_rate
        blockers = self._packed_backend_blockers()
        self._supports_packed_backend = not blockers
        if blockers:
            logger.info(
                "GraphTransformerEncoder will run on padded sequences, not packed "
                "tokens, because: " + "; ".join(blockers)
            )
        self._packed_fallback_count = 0
        # The packed path looks up hops 0 to ``hop_distance`` in the embedding. The
        # padded path only looks up the values stored on the batch, so only configs
        # that can run packed are checked.
        if (
            self._supports_packed_backend
            and self._anchor_based_input_embedding_dict is not None
            and HOP_DISTANCE_FEATURE_NAME in self._anchor_based_input_embedding_dict
        ):
            num_embeddings = getattr(
                self._anchor_based_input_embedding_dict[HOP_DISTANCE_FEATURE_NAME],
                "num_embeddings",
                None,
            )
            if num_embeddings is not None and num_embeddings <= self._hop_distance:
                raise ValueError(
                    f"The '{HOP_DISTANCE_FEATURE_NAME}' embedding has {num_embeddings} "
                    f"entries, but hop_distance={self._hop_distance} needs at least "
                    f"{self._hop_distance + 1} (hops 0 to {self._hop_distance})."
                )

    def _packed_backend_blockers(self) -> list[str]:
        """Lists why this config cannot run on packed tokens; empty if it can."""
        blockers: list[str] = []
        if self._sequence_construction_method != "khop":
            blockers.append("sequence_construction_method is not 'khop'")
        if self._relation_attention_mode != "none":
            blockers.append("relation_attention_mode is not 'none'")
        if self._anchor_based_attention_bias_attr_names:
            blockers.append("anchor-based attention biases are set")
        if self._pairwise_attention_bias_attr_names:
            blockers.append("pairwise attention biases are set")
        if not set(self._anchor_based_input_attr_names) <= {HOP_DISTANCE_FEATURE_NAME}:
            blockers.append("anchor-based inputs other than hop_distance are set")
        # The CUDA memory-efficient varlen kernel only has aligned variants for
        # head dims divisible by 8. For other dims, the padded path's SDPA picks
        # a different kernel.
        if (self._hid_dim // self._num_heads) % 8 != 0:
            blockers.append("head dim (hid_dim // num_heads) is not divisible by 8")
        # Packed memory is O(anchors * width ** hop_distance), and the candidate
        # budget sends batches over it to the padded path. Past four hops nearly
        # every batch would exceed it.
        if self._hop_distance > 4:
            blockers.append("hop_distance is above 4")
        # The varlen kernel's dropout has not been tested against the padded
        # path's per-head, per-sequence masks. Treating dropout as a blocker keeps
        # training and eval on the same path.
        if self._attention_dropout_rate > 0:
            blockers.append("attention_dropout_rate is above 0")
        return blockers

    def _use_packed_backend(self, device: torch.device) -> bool:
        """Returns whether this forward runs on packed tokens (supported config, on CUDA)."""
        return self._supports_packed_backend and device.type == "cuda"

    def forward(
        self,
        data: torch_geometric.data.hetero_data.HeteroData,
        anchor_node_type: Optional[NodeType] = None,
        anchor_node_ids: Optional[Int64[Tensor, "anchors"]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Float[torch.Tensor, "anchors output_dim"]:
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
            device = data[data.node_types[0]].x.device
        else:
            device = torch.device(device)

        # Use first node type if not specified
        if anchor_node_type is None:
            anchor_node_type = list(data.node_types)[0]

        # 0. Apply feature embedding if provided (without modifying original data)
        # 1. Project all node features to hid_dim
        # Build a new x_dict with processed features to avoid in-place
        # modifications. ``data.x_dict`` is avoided: it resolves through a regex
        # in ``HeteroData.__getattr__``, and ``torch.compile`` guards on the
        # global ``re`` cache, which other code mutates.
        projected_x_dict: dict[NodeType, torch.Tensor] = {}
        for node_type in data.node_types:
            x_processed = data[node_type].x.to(device)
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

        # Plain dicts of the batch's edges, in the batch's edge type order.
        edge_index_dict: dict[tuple[str, str, str], Tensor] = {}
        edge_attr_dict: dict[tuple[str, str, str], Tensor] = {}
        ppr_is_expanded_dict: dict[tuple[str, str, str], Tensor] = {}
        for edge_type in data.edge_types:
            edge_store = data[edge_type]
            if "edge_index" not in edge_store:
                continue
            edge_index_dict[edge_type] = edge_store.edge_index
            if getattr(edge_store, "edge_attr", None) is not None:
                edge_attr_dict[edge_type] = edge_store.edge_attr
            # PPR expansion provenance lets the relation filter drop a
            # co-seeded positive's leaked edges.
            if getattr(edge_store, PPR_EXPANDED_FEATURE_NAME, None) is not None:
                ppr_is_expanded_dict[edge_type] = getattr(
                    edge_store, PPR_EXPANDED_FEATURE_NAME
                )
        # 2. Build sequences and run transformer
        # If anchor_node_ids provided, use those; otherwise use first batch_size nodes
        if anchor_node_ids is not None:
            num_anchor_nodes = anchor_node_ids.size(0)
        else:
            anchor_store = data[anchor_node_type]
            if hasattr(anchor_store, "batch_size"):
                num_anchor_nodes = anchor_store.batch_size
            else:
                num_anchor_nodes = projected_x_dict[anchor_node_type].size(0)

        if num_anchor_nodes > 0 and self._use_packed_backend(device):
            packed_embeddings = self._encode_packed(
                projected_x_dict=projected_x_dict,
                edge_index_dict=edge_index_dict,
                anchor_node_type=anchor_node_type,
                num_anchor_nodes=num_anchor_nodes,
                anchor_node_ids=anchor_node_ids,
                device=device,
            )
            if packed_embeddings is not None:
                return packed_embeddings
            if (
                HOP_DISTANCE_FEATURE_NAME in self._anchor_based_input_attr_names
                and not hasattr(data, HOP_DISTANCE_FEATURE_NAME)
            ):
                raise ValueError(
                    "This batch's k-hop layout is too large for the packed path, "
                    "so it runs on padded sequences, which read the "
                    f"'{HOP_DISTANCE_FEATURE_NAME}' attribute, and the batch has "
                    "none. Add AddHeteroHopDistanceEncoding(is_undirected=False, "
                    f"sampling_direction='{self._sampling_direction}', "
                    f"h_max>={self._hop_distance}) to the pipeline, or sample fewer "
                    "neighbours per node so every batch fits the layout budget "
                    "(gigl.transforms.static_khop.MAX_LAYOUT_CANDIDATES)."
                )

        # Relative-encoding attributes (e.g., hop_distance stored as sparse matrix)
        relative_pe_attr_names = {
            attr_name
            for attr_name in (self._anchor_based_attention_bias_attr_names or [])
            if attr_name != PPR_WEIGHT_FEATURE_NAME
        }
        relative_pe_attr_names.update(self._anchor_based_input_attr_names or [])
        relative_pe_attr_names.update(self._pairwise_attention_bias_attr_names or [])
        relative_pe_attr_names.discard(PPR_WEIGHT_FEATURE_NAME)
        relative_pe_attr_names.discard(PPR_RELATION_FEATURES_NAME)
        graph_attr_dict = {
            attr_name: getattr(data, attr_name)
            for attr_name in sorted(relative_pe_attr_names)
            if hasattr(data, attr_name)
        }

        (
            sequences,
            valid_mask,
            sequence_auxiliary_data,
        ) = tensors_to_graph_transformer_input(
            x_dict=projected_x_dict,
            edge_index_dict=edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            graph_attr_dict=graph_attr_dict,
            ppr_is_expanded_dict=ppr_is_expanded_dict,
            # Typed-PPR per-edge-type channel masks drive the relation filter.
            # Check membership first: a missing-attribute lookup on HeteroData
            # runs a regex, which torch.compile guards on.
            ppr_relation_channel_masks=(
                data.ppr_relation_channel_masks
                if "ppr_relation_channel_masks" in data
                else None
            ),
            batch_size=num_anchor_nodes,
            max_seq_len=self._max_seq_len,
            anchor_node_type=anchor_node_type,
            anchor_node_ids=anchor_node_ids,
            hop_distance=self._hop_distance,
            sequence_construction_method=self._sequence_construction_method,
            sampling_direction=self._sampling_direction,
            prioritize_hop_order=self._prioritize_hop_order,
            anchor_based_attention_bias_attr_names=self._anchor_based_attention_bias_attr_names,
            anchor_based_input_attr_names=self._anchor_based_input_attr_names,
            pairwise_attention_bias_attr_names=self._pairwise_attention_bias_attr_names,
            relation_edge_types=(
                self._relation_attention_edge_types
                if self._relation_attention_mode
                in {"edge_type_bilinear", "edge_type_hgt"}
                or self._relation_message_mode != "none"
                else None
            ),
        )

        # Free memory after sequences are built
        del projected_x_dict

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
            pairwise_relation_indices=sequence_auxiliary_data[
                "pairwise_relation_indices"
            ],
        )
        return self._project_output(embeddings)

    def _project_output(self, embeddings: Tensor) -> Tensor:
        """Projects encoder outputs to ``out_dim`` and optionally L2-normalizes."""
        embeddings = self._output_projection(embeddings)
        if self._should_l2_normalize_embedding_layer_output:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    @torch.compiler.disable
    def _record_packed_fallback(self, error: KhopLayoutTooLargeError) -> None:
        """Counts a batch that fell back to padded sequences; warns on the 1st, 2nd, 4th, ...

        Excluded from ``torch.compile``: Dynamo guards on the value of a Python
        counter, so compiling this would recompile every time the count changes.
        """
        self._packed_fallback_count += 1
        count = self._packed_fallback_count
        if count & (count - 1) == 0:
            logger.warning(
                f"Running a batch on padded sequences ({count} so far) because "
                f"the packed k-hop layout is too large: {error}"
            )

    def _encode_packed(
        self,
        projected_x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
        anchor_node_type: NodeType,
        num_anchor_nodes: int,
        anchor_node_ids: Optional[Tensor],
        device: torch.device,
    ) -> Optional[Tensor]:
        """Encodes the anchors on packed tokens.

        ``_build_packed_layout`` calls ``pack_khop_layout``
        (``gigl.transforms.static_khop``), which decides which nodes each anchor
        sees and packs them into one flat token buffer described by a
        ``PackedLayout``. Packed attention (``gigl.nn.packed_attention``) then
        keeps every anchor's tokens apart using only ``cu_seqlens``.

        Worked example, run on CPU. A graph of 6 nodes with edges
        ``0->1, 0->2, 1->3, 2->3, 3->4, 4->5``, sampling direction ``"out"``,
        anchors ``[0, 3]``, ``hop_distance=2``, ``max_seq_len=4``.

        Each anchor's sequence is the anchor, then its neighbours in ascending
        node id. Anchor 0 sees node 0 (hop 0), nodes 1 and 2 (hop 1) and node 3
        (hop 2). Anchor 3 sees node 4 in one hop and node 5 in two, so its fourth
        grid cell is padding::

            >>> from gigl.transforms.static_khop import pack_khop_layout
            >>> edge_index = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 3, 3, 4, 5]])
            >>> layout = pack_khop_layout(
            ...     edge_index,
            ...     num_nodes=6,
            ...     anchor_indices=torch.tensor([0, 3]),
            ...     hop_distance=2,
            ...     max_seq_len=4,
            ... )
            >>> layout.cu_seqlens.tolist(), layout.max_seq_len
            ([0, 4, 7], 4)
            >>> layout.node_index.tolist()
            [0, 1, 2, 3, 3, 4, 5]
            >>> layout.token_attrs["hop_distance"].tolist()
            [0, 1, 1, 2, 0, 1, 2]
            >>> layout.sequence_id.tolist()
            [0, 0, 0, 0, 1, 1, 1]
            >>> layout.position.tolist()
            [0, 1, 2, 3, 0, 1, 2]

        ``flat_index`` selects the valid cells of the flattened
        ``[num_anchors, max_seq_len]`` grid. With 8 grid cells and 7 valid ones it
        is ``[0, 1, 2, 3, 4, 5, 6]``, and the cell ``[1, 3]`` (index 7) is the only
        one dropped. In a real batch most of the grid is padding.

        The token buffer ``x = node_features[layout.node_index]`` has shape
        ``[7, hidden]``. Rows 0 to 3 are anchor 0's tokens and rows 4 to 6 are
        anchor 3's. The hop embedding is added per token from
        ``token_attrs["hop_distance"]``.

        Every layer then runs on ``[num_tokens, hidden]``. The norm, the q/k/v
        projections, the output projection and the FFN act on each row alone, so
        they never see sequence boundaries. Attention gets
        ``cu_seqlens=[0, 4, 7]`` and calls the varlen kernel, so rows 0 to 3 only
        attend to each other and rows 4 to 6 only to each other.

        The first token of each sequence is the anchor, at row
        ``cu_seqlens[:-1]``, here rows ``[0, 4]``. The last layer needs only those
        rows as queries. It projects just those two rows and passes
        ``cu_seqlens_q=arange(num_sequences + 1)`` (``[0, 1, 2]``) with one query
        per sequence, while keys and values still cover all 7 tokens. Its output
        is ``[num_anchors, hidden]`` and goes straight to the final norm.
        With ``readout_mode="anchor_neighbor_attention"`` the last layer runs on all
        tokens instead, and ``_packed_neighbor_readout`` adds each anchor's
        attention-weighted neighbours to it.

        Returns:
            ``[num_anchor_nodes, out_dim]`` embeddings after the final norm, the readout
            and the output projection, or None if the batch's k-hop candidate set is
            too large to lay out. The caller then runs the padded path.
        """
        node_type_offsets = get_node_type_offsets(projected_x_dict)
        node_features = concat_tensors(list(projected_x_dict.values()), dim=0)
        edge_index = to_homogeneous_edge_index(
            edge_index_dict=edge_index_dict,
            node_type_offsets=node_type_offsets,
            device=device,
        )
        if anchor_node_ids is not None:
            anchor_local_indices = anchor_node_ids.to(device)
        else:
            anchor_local_indices = torch.arange(num_anchor_nodes, device=device)
        layout = self._build_packed_layout(
            edge_index=edge_index,
            num_nodes=node_features.size(0),
            anchor_indices=node_type_offsets[anchor_node_type] + anchor_local_indices,
            edge_index_dict=edge_index_dict,
            node_type_offsets=node_type_offsets,
        )
        if layout is None:
            return None
        x = self._packed_token_inputs(node_features[layout.node_index], layout)
        bias = self._packed_attention_bias(layout)
        x = self._run_packed_layers(x, layout, bias)
        return self._project_output(self._packed_readout(x, layout))

    def _build_packed_layout(
        self,
        edge_index: Tensor,
        num_nodes: int,
        anchor_indices: Tensor,
        edge_index_dict: dict[tuple[str, str, str], Tensor],
        node_type_offsets: dict[NodeType, int],
    ) -> Optional[PackedLayout]:
        """Builds the packed k-hop layout of the anchors.

        Args:
            edge_index: ``[2, num_edges]`` homogeneous batch edges.
            num_nodes: Number of nodes over all node types.
            anchor_indices: Homogeneous node id of each anchor.
            edge_index_dict: Batch edges by edge type.
            node_type_offsets: Offset of each node type in the homogeneous ids.

        Returns:
            The layout, with ``relation_edges`` set when a relation message mode
            is on, or None if the batch's k-hop candidate set is too large to lay
            out.
        """
        try:
            layout = pack_khop_layout(
                edge_index,
                num_nodes=num_nodes,
                anchor_indices=anchor_indices,
                hop_distance=self._hop_distance,
                max_seq_len=self._max_seq_len,
                sampling_direction=self._sampling_direction,
                prioritize_hop_order=self._prioritize_hop_order,
            )
        except KhopLayoutTooLargeError as error:
            self._record_packed_fallback(error)
            return None
        if self._relation_message_mode == "none":
            return layout
        return layout._replace(
            relation_edges=self._build_packed_relation_edges(
                edge_index_dict=edge_index_dict,
                node_type_offsets=node_type_offsets,
                num_nodes=num_nodes,
                layout=layout,
            )
        )

    def _packed_token_inputs(
        self,
        x: Float[Tensor, "tokens hid_dim"],
        layout: PackedLayout,
    ) -> Float[Tensor, "tokens hid_dim"]:
        """Adds the hop distance input to the packed token features.

        An embedding is added when the hop distance has one. Otherwise the hop
        distance is a continuous input and goes through the token projection.
        """
        if HOP_DISTANCE_FEATURE_NAME not in self._anchor_based_input_attr_names:
            return x
        token_hop = layout.token_attrs[HOP_DISTANCE_FEATURE_NAME]
        embedding_dict = self._anchor_based_input_embedding_dict
        if embedding_dict is not None and HOP_DISTANCE_FEATURE_NAME in embedding_dict:
            return x + embedding_dict[HOP_DISTANCE_FEATURE_NAME](token_hop).to(x.dtype)
        # A hop distance without an embedding is a continuous input, so the
        # constructor created the projection.
        token_projection = cast(nn.Module, self._token_input_projection)
        return x + token_projection(token_hop.unsqueeze(-1).to(x.dtype))

    def _packed_attention_bias(
        self, layout: PackedLayout
    ) -> Optional[PackedAttentionBias]:
        """The attention bias shared by every layer, or None for no bias."""
        return None

    def _run_packed_layers(
        self,
        x: Float[Tensor, "tokens hid_dim"],
        layout: PackedLayout,
        bias: Optional[PackedAttentionBias],
    ) -> Float[Tensor, "rows hid_dim"]:
        """Runs the encoder layers on packed tokens.

        Only the anchor token is read out with ``anchor_only``, so the last layer
        computes just those rows. The neighbour readout needs every token.

        Returns:
            ``[num_sequences, hid_dim]`` anchor states for ``readout_mode``
            ``"anchor_only"``, including with no layers, and
            ``[num_tokens, hid_dim]`` otherwise.
        """
        anchor_only = self._readout_mode == "anchor_only"
        num_layers = len(self._encoder_layers)
        for layer_index, encoder_layer in enumerate(self._encoder_layers):
            first_token_output = anchor_only and layer_index == num_layers - 1
            relation_edges = layout.relation_edges
            if first_token_output and relation_edges is not None:
                relation_edges = _anchor_relation_edges(relation_edges, layout)
            layer_kwargs = {}
            if bias is not None:
                layer_kwargs["bias"] = bias
            x = self._run_packed_layer(
                cast(GraphTransformerEncoderLayer, encoder_layer),
                x,
                layout,
                first_token_output,
                relation_edges,
                **layer_kwargs,
            )
        if anchor_only and num_layers == 0:
            return x[layout.cu_seqlens[:-1].long()]
        return x

    def _packed_readout(
        self,
        x: Float[Tensor, "rows hid_dim"],
        layout: PackedLayout,
    ) -> Float[Tensor, "sequences hid_dim"]:
        """Applies the final norm, then the readout.

        Args:
            x: Output of ``_run_packed_layers``.
            layout: Layout of the sequences in ``x``.

        Returns:
            One embedding per sequence.
        """
        x = self._final_norm(x)
        if self._readout_mode == "anchor_only":
            return x
        return self._packed_neighbor_readout(x, layout)

    def _packed_neighbor_readout(
        self,
        x: Float[Tensor, "tokens hid_dim"],
        layout: PackedLayout,
    ) -> Float[Tensor, "sequences hid_dim"]:
        """Packed version of the ``anchor_neighbor_attention`` readout.

        Each anchor gets itself plus the sum of its neighbours weighted by a softmax
        over its own sequence. Boolean indexing and the softmax's default node count
        would read sizes back from the GPU, so the anchor tokens are masked out of
        the softmax instead of dropped, and the softmax is given ``num_nodes``.

        Args:
            x: Final-normed token states, the anchor first in each sequence.
            layout: Layout of the sequences in ``x``.

        Returns:
            One embedding per sequence. A sequence with no neighbours gives its anchor.
        """
        if self._readout_attention is None:
            raise ValueError("Readout attention layer is not initialized.")
        num_sequences = layout.cu_seqlens.numel() - 1
        sequence_id = layout.sequence_id
        is_neighbor = layout.position != 0
        anchor = x[layout.cu_seqlens[:-1].long()]
        scores = self._readout_attention(
            torch.cat([anchor[sequence_id], x], dim=-1)
        ).squeeze(-1)
        # A finite fill keeps a sequence with only its anchor free of NaN.
        scores = scores.masked_fill(~is_neighbor, torch.finfo(scores.dtype).min)
        weights = softmax(scores, index=sequence_id, num_nodes=num_sequences)
        weights = weights * is_neighbor.to(weights.dtype)
        return anchor.index_add(0, sequence_id, x * weights.unsqueeze(-1))

    def _run_packed_layer(
        self,
        layer: GraphTransformerEncoderLayer,
        x: Tensor,
        layout: PackedLayout,
        first_token_output: bool,
        relation_edges: Optional[PackedRelationEdges],
        bias: Optional[PackedAttentionBias] = None,
    ) -> Tensor:
        """Runs one layer on packed tokens.

        A subclass can override this to wrap the call, for example in
        ``torch.utils.checkpoint.checkpoint``. The layer reads no host values,
        so it can be recomputed. ``_run_packed_layers`` passes ``bias`` only when
        there is one.

        Args:
            layer: The layer to run.
            x: Packed token states.
            layout: Layout of the sequences in ``x``.
            first_token_output: Compute only each sequence's first token.
            relation_edges: The relation edges for this layer, which replace
                ``layout.relation_edges``.
            bias: Optional additive attention bias.
        """
        return layer.forward_packed(
            x,
            layout._replace(relation_edges=relation_edges),
            first_token_output=first_token_output,
            bias=bias,
        )

    def _build_packed_relation_edges(
        self,
        edge_index_dict: dict[tuple[str, str, str], Tensor],
        node_type_offsets: dict[NodeType, int],
        num_nodes: int,
        layout: PackedLayout,
    ) -> PackedRelationEdges:
        """Matches the batch's edges to packed token pairs.

        Uses the padded path's edge matcher on the layout, so the edges are the
        ones it would give the padded sequences: those between two kept tokens
        whose target (``"in"`` sampling) or source (``"out"``) is fewer than
        ``hop_distance`` hops from the anchor.

        Args:
            edge_index_dict: Batch edges by edge type.
            node_type_offsets: Offset of each node type in the homogeneous ids.
            num_nodes: Number of nodes over all node types.
            layout: Layout of the packed tokens, with ``hop_distance`` in
                ``token_attrs``.

        Returns:
            The edges into every token, grouped by relation.
        """
        device = layout.node_index.device
        num_sequences = layout.cu_seqlens.numel() - 1
        grid_size = num_sequences * layout.max_seq_len
        node_index = torch.full((grid_size,), -1, dtype=torch.long, device=device)
        node_index[layout.flat_index] = layout.node_index
        node_index = node_index.view(num_sequences, layout.max_seq_len)
        hop_index = torch.zeros(grid_size, dtype=torch.long, device=device)
        hop_index[layout.flat_index] = layout.token_attrs[HOP_DISTANCE_FEATURE_NAME]
        hop_index = hop_index.view(num_sequences, layout.max_seq_len)
        lengths = layout.cu_seqlens[1:] - layout.cu_seqlens[:-1]
        valid_mask = torch.arange(
            layout.max_seq_len, device=device
        ) < lengths.unsqueeze(1)
        # Sorted batch * num_nodes + node keys of the tokens fewer than hop_distance
        # hops out; the rest sort last as a key no edge has.
        batch_node_keys = (
            torch.arange(num_sequences, device=device).unsqueeze(1) * num_nodes
            + node_index
        )
        frontier_keys, _ = (
            torch.where(
                valid_mask & (hop_index < self._hop_distance),
                batch_node_keys,
                torch.iinfo(torch.long).max,
            )
            .flatten()
            .sort()
        )
        relation_edges = _lookup_pairwise_relation_indices(
            edge_index_dict=edge_index_dict,
            node_index_sequences=node_index,
            valid_mask=valid_mask,
            relation_edge_types=self._relation_attention_edge_types,
            node_type_offsets=node_type_offsets,
            num_nodes=num_nodes,
            device=device,
            frontier_batch_node_keys=frontier_keys,
            sampling_direction=self._sampling_direction,
        )
        if relation_edges is None:
            raise ValueError("Relation edge types are not initialized.")

        first_tokens = layout.cu_seqlens[:-1].long()
        batch_indices, target_positions, source_positions, relations = (
            relation_edges.unbind(dim=1)
        )
        relations = relations.contiguous()
        edge_first_tokens = first_tokens[batch_indices]
        # Each relation's edges are contiguous, so one host read of the relation
        # boundaries gives all the segments.
        relation_ids = torch.arange(
            len(self._relation_attention_edge_types) + 1, device=relations.device
        )
        bounds = torch.searchsorted(relations, relation_ids).tolist()
        return PackedRelationEdges(
            target=edge_first_tokens + target_positions,
            source=edge_first_tokens + source_positions,
            relation=relations,
            segments=_get_relation_segments_from_bounds(bounds),
        )

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

        1. **Padding mask bias**: Sets padded positions to a large negative value
           (``finfo.min / 2``, leaving additive headroom in low-precision dtypes)
           so they receive exactly zero attention weight after softmax.
           Shape: (batch, 1, 1, seq) broadcasts to (batch, num_heads, seq, seq)
           for key masking.

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
                - "pairwise_nonmissing_indices": (num_pairs, 3) or None

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
        # Large negative fill for padded key positions. finfo.min / 2 (rather
        # than finfo.min) leaves headroom so later ADDITIVE bias terms
        # (anchor-relative / pairwise projections) cannot overflow to -inf in
        # low-precision dtypes (fp16 min is only -65504) and yield NaNs in the
        # softmax. Half the dtype minimum still zeroes the attention weight
        # exactly after softmax.
        negative_inf = torch.finfo(dtype).min / 2.0

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

        pairwise_nonmissing_indices = attention_bias_data.get(
            "pairwise_nonmissing_indices"
        )
        if pairwise_nonmissing_indices is not None:
            if self._pairwise_nonmissing_attention_bias is None:
                raise ValueError(
                    "Pairwise nonmissing attention bias is not initialized."
                )
            if pairwise_nonmissing_indices.numel() > 0:
                if attn_bias.shape[1:] != (self._num_heads, seq_len, seq_len):
                    attn_bias = attn_bias.expand(
                        batch_size,
                        self._num_heads,
                        seq_len,
                        seq_len,
                    ).clone()
                pairwise_nonmissing_indices = pairwise_nonmissing_indices.to(
                    device=device,
                    dtype=torch.long,
                )
                attn_bias_by_position = attn_bias.permute(0, 2, 3, 1)
                nonmissing_bias = self._pairwise_nonmissing_attention_bias.to(
                    dtype=attn_bias.dtype
                ).view(1, -1)
                attn_bias_by_position.index_put_(
                    (
                        pairwise_nonmissing_indices[:, 0],
                        pairwise_nonmissing_indices[:, 1],
                        pairwise_nonmissing_indices[:, 2],
                    ),
                    nonmissing_bias.expand(pairwise_nonmissing_indices.size(0), -1),
                    accumulate=True,
                )

        return attn_bias

    def _encode_and_readout(
        self,
        sequences: Tensor,
        valid_mask: Tensor,
        attn_bias: Optional[Tensor] = None,
        pairwise_relation_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Process sequences through transformer layers and attention readout.

        Args:
            sequences: Input tensor of shape ``(batch_size, max_seq_len, hid_dim)``.
            valid_mask: Boolean mask of shape ``(batch_size, max_seq_len)``.
            attn_bias: Optional additive attention bias broadcastable to
                ``(batch_size, num_heads, seq, seq)``.
            pairwise_relation_indices: Optional sparse relation coordinates shaped
                ``(num_relation_edges, 4)``.

        Returns:
            Output embeddings of shape ``(batch_size, hid_dim)``.
        """
        x = sequences * valid_mask.unsqueeze(-1).to(sequences.dtype)

        encoder_layers = self._encoder_layers
        use_anchor_only_final_layer = (
            self._readout_mode == "anchor_only" and len(encoder_layers) > 0
        )
        final_encoder_layer = (
            cast(GraphTransformerEncoderLayer, encoder_layers[-1])
            if use_anchor_only_final_layer
            else None
        )
        normal_encoder_layer_count = len(encoder_layers) - int(
            final_encoder_layer is not None
        )
        for encoder_layer_index in range(normal_encoder_layer_count):
            encoder_layer = cast(
                GraphTransformerEncoderLayer,
                encoder_layers[encoder_layer_index],
            )
            x = encoder_layer(
                x,
                attn_bias=attn_bias,
                pairwise_relation_indices=pairwise_relation_indices,
                valid_mask=valid_mask,
            )

        # ``anchor_only`` readout returns only position zero. In the final layer,
        # non-anchor outputs cannot affect that result, but the anchor still
        # attends over the full sequence as keys and values. Compute only the
        # anchor query/output and shorten the mask to match that one-token result.
        output_valid_mask = valid_mask
        if final_encoder_layer is not None:
            x = final_encoder_layer(
                x,
                attn_bias=attn_bias,
                pairwise_relation_indices=pairwise_relation_indices,
                valid_mask=valid_mask,
                query_seq_len=1,
            )
            output_valid_mask = valid_mask[:, :1]

        x = self._final_norm(x)
        x = x * output_valid_mask.unsqueeze(-1).to(x.dtype)

        # Readout: anchor (position 0) + attention-weighted neighbor aggregation
        anchor = x[:, 0, :].unsqueeze(1)  # (batch, 1, hid_dim)
        if self._readout_mode == "anchor_only":
            return anchor.squeeze(1)

        neighbors = x[:, 1:, :]  # (batch, seq-1, hid_dim)
        neighbor_valid_mask = valid_mask[:, 1:]
        seq_minus_one = neighbors.size(1)

        if seq_minus_one == 0:
            return anchor.squeeze(1)

        # Expand anchor to match neighbor dimension for concatenation
        anchor_expanded = anchor.expand(-1, seq_minus_one, -1)

        # Compute attention scores over neighbors
        if self._readout_attention is None:
            raise ValueError("Readout attention layer is not initialized.")
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
