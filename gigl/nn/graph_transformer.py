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
from typing import Callable, Literal, Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.hetero_data
from jaxtyping import Bool, Float, Int64
from torch import Tensor

from gigl.common.logger import Logger
from gigl.nn.packed_attention import (
    PackedForwardConfig,
    PackedTokens,
    pack_tokens,
    packed_attention,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.transforms.graph_transformer import (
    PPR_RELATION_FEATURES_NAME,
    PPR_WEIGHT_FEATURE_NAME,
    SequenceAuxiliaryData,
    TokenInputData,
    _concat_tensors,
    _get_node_type_offsets,
    _to_homogeneous_edge_index,
    _validate_reserved_anchor_feature_usage,
    tensors_to_graph_transformer_input,
)
from gigl.transforms.static_khop import (
    build_neighbor_table,
    gather_rows,
    khop_sequence_layout,
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
        packed_tokens: Optional[PackedTokens] = None,
    ) -> Float[Tensor, "batch query_sequence model_dim"]:
        """Forward pass.

        With ``packed_tokens``, ``x`` is ``(1, token_capacity, model_dim)``:
        every sequence's tokens packed back to back (see
        ``gigl.nn.packed_attention``), with zeros on slack slots. Each token
        attends only within its own sequence, and slack slots stay zero.
        ``query_seq_len=1`` then returns ``(1, num_sequences, model_dim)``, the
        first token of each sequence. Packed tokens require
        ``relation_attention_mode`` and ``relation_message_mode``
        ``"none"``, no ``attn_bias``, and no attention dropout in training.

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
            packed_tokens: Optional packed layout of ``x``; see above.

        Returns:
            Output tensor of shape ``(batch, query_seq_len, model_dim)``.

        Raises:
            ValueError: If ``packed_tokens`` is combined with ``attn_bias``,
                relation modes, a ``query_seq_len`` other than 1, or attention
                dropout in training.
        """
        if packed_tokens is not None:
            if (
                attn_bias is not None
                or self._relation_attention_mode != "none"
                or self._relation_message_mode != "none"
            ):
                raise ValueError(
                    "packed_tokens supports no attn_bias and relation modes 'none'."
                )
            if query_seq_len not in (None, 1):
                raise ValueError(
                    f"packed_tokens supports query_seq_len None or 1, got {query_seq_len}."
                )
            if self.training and self._attention_dropout_rate > 0:
                raise ValueError(
                    "Packed attention does not apply attention dropout. For "
                    "training with packed_forward, set attention_dropout_rate=0, "
                    "or omit packed_forward."
                )
            return self._forward_packed_tokens(
                x=x[0], packed=packed_tokens, first_token_output=query_seq_len == 1
            ).unsqueeze(0)
        return self._forward_query_prefix(
            x=x,
            seq_len=x.size(1) if query_seq_len is None else query_seq_len,
            attn_bias=attn_bias,
            valid_mask=valid_mask,
            pairwise_relation_indices=pairwise_relation_indices,
        )

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

    def _forward_packed_tokens(
        self, x: Tensor, packed: PackedTokens, first_token_output: bool
    ) -> Tensor:
        """Runs the layer on ``[token_capacity, model_dim]`` packed tokens.

        Returns:
            ``[num_sequences, model_dim]`` when ``first_token_output``, else
            ``[token_capacity, model_dim]`` with zeros on slack.
        """
        x_norm = self._attention_norm(x)
        key = self._key_projection(x_norm).view(-1, self._num_heads, self._head_dim)
        value = self._value_projection(x_norm).view(-1, self._num_heads, self._head_dim)
        if first_token_output:
            residual = x[packed.sequence_starts]
            query_input = x_norm[packed.sequence_starts]
        else:
            residual = x
            query_input = x_norm
        query = self._query_projection(query_input).view(
            -1, self._num_heads, self._head_dim
        )
        attention_output = packed_attention(
            query, key, value, packed, query_is_first_token=first_token_output
        ).reshape(-1, self._num_heads * self._head_dim)
        x = residual + self._dropout(self._output_projection(attention_output))
        # Slack rows can hold NaN, and NaN * 0 is NaN, so they are cleared with
        # ``where`` rather than a mask product.
        if not first_token_output:
            x = torch.where(packed.token_valid.unsqueeze(-1), x, 0.0)
        # The FFN acts on each row alone; its contract is (batch, sequence, dim).
        x = x + self._ffn(self._ffn_norm(x).unsqueeze(0)).squeeze(0)
        if not first_token_output:
            x = torch.where(packed.token_valid.unsqueeze(-1), x, 0.0)
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
        messages = torch.zeros(
            (batch_size, seq_len, x_norm.size(-1)),
            dtype=x_norm.dtype,
            device=x_norm.device,
        )
        if pairwise_relation_indices.numel() == 0:
            return messages
        batch_indices = pairwise_relation_indices[:, 0]
        target_indices = pairwise_relation_indices[:, 1]
        source_indices = pairwise_relation_indices[:, 2]
        relation_indices = pairwise_relation_indices[:, 3]

        # Per-(relation, batch, target) in-degree for mean aggregation, computed
        # globally up front so grouping order of the entries does not matter.
        target_degrees = torch.zeros(
            (self._num_relations, batch_size, seq_len),
            dtype=x_norm.dtype,
            device=x_norm.device,
        )
        target_degrees.index_put_(
            (relation_indices, batch_indices, target_indices),
            torch.ones(
                relation_indices.size(0), dtype=x_norm.dtype, device=x_norm.device
            ),
            accumulate=True,
        )
        edge_degrees = target_degrees[relation_indices, batch_indices, target_indices]

        relation_matrices = self._relation_message_matrices.to(dtype=x_norm.dtype)
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
            edge_batch_indices = batch_indices[relation_start:relation_end]
            edge_target_indices = target_indices[relation_start:relation_end]
            edge_source_indices = source_indices[relation_start:relation_end]

            source_features = x_norm[edge_batch_indices, edge_source_indices]
            edge_messages = source_features @ relation_matrices[relation_idx].t()
            edge_messages = edge_messages / edge_degrees[
                relation_start:relation_end
            ].unsqueeze(-1)
            messages.index_put_(
                (edge_batch_indices, edge_target_indices),
                edge_messages.to(dtype=messages.dtype),
                accumulate=True,
            )
            relation_start = relation_end

        return messages

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
        if self._relation_message_matrices is None:
            raise ValueError("Relation message matrices are not initialized.")
        if self._relation_message_attention_matrices is None:
            raise ValueError("Relation message attention matrices are not initialized.")
        if self._relation_message_attention_priors is None:
            raise ValueError("Relation message attention priors are not initialized.")
        messages = torch.zeros(
            (batch_size, seq_len, x_norm.size(-1)),
            dtype=x_norm.dtype,
            device=x_norm.device,
        )
        if pairwise_relation_indices.numel() == 0:
            return messages
        batch_indices = pairwise_relation_indices[:, 0]
        target_indices = pairwise_relation_indices[:, 1]
        source_indices = pairwise_relation_indices[:, 2]
        relation_indices = pairwise_relation_indices[:, 3]

        # Gather per-edge queries/keys: (num_edges, num_heads, head_dim).
        edge_queries = query[batch_indices, :, target_indices]
        edge_keys = key[batch_indices, :, source_indices]

        relation_attention_matrices = self._relation_message_attention_matrices.to(
            dtype=edge_queries.dtype
        )
        relation_attention_priors = self._relation_message_attention_priors.to(
            dtype=edge_queries.dtype
        )
        unique_relation_indices, relation_counts = torch.unique_consecutive(
            relation_indices,
            return_counts=True,
        )
        score_parts: list[Tensor] = []
        relation_start = 0
        for relation_idx_tensor, relation_count_tensor in zip(
            unique_relation_indices,
            relation_counts,
        ):
            relation_idx = int(relation_idx_tensor.item())
            relation_end = relation_start + int(relation_count_tensor.item())
            transformed_keys = torch.einsum(
                "ehd,hdf->ehf",
                edge_keys[relation_start:relation_end],
                relation_attention_matrices[relation_idx],
            )
            slice_scores = (
                edge_queries[relation_start:relation_end] * transformed_keys
            ).sum(dim=-1)
            score_parts.append(
                slice_scores
                * relation_attention_priors[relation_idx]
                / math.sqrt(self._head_dim)
            )
            relation_start = relation_end
        scores = torch.cat(score_parts, dim=0)  # (num_edges, num_heads)

        # Numerically stable scatter-softmax per (relation, batch, target) group
        # and head. Group buffers are (num_relations * batch * seq, num_heads) —
        # tiny compared to any (seq, seq) tensor.
        group_indices = (
            relation_indices * batch_size + batch_indices
        ) * seq_len + target_indices
        head_group_indices = group_indices.unsqueeze(-1).expand(-1, self._num_heads)
        num_groups = self._num_relations * batch_size * seq_len
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
        edge_weights = normalized_scores / group_denominators.gather(
            0, head_group_indices
        )  # (num_edges, num_heads), sums to 1 per (relation, target) and head

        relation_matrices = self._relation_message_matrices.to(dtype=x_norm.dtype)
        relation_start = 0
        for relation_idx_tensor, relation_count_tensor in zip(
            unique_relation_indices,
            relation_counts,
        ):
            relation_idx = int(relation_idx_tensor.item())
            relation_end = relation_start + int(relation_count_tensor.item())
            edge_batch_indices = batch_indices[relation_start:relation_end]
            edge_target_indices = target_indices[relation_start:relation_end]
            edge_source_indices = source_indices[relation_start:relation_end]

            source_features = x_norm[edge_batch_indices, edge_source_indices]
            edge_messages = source_features @ relation_matrices[relation_idx].t()
            weighted_messages = edge_messages.view(
                -1, self._num_heads, self._head_dim
            ) * edge_weights[relation_start:relation_end].unsqueeze(-1)
            messages.index_put_(
                (edge_batch_indices, edge_target_indices),
                weighted_messages.reshape(-1, x_norm.size(-1)).to(dtype=messages.dtype),
                accumulate=True,
            )
            relation_start = relation_end

        return messages


def _select_sequence_inputs(
    sequences: Tensor,
    valid_mask: Tensor,
    sequence_auxiliary_data: SequenceAuxiliaryData,
    anchor_idx: Tensor,
    seq_len: int,
) -> tuple[Tensor, Tensor, SequenceAuxiliaryData]:
    """Selects a subset of anchors and trims the sequence dim to ``seq_len``.

    Every per-token tensor (``[B, L, ...]``) and pairwise tensor
    (``[B, L, L, ...]``) is sliced the same way, and sparse ``(anchor, ...)``
    index rows are kept for the selected anchors with the anchor renumbered.

    Callers must ensure no selected anchor has a valid position at or past
    ``seq_len``.
    """
    num_anchors = valid_mask.size(0)

    def per_token(tensor: Tensor) -> Tensor:
        return tensor[anchor_idx, :seq_len]

    def pairwise(tensor: Optional[Tensor]) -> Optional[Tensor]:
        if tensor is None:
            return None
        return tensor[anchor_idx, :seq_len, :seq_len]

    def sparse(indices: Optional[Tensor]) -> Optional[Tensor]:
        # Rows are (anchor, query, key[, relation]). Every valid token lies
        # within ``seq_len``, so only the anchor column needs renumbering.
        if indices is None:
            return None
        new_anchor_by_old = torch.full(
            (num_anchors,), -1, dtype=torch.long, device=indices.device
        )
        new_anchor_by_old[anchor_idx.to(indices.device)] = torch.arange(
            anchor_idx.numel(), device=indices.device
        )
        new_anchor = new_anchor_by_old[indices[:, 0]]
        keep = new_anchor >= 0
        indices = indices[keep].clone()
        indices[:, 0] = new_anchor[keep]
        return indices

    anchor_bias = sequence_auxiliary_data["anchor_bias"]
    token_input = sequence_auxiliary_data["token_input"]
    selected_auxiliary_data: SequenceAuxiliaryData = {
        "anchor_bias": None if anchor_bias is None else per_token(anchor_bias),
        "pairwise_bias": pairwise(sequence_auxiliary_data["pairwise_bias"]),
        "pairwise_relation_indices": sparse(
            sequence_auxiliary_data["pairwise_relation_indices"]
        ),
        "pairwise_nonmissing_indices": sparse(
            sequence_auxiliary_data["pairwise_nonmissing_indices"]
        ),
        "token_input": (
            None
            if token_input is None
            else {name: per_token(value) for name, value in token_input.items()}
        ),
    }
    return per_token(sequences), per_token(valid_mask), selected_auxiliary_data


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
            only the encoded anchor token.
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
        length_buckets: Optional sequence-length bounds, e.g.
            ``[64, 128, 192]``, each an int in ``[1, max_seq_len]``; order and
            duplicates do not matter. Anchors are grouped by valid sequence
            length (up to their last valid token), and each group runs the
            encoder layers on sequences trimmed to the smallest bound that fits
            it, or to ``max_seq_len``. Trimmed positions are padding, so
            embeddings and gradients match the untrimmed forward up to float
            error (with dropout off) while attention and FFNs skip most padded
            tokens.

            Costs per batch:

            - One host sync for the group sizes.
            - Per non-empty group, the host syncs of selecting that group's
              rows of sparse relation and pairwise-bias indices, when those are
              used.
            - Under ``torch.compile``, each group runs at its own
              ``(num_anchors, bound)`` shape. With the default ``dynamic``
              setting, the sequence dim turns dynamic after the first new shape
              and size-1 groups specialize. On CUDA, compiled layers with a
              dynamic sequence dim can hit pytorch#163876 for a group of one
              anchor (fixed in torch 2.11); compiling the layers with
              ``dynamic=False`` and marking only the batch dim dynamic
              (``torch._dynamo.mark_dynamic(x, 0)``) avoids it.

            (default: None)
        layout_hop_distance_attr_name: Optional anchor-relative attribute name
            (usually ``"hop_distance"``) whose values are read off the k-hop
            sequence layout instead of a sparse matrix on the input data, so the
            batch needs no all-pairs ``AddHeteroHopDistanceEncoding``. The
            values equal that transform's with ``is_undirected=False`` and the
            same ``sampling_direction``. They range over ``0..hop_distance``,
            so an embedding for it needs ``hop_distance + 1`` rows. It must
            appear in ``anchor_based_attention_bias_attr_names`` or
            ``anchor_based_input_attr_names``, and requires
            ``sequence_construction_method="khop"``. See
            ``heterodata_to_graph_transformer_input``. (default: None)
        packed_forward: Optional ``PackedForwardConfig``. ``forward`` then
            builds k-hop sequences with the fixed-shape layout in
            ``gigl.transforms.static_khop`` and runs the encoder layers on
            packed tokens (``gigl.nn.packed_attention``). Apart from one read
            of the overflow flag there are no host syncs, and on CUDA no
            padding FLOPs; on CPU, attention runs on a padded grid as a
            fallback. Each anchor gets at most
            ``min(tokens_per_anchor, max_seq_len)`` packed slots. Embeddings
            and gradients match the padded path up to float error; training
            needs ``attention_dropout_rate=0``. A batch that exceeds the
            config's token capacity or neighbor table width is recomputed on
            the padded path and counted in ``packed_overflow_count``.
            ``length_buckets`` then apply to those batches only. Supports
            ``sequence_construction_method="khop"`` without
            ``prioritize_hop_order``, ``readout_mode="anchor_only"``, relation
            modes ``"none"``, no anchor-based or pairwise attention bias, and
            token inputs limited to ``layout_hop_distance_attr_name``.
            (default: None)

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
        length_buckets: Optional[list[int]] = None,
        layout_hop_distance_attr_name: Optional[str] = None,
        packed_forward: Optional[PackedForwardConfig] = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        del kwargs
        self._packed_forward = packed_forward
        # Batches that overflowed the packed layout and ran on the padded path.
        self.packed_overflow_count = 0

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

        self._hid_dim = hid_dim
        self._out_dim = out_dim
        self._max_seq_len = max_seq_len
        self._length_bucket_bounds: Optional[list[int]] = None
        if length_buckets is not None:
            if not length_buckets or any(
                isinstance(bound, bool)
                or not isinstance(bound, int)
                or not 1 <= bound <= max_seq_len
                for bound in length_buckets
            ):
                raise ValueError(
                    "length_buckets must be a non-empty list of ints in "
                    f"[1, max_seq_len={max_seq_len}], got {list(length_buckets)}"
                )
            # Bounds are host constants, so the bucket loop needs no device
            # unique; the device copy is a buffer so no batch pays an HtoD copy.
            self._length_bucket_bounds = sorted(set(length_buckets) | {max_seq_len})
            self.register_buffer(
                "_length_bucket_bounds_tensor",
                torch.tensor(self._length_bucket_bounds, dtype=torch.long),
                persistent=False,
            )
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
        if (
            layout_hop_distance_attr_name is not None
            and sequence_construction_method != "khop"
        ):
            raise ValueError(
                "layout_hop_distance_attr_name requires "
                "sequence_construction_method='khop'."
            )
        if layout_hop_distance_attr_name is not None and (
            layout_hop_distance_attr_name
            not in list(anchor_based_attention_bias_attr_names or [])
            + list(anchor_based_input_attr_names or [])
        ):
            raise ValueError(
                f"layout_hop_distance_attr_name '{layout_hop_distance_attr_name}' "
                "must be an anchor-based attention bias or token input."
            )
        self._layout_hop_distance_attr_name = layout_hop_distance_attr_name
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

        if self._packed_forward is not None:
            self._validate_packed_forward()

    def _validate_packed_forward(self) -> None:
        """Raises a ValueError listing every setting the packed path does not support."""
        reasons: list[str] = []
        if self._sequence_construction_method != "khop":
            reasons.append(
                f"sequence_construction_method={self._sequence_construction_method!r}"
            )
        if self._prioritize_hop_order:
            reasons.append("prioritize_hop_order=True")
        if self._readout_mode != "anchor_only":
            reasons.append(f"readout_mode={self._readout_mode!r}")
        if self._relation_attention_mode != "none":
            reasons.append(f"relation_attention_mode={self._relation_attention_mode!r}")
        if self._relation_message_mode != "none":
            reasons.append(f"relation_message_mode={self._relation_message_mode!r}")
        if self._anchor_based_attention_bias_attr_names:
            reasons.append(
                "anchor_based_attention_bias_attr_names="
                f"{self._anchor_based_attention_bias_attr_names}"
            )
        if self._pairwise_attention_bias_attr_names:
            reasons.append(
                "pairwise_attention_bias_attr_names="
                f"{self._pairwise_attention_bias_attr_names}"
            )
        unsupported_token_inputs = [
            attr_name
            for attr_name in self._anchor_based_input_attr_names
            if attr_name != self._layout_hop_distance_attr_name
        ]
        if unsupported_token_inputs:
            reasons.append(
                f"anchor_based_input_attr_names {unsupported_token_inputs} other "
                "than layout_hop_distance_attr_name"
            )
        if reasons:
            raise ValueError(
                "packed_forward does not support: " + "; ".join(reasons) + "."
            )

    def forward(
        self,
        data: torch_geometric.data.hetero_data.HeteroData,
        anchor_node_type: Optional[NodeType] = None,
        anchor_node_ids: Optional[Int64[Tensor, "anchors"]] = None,
        device: Optional[torch.device] = None,
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
        for edge_type in data.edge_types:
            edge_store = data[edge_type]
            edge_index_dict[edge_type] = edge_store.edge_index
            if getattr(edge_store, "edge_attr", None) is not None:
                edge_attr_dict[edge_type] = edge_store.edge_attr
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
        relative_pe_attr_names.discard(self._layout_hop_distance_attr_name)
        graph_attr_dict = {
            attr_name: getattr(data, attr_name)
            for attr_name in sorted(relative_pe_attr_names)
            if hasattr(data, attr_name)
        }

        # 2. Build sequences and run transformer
        # If anchor_node_ids provided, use those; otherwise use first batch_size nodes
        if anchor_node_ids is not None:
            num_anchor_nodes = anchor_node_ids.size(0)
        else:
            anchor_store = data[anchor_node_type]
            num_anchor_nodes = (
                anchor_store.batch_size
                if hasattr(anchor_store, "batch_size")
                else projected_x_dict[anchor_node_type].size(0)
            )

        if self._packed_forward is not None and num_anchor_nodes > 0:
            anchor_states, overflow = self._encode_packed(
                projected_x_dict=projected_x_dict,
                edge_index_dict=edge_index_dict,
                anchor_node_type=anchor_node_type,
                num_anchor_nodes=num_anchor_nodes,
                anchor_node_ids=anchor_node_ids,
                device=device,
            )
            # The one host read of the packed path, after its kernels are
            # queued, so a wrong result never reaches the caller.
            if not bool(overflow):
                return self._project_output(anchor_states)
            if self.packed_overflow_count == 0:
                logger.warning(
                    "A batch overflowed packed_forward (more tokens than "
                    "tokens_per_anchor allows, or a node with more edges than "
                    "neighbor_table_width) and is recomputed on the padded path. "
                    "packed_overflow_count counts these batches; if it grows with "
                    "every batch, raise the config's limits."
                )
            self.packed_overflow_count += 1
            # Free the discarded packed activations (and, in training, their
            # autograd graph) before the padded recompute allocates its own.
            del anchor_states, overflow

        (
            sequences,
            valid_mask,
            sequence_auxiliary_data,
        ) = tensors_to_graph_transformer_input(
            x_dict=projected_x_dict,
            edge_index_dict=edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            graph_attr_dict=graph_attr_dict,
            batch_size=num_anchor_nodes,
            max_seq_len=self._max_seq_len,
            anchor_node_type=anchor_node_type,
            anchor_node_ids=anchor_node_ids,
            hop_distance=self._hop_distance,
            sequence_construction_method=self._sequence_construction_method,
            sampling_direction=self._sampling_direction,
            prioritize_hop_order=self._prioritize_hop_order,
            layout_hop_distance_attr_name=self._layout_hop_distance_attr_name,
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

        if self._length_bucket_bounds is None:
            embeddings = self._encode_sequences(
                sequences=sequences,
                valid_mask=valid_mask,
                sequence_auxiliary_data=sequence_auxiliary_data,
            )
        else:
            embeddings = self._encode_sequences_by_length_bucket(
                sequences=sequences,
                valid_mask=valid_mask,
                sequence_auxiliary_data=sequence_auxiliary_data,
            )
        return self._project_output(embeddings)

    def _project_output(self, embeddings: Tensor) -> Tensor:
        """Projects encoder outputs to ``out_dim`` and optionally L2-normalizes."""
        embeddings = self._output_projection(embeddings)
        if self._should_l2_normalize_embedding_layer_output:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _encode_packed(
        self,
        projected_x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
        anchor_node_type: NodeType,
        num_anchor_nodes: int,
        anchor_node_ids: Optional[Tensor],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Encodes the anchors on packed tokens.

        Every shape is fixed by the inputs and the ``PackedForwardConfig``, so
        this runs without host syncs.

        Returns:
            - ``[num_anchor_nodes, hid_dim]`` final-norm anchor states, wrong
              when the overflow flag is set.
            - Scalar bool overflow flag: some node has more edges than the
              neighbor table width, or the sequences hold more tokens than the
              packed capacity.
        """
        assert self._packed_forward is not None
        node_type_offsets = _get_node_type_offsets(projected_x_dict)
        node_features = _concat_tensors(list(projected_x_dict.values()), dim=0)
        edge_index = _to_homogeneous_edge_index(
            edge_index_dict=edge_index_dict,
            node_type_offsets=node_type_offsets,
            device=device,
        ).to(device=device, dtype=torch.long)
        anchor_local_indices = (
            anchor_node_ids.to(device)
            if anchor_node_ids is not None
            else torch.arange(num_anchor_nodes, device=device)
        )
        neighbor_table, table_overflow = build_neighbor_table(
            edge_index,
            num_nodes=node_features.size(0),
            width=self._packed_forward.neighbor_table_width,
            sampling_direction=self._sampling_direction,
        )
        node_index, hop_index, lengths = khop_sequence_layout(
            neighbor_table,
            node_type_offsets[anchor_node_type] + anchor_local_indices,
            hop_distance=self._hop_distance,
            max_seq_len=self._max_seq_len,
        )
        packed = pack_tokens(
            node_index,
            hop_index,
            lengths,
            # No sequence is longer than max_seq_len.
            token_capacity=num_anchor_nodes
            * min(self._packed_forward.tokens_per_anchor, self._max_seq_len),
        )

        tokens = gather_rows(node_features, packed.token_node, packed.token_valid)
        x = tokens.unsqueeze(0)
        if self._layout_hop_distance_attr_name in self._anchor_based_input_attr_names:
            x = x + self._build_token_input_contribution(
                token_input_features={
                    self._layout_hop_distance_attr_name: packed.token_hop.view(
                        1, -1, 1
                    ).float()
                },
                sequences=x,
                valid_mask=packed.token_valid.unsqueeze(0),
            )
        encoder_layers = self._encoder_layers
        for layer_index, encoder_layer in enumerate(encoder_layers):
            x = encoder_layer(
                x,
                query_seq_len=1 if layer_index == len(encoder_layers) - 1 else None,
                packed_tokens=packed,
            )
        if len(encoder_layers) == 0:
            x = x[:, packed.sequence_starts]
        return self._final_norm(x).squeeze(0), table_overflow | packed.overflow

    def _encode_sequences(
        self,
        sequences: Tensor,
        valid_mask: Tensor,
        sequence_auxiliary_data: SequenceAuxiliaryData,
    ) -> Tensor:
        """Adds token inputs and positional encodings, then encodes and reads out."""
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

        return self._encode_and_readout(
            sequences=sequences,
            valid_mask=valid_mask,
            attn_bias=attn_bias,
            pairwise_relation_indices=sequence_auxiliary_data[
                "pairwise_relation_indices"
            ],
        )

    def _encode_sequences_by_length_bucket(
        self,
        sequences: Tensor,
        valid_mask: Tensor,
        sequence_auxiliary_data: SequenceAuxiliaryData,
    ) -> Tensor:
        """Encodes anchors grouped by valid length, each group trimmed to its bound.

        Outputs are scattered back to the original anchor order.
        """
        assert self._length_bucket_bounds is not None
        if valid_mask.size(0) == 0:
            return self._encode_sequences(
                sequences=sequences,
                valid_mask=valid_mask,
                sequence_auxiliary_data=sequence_auxiliary_data,
            )
        bounds = self._length_bucket_bounds
        positions = torch.arange(1, valid_mask.size(1) + 1, device=valid_mask.device)
        # Length up to and including the last valid token; at least 1.
        anchor_lengths = (valid_mask.long() * positions).amax(dim=1).clamp(min=1)
        anchor_buckets = torch.bucketize(
            anchor_lengths, cast(Tensor, self._length_bucket_bounds_tensor)
        )
        # A stable sort keeps anchors in ascending order within each bucket; the
        # bucket sizes are the one host read.
        anchor_order = torch.argsort(anchor_buckets, stable=True)
        bucket_counts = torch.zeros(
            len(bounds), dtype=torch.long, device=anchor_lengths.device
        ).index_add_(0, anchor_buckets, torch.ones_like(anchor_buckets))
        outputs: Optional[Tensor] = None
        start = 0
        for bound, count in zip(bounds, bucket_counts.tolist()):
            if count == 0:
                continue
            anchor_idx = anchor_order[start : start + count]
            start += count
            (
                bucket_sequences,
                bucket_valid_mask,
                bucket_auxiliary_data,
            ) = _select_sequence_inputs(
                sequences=sequences,
                valid_mask=valid_mask,
                sequence_auxiliary_data=sequence_auxiliary_data,
                anchor_idx=anchor_idx,
                seq_len=bound,
            )
            bucket_outputs = self._encode_sequences(
                sequences=bucket_sequences,
                valid_mask=bucket_valid_mask,
                sequence_auxiliary_data=bucket_auxiliary_data,
            )
            if outputs is None:
                outputs = bucket_outputs.new_empty(
                    (valid_mask.size(0),) + bucket_outputs.shape[1:]
                )
            outputs[anchor_idx] = bucket_outputs
        assert outputs is not None
        return outputs

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
