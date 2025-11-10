from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn.conv import LGConv
from torchrec.distributed.types import Awaitable
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing_extensions import Self

from gigl.src.common.types.graph_data import NodeType
from gigl.types.graph import to_heterogeneous_node


class LinkPredictionGNN(nn.Module):
    """
    Link Prediction GNN model for both homogeneous and heterogeneous use cases
    Args:
        encoder (nn.Module): Either BasicGNN or Heterogeneous GNN for generating embeddings
        decoder (nn.Module): Decoder for transforming embeddings into scores.
            Recommended to use `gigl.src.common.models.pyg.link_prediction.LinkPredictionDecoder`
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(
        self,
        data: Union[Data, HeteroData],
        device: torch.device,
        output_node_types: Optional[list[NodeType]] = None,
    ) -> Union[torch.Tensor, dict[NodeType, torch.Tensor]]:
        if isinstance(data, HeteroData):
            if output_node_types is None:
                raise ValueError(
                    "Output node types must be specified in forward() pass for heterogeneous model"
                )
            return self._encoder(
                data=data, output_node_types=output_node_types, device=device
            )
        else:
            return self._encoder(data=data, device=device)

    def decode(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self._decoder(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
        )

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    def to_ddp(
        self,
        device: Optional[torch.device],
        find_unused_encoder_parameters: bool = False,
    ) -> Self:
        """
        Converts the model to DistributedDataParallel (DDP) mode.

        We do this because DDP does *not* expect the forward method of the modules it wraps to be called directly.
        See how DistributedDataParallel.forward calls _pre_forward:
        https://github.com/pytorch/pytorch/blob/26807dcf277feb2d99ab88d7b6da526488baea93/torch/nn/parallel/distributed.py#L1657
        If we do not do this, then calling forward() on the individual modules may not work correctly.

        Calling this function makes it safe to do: `LinkPredictionGNN.decoder(data, device)`

        Args:
            device (Optional[torch.device]): The device to which the model should be moved.
                If None, will default to CPU.
            find_unused_encoder_parameters (bool): Whether to find unused parameters in the model.
                This should be set to True if the model has parameters that are not used in the forward pass.
        Returns:
            LinkPredictionGNN: A new instance of LinkPredictionGNN for use with DDP.
        """

        if device is None:
            device = torch.device("cpu")
        ddp_encoder = DistributedDataParallel(
            self._encoder.to(device),
            device_ids=[device] if device.type != "cpu" else None,
            find_unused_parameters=find_unused_encoder_parameters,
        )
        # Do this "backwards" so the we can define "ddp_decoder" as a nn.Module first...
        if not any(p.requires_grad for p in self._decoder.parameters()):
            # If the decoder has no trainable parameters, we can just use it as is
            ddp_decoder = self._decoder.to(device)
        else:
            # Only wrap the decoder in DDP if it has parameters that require gradients
            # Otherwise DDP will complain about no parameters to train.
            ddp_decoder = DistributedDataParallel(
                self._decoder.to(device),
                device_ids=[device] if device.type != "cpu" else None,
            )
        self._encoder = ddp_encoder
        self._decoder = ddp_decoder
        return self

    def unwrap_from_ddp(self) -> "LinkPredictionGNN":
        """
        Unwraps the model from DistributedDataParallel if it is wrapped.

        Returns:
            LinkPredictionGNN: A new instance of LinkPredictionGNN with the original encoder and decoder.
        """
        if isinstance(self._encoder, DistributedDataParallel):
            encoder = self._encoder.module
        else:
            encoder = self._encoder

        if isinstance(self._decoder, DistributedDataParallel):
            decoder = self._decoder.module
        else:
            decoder = self._decoder

        return LinkPredictionGNN(encoder=encoder, decoder=decoder)


def _get_feature_key(node_type: Union[str, NodeType]) -> str:
    """
    Get the feature key for a node type's embedding table.

    Args:
        node_type: Node type as string or NodeType object.

    Returns:
        str: Feature key in format "{node_type}_id"
    """
    return f"{node_type}_id"


# TODO(swong3): Move specific models to gigl.nn.models whenever we restructure model placement.
# TODO(swong3): Abstract TorchRec functionality, and make this LightGCN specific
# TODO(swong3): Remove device context from LightGCN module (use meta, but will have to figure out how to handle buffer transfer)
class LightGCN(nn.Module):
    """
    LightGCN model with TorchRec integration for distributed ID embeddings.

    Reference: https://arxiv.org/pdf/2002.02126

    This class extends the basic LightGCN implementation to use TorchRec's
    distributed embedding tables for handling large-scale ID embeddings.

    Args:
        node_type_to_num_nodes (Union[int, Dict[NodeType, int]]): Map from node types
            to node counts. Can also pass a single int for homogeneous graphs.
        embedding_dim (int): Dimension of node embeddings D. Default: 64.
        num_layers (int): Number of LightGCN propagation layers K. Default: 2.
        device (torch.device): Device to run the computation on. Default: CPU.
        layer_weights (Optional[List[float]]): Weights for [e^(0), e^(1), ..., e^(K)].
            Must have length K+1. If None, uses uniform weights 1/(K+1). Default: None.
    """

    def __init__(
        self,
        node_type_to_num_nodes: Union[int, dict[NodeType, int]],
        embedding_dim: int = 64,
        num_layers: int = 2,
        device: torch.device = torch.device("cpu"),
        layer_weights: Optional[list[float]] = None,
    ):
        super().__init__()

        self._node_type_to_num_nodes = to_heterogeneous_node(node_type_to_num_nodes)
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._device = device

        # Construct LightGCN α weights: include e^(0) + K propagated layers ==> K+1 weights
        if layer_weights is None:
            layer_weights = [1.0 / (num_layers + 1)] * (num_layers + 1)
        else:
            if len(layer_weights) != (num_layers + 1):
                raise ValueError(
                    f"layer_weights must have length K+1={num_layers+1}, got {len(layer_weights)}"
                )

        # Register layer weights as a buffer so it moves with the model to different devices
        self.register_buffer(
            "_layer_weights",
            torch.tensor(layer_weights, dtype=torch.float32),
        )

        # Build TorchRec EBC (one table per node type)
        self._feature_keys: list[str] = [
            _get_feature_key(node_type) for node_type in self._node_type_to_num_nodes.keys()
        ]

        # Validate model configuration: restrict to homogeneous or bipartite graphs
        num_node_types = len(self._feature_keys)
        if num_node_types not in [1, 2]:
            raise ValueError(
                f"LightGCN only supports homogeneous (1 node type) or bipartite (2 node types) graphs; "
                f"got {num_node_types} node types: {self._feature_keys}"
            )

        tables: list[EmbeddingBagConfig] = []
        for node_type, num_nodes in self._node_type_to_num_nodes.items():
            tables.append(
                EmbeddingBagConfig(
                    name=f"node_embedding_{node_type}",
                    embedding_dim=embedding_dim,
                    num_embeddings=num_nodes,
                    feature_names=[_get_feature_key(node_type)],
                )
            )

        self._embedding_bag_collection = EmbeddingBagCollection(
            tables=tables, device=self._device
        )

        # Construct LightGCN propagation layers (LGConv = Ā X)
        self._convs = nn.ModuleList(
            [LGConv() for _ in range(self._num_layers)]
        )  # K layers

    def forward(
        self,
        data: Union[Data, HeteroData],
        device: torch.device,
        output_node_types: Optional[list[NodeType]] = None,
        anchor_node_ids: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, dict[NodeType, torch.Tensor]]:
        """
        Forward pass of the LightGCN model.

        Args:
            data (Union[Data, HeteroData]): Graph data.
                - For homogeneous: Data object with edge_index and node field
                - For heterogeneous: HeteroData with node types and edge_index_dict
            device (torch.device): Device to run the computation on.
            output_node_types (Optional[List[NodeType]]): Node types to return embeddings for.
                Required for heterogeneous graphs. If None, returns embeddings for all node types. Default: None.
            anchor_node_ids (Optional[Union[torch.Tensor, Dict[NodeType, torch.Tensor]]]):
                Local node indices to return embeddings for.
                - For homogeneous: torch.Tensor of shape [num_anchors]
                - For heterogeneous: dict mapping node types to anchor tensors
                If None, returns embeddings for all nodes. Default: None.

        Returns:
            Union[torch.Tensor, Dict[NodeType, torch.Tensor]]: Node embeddings.
                - For homogeneous: tensor of shape [num_nodes, embedding_dim]
                - For heterogeneous: dict mapping node types to embeddings
        """
        is_heterogeneous = isinstance(data, HeteroData)

        if is_heterogeneous:
            # For heterogeneous graphs, anchor_node_ids must be a dict, not a Tensor
            if anchor_node_ids is not None and not isinstance(anchor_node_ids, dict):
                raise TypeError(
                    f"For heterogeneous graphs, anchor_node_ids must be a dict or None, "
                    f"got {type(anchor_node_ids)}"
                )
            return self._forward_heterogeneous(data, device, output_node_types, anchor_node_ids)
        else:
            # For homogeneous graphs, anchor_node_ids must be a Tensor, not a dict
            if anchor_node_ids is not None and not isinstance(anchor_node_ids, torch.Tensor):
                raise TypeError(
                    f"For homogeneous graphs, anchor_node_ids must be a Tensor or None, "
                    f"got {type(anchor_node_ids)}"
                )
            return self._forward_homogeneous(data, device, anchor_node_ids)

    def _forward_homogeneous(
        self,
        data: Data,
        device: torch.device,
        anchor_node_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for homogeneous graphs using LightGCN propagation.

        Notation follows the LightGCN paper (https://arxiv.org/pdf/2002.02126):
        - e^(0): Initial embeddings (no propagation)
        - e^(k): Embeddings after k layers of graph convolution
        - z: Final embedding = weighted sum of [e^(0), e^(1), ..., e^(K)]

        Variable naming:
        - embeddings_0: Initial embeddings e^(0) for subgraph nodes
        - embeddings_k: Current layer embeddings during propagation
        - all_layer_embeddings: List containing [e^(0), e^(1), ..., e^(K)]
        - final_embeddings: Final node embeddings (weighted sum)

        Args:
            data (Data): PyG Data object containing edge_index and node IDs.
            device (torch.device): Device to run computation on.
            anchor_node_ids (Optional[torch.Tensor]): Local node indices to return
                embeddings for. If None, returns embeddings for all nodes. Default: None.

        Returns:
            torch.Tensor: Tensor of shape [num_nodes, embedding_dim] containing
                final LightGCN embeddings.
        """
        # Check if model is setup to be homogeneous
        assert len(self._feature_keys) == 1, (
            f"Homogeneous path expects exactly one node type; got "
            f"{len(self._feature_keys)} types: {self._feature_keys}"
        )
        key = self._feature_keys[0]
        edge_index = data.edge_index.to(
            device
        )  # shape [2, E], where E is the number of edges

        assert hasattr(
            data, "node"
        ), "Subgraph must include .node to map local→global IDs."
        global_ids = data.node.to(
            device
        ).long()  # shape [N_sub], maps local 0..N_sub-1 → global ids

        embeddings_0 = self._lookup_embeddings_for_single_node_type(
            key, global_ids
        )  # shape [N_sub, D], where N_sub is number of nodes in subgraph and D is embedding_dim

        # When using DMP, EmbeddingBagCollection returns Awaitable that needs to be resolved
        if isinstance(embeddings_0, Awaitable):
            embeddings_0 = embeddings_0.wait()

        all_layer_embeddings: list[torch.Tensor] = [embeddings_0]
        embeddings_k = embeddings_0

        for conv in self._convs:
            embeddings_k = conv(
                embeddings_k, edge_index
            )  # shape [N_sub, D], normalized neighbor averaging over *subgraph* edges
            all_layer_embeddings.append(embeddings_k)

        final_embeddings = self._weighted_layer_sum(
            all_layer_embeddings
        )  # shape [N_sub, D], weighted sum of all layer embeddings

        # If anchor node ids are provided, return the embeddings for the anchor nodes only
        if anchor_node_ids is not None:
            anchors_local = anchor_node_ids.to(device).long()  # shape [num_anchors]
            return final_embeddings[
                anchors_local
            ]  # shape [num_anchors, D], embeddings for anchor nodes only

        # Otherwise, return the embeddings for all nodes in the subgraph
        return (
            final_embeddings  # shape [N_sub, D], embeddings for all nodes in subgraph
        )

    def _forward_heterogeneous(
        self,
        data: HeteroData,
        device: torch.device,
        output_node_types: Optional[list[NodeType]] = None,
        anchor_node_ids: Optional[dict[NodeType, torch.Tensor]] = None,
    ) -> dict[NodeType, torch.Tensor]:
        """
        Forward pass for heterogeneous graphs using LightGCN propagation.

        For heterogeneous graphs (e.g., user-item), we have
        multiple node types. Note that we restrict to one edge type. LightGCN propagates embeddings across
        all node types by creating a unified node space, running propagation, then splitting
        back into per-type embeddings.

        Args:
            data (HeteroData): PyG HeteroData object with node types.
            device (torch.device): Device to run computation on.
            output_node_types (Optional[List[NodeType]]): Node types to return embeddings for.
                If None, returns all node types. Default: None.
            anchor_node_ids (Optional[Dict[NodeType, torch.Tensor]]): Dict mapping node types
                to local anchor indices. If None, returns all nodes. Default: None.

        Returns:
            Dict[NodeType, torch.Tensor]: Dict mapping node types to their embeddings,
                each of shape [num_nodes_of_type, embedding_dim].
        """
        # Determine which node types to process
        if output_node_types is None:
            # Sort node types for deterministic ordering across machines
            output_node_types = sorted([NodeType(str(nt)) for nt in data.node_types], key=str)

        # Lookup initial embeddings e^(0) for each node type
        node_type_to_embeddings_0: dict[NodeType, torch.Tensor] = {}

        for node_type in output_node_types:
            node_type_str = str(node_type)
            key = _get_feature_key(node_type_str)

            assert hasattr(data[node_type_str], "node"), (
                f"Subgraph must include .node field for node type {node_type_str}"
            )

            global_ids = data[node_type_str].node.to(device).long()  # shape [N_type]

            embeddings = self._lookup_embeddings_for_single_node_type(
                key, global_ids
            )  # shape [N_type, D]

            # Handle DMP Awaitable
            if isinstance(embeddings, Awaitable):
                embeddings = embeddings.wait()

            node_type_to_embeddings_0[node_type] = embeddings

        # LightGCN propagation across node types
        # Sort node types for deterministic ordering across machines
        all_node_types = sorted(node_type_to_embeddings_0.keys(), key=str)

        # For heterogeneous graphs, we need to create a unified edge representation
        # Collect all edges and map node indices to a combined space
        # E.g., node type 0 gets indices [0, num_type_0), node type 1 gets [num_type_0, num_type_0 + num_type_1)
        node_type_to_offset: dict[NodeType, int] = {}
        offset = 0
        for node_type in all_node_types:
            node_type_to_offset[node_type] = offset
            node_type_str = str(node_type)
            offset += data[node_type_str].num_nodes

        # Combine all embeddings into a single tensor
        combined_embeddings_0 = torch.cat(
            [node_type_to_embeddings_0[nt] for nt in all_node_types], dim=0
        )  # shape [total_nodes, D]

        # Combine all edges into a single edge_index
        combined_edge_list: list[torch.Tensor] = []
        for edge_type_tuple in data.edge_types:
            src_nt_str, _, dst_nt_str = edge_type_tuple
            src_node_type = NodeType(src_nt_str)
            dst_node_type = NodeType(dst_nt_str)

            edge_index = data[edge_type_tuple].edge_index.to(device)  # shape [2, E]

            # Offset the indices to the combined node space
            src_offset = node_type_to_offset[src_node_type]
            dst_offset = node_type_to_offset[dst_node_type]

            offset_edge_index = edge_index.clone()
            offset_edge_index[0] += src_offset
            offset_edge_index[1] += dst_offset

            combined_edge_list.append(offset_edge_index)

        combined_edge_index = torch.cat(combined_edge_list, dim=1)  # shape [2, total_edges]

        # Track all layer embeddings
        all_layer_embeddings: list[torch.Tensor] = [combined_embeddings_0]
        current_embeddings = combined_embeddings_0

        # Perform K layers of propagation
        for conv in self._convs:
            current_embeddings = conv(current_embeddings, combined_edge_index)  # shape [total_nodes, D]
            all_layer_embeddings.append(current_embeddings)

        # Weighted sum across layers
        combined_final_embeddings = self._weighted_layer_sum(all_layer_embeddings)  # shape [total_nodes, D]

        # Split back into per-node-type embeddings
        final_embeddings: dict[NodeType, torch.Tensor] = {}
        for node_type in all_node_types:
            start_idx = node_type_to_offset[node_type]
            node_type_str = str(node_type)
            num_nodes = data[node_type_str].num_nodes
            end_idx = start_idx + num_nodes

            final_embeddings[node_type] = combined_final_embeddings[start_idx:end_idx]  # shape [num_nodes, D]

        # Extract anchor nodes if specified
        if anchor_node_ids is not None:
            for node_type in all_node_types:
                if node_type in anchor_node_ids:
                    anchors = anchor_node_ids[node_type].to(device).long()
                    final_embeddings[node_type] = final_embeddings[node_type][anchors]

        return final_embeddings

    def _lookup_embeddings_for_single_node_type(
        self, node_type: str, ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Fetch per-ID embeddings for a single node type using EmbeddingBagCollection.

        This method constructs a KeyedJaggedTensor (KJT) that includes all EBC feature
        keys to ensure consistent forward pass behavior. For the requested node type,
        we create B bags of length 1 (one per ID). For all other node types, we create
        B bags of length 0. With SUM pooling, non-requested node types contribute zeros
        and the requested node type acts as identity lookup.

        Args:
            node_type (str): Feature key for the node type (e.g., "user_id", "item_id").
            ids (torch.Tensor): Node IDs to look up, shape [batch_size].

        Returns:
            torch.Tensor: Embeddings for the requested node type, shape [batch_size, embedding_dim].
        """
        if node_type not in self._feature_keys:
            raise KeyError(
                f"Unknown feature key '{node_type}'. Valid keys: {self._feature_keys}"
            )

        # Number of examples (one ID per "bag")
        batch_size = int(ids.numel())  # B is the number of node IDs to lookup
        device = ids.device

        # Build lengths in key-major order: for each key, we give B lengths.
        # - requested key: ones (each example has 1 id)
        # - other keys: zeros (each example has 0 ids)
        lengths_per_key: list[torch.Tensor] = []
        for nt in self._feature_keys:
            if nt == node_type:
                lengths_per_key.append(
                    torch.ones(batch_size, dtype=torch.long, device=device)
                )  # shape [B], all ones for requested key
            else:
                lengths_per_key.append(
                    torch.zeros(batch_size, dtype=torch.long, device=device)
                )  # shape [B], all zeros for other keys

        lengths = torch.cat(
            lengths_per_key, dim=0
        )  # shape [batch_size * num_keys], concatenated lengths for all keys

        # Values only contain the requested key's ids (sum of other lengths is 0)
        kjt = KeyedJaggedTensor(
            keys=self._feature_keys,  # include ALL keys known by EBC
            values=ids.long(),  # shape [batch_size], only batch_size values for the requested key
            lengths=lengths,  # shape [batch_size * num_keys], batch_size lengths per key, concatenated key-major
        )

        out = self._embedding_bag_collection(
            kjt
        )  # KeyedTensor (dict-like): out[key] -> [batch_size, D]
        return out[node_type]  # shape [batch_size, D], embeddings for the requested key

    def _weighted_layer_sum(
        self, all_layer_embeddings: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes weighted sum: w_0 * e^(0) + w_1 * e^(1) + ... + w_K * e^(K).

        This implements the final aggregation step in LightGCN where embeddings from
        all layers (including the initial e^(0)) are combined using learned weights.

        Args:
            all_layer_embeddings (List[torch.Tensor]): List [e^(0), e^(1), ..., e^(K)]
                where each tensor has shape [N, D].

        Returns:
            torch.Tensor: Weighted sum of all layer embeddings, shape [N, D].
        """
        if len(all_layer_embeddings) != len(self._layer_weights):
            raise ValueError(
                f"Got {len(all_layer_embeddings)} layer tensors but {len(self._layer_weights)} weights."
            )

        # Stack all layer embeddings and compute weighted sum
        # _layer_weights is already a tensor buffer registered in __init__
        stacked = torch.stack(all_layer_embeddings, dim=0)  # shape [K+1, N, D]
        w = self._layer_weights.to(stacked.device)  # shape [K+1], ensure on same device
        out = (stacked * w.view(-1, 1, 1)).sum(
            dim=0
        )  # shape [N, D], w_0*X_0 + w_1*X_1 + ...

        return out
