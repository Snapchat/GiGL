from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import DeviceDict
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn.conv import LGConv
from torch_geometric import utils
from typing_extensions import Self
import torch.distributed as dist
from torch_sparse import SparseTensor

from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
)

from torchrec.distributed.planner import Topology, EmbeddingShardingPlanner
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel

from gigl.src.common.types.graph_data import NodeType
from gigl.types.graph import to_homogeneous
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

# TODO(swong3): Move specific models to gigl.nn.models whenever we restructure model placement.
# TODO(swong3): Abstract TorchRec functionality, and make this LightGCN specific
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
        # feature key naming convention: f"{node_type}_id"
        self._feature_keys: list[str] = [f"{node_type}_id" for node_type in self._node_type_to_num_nodes.keys()]
        tables: list[EmbeddingBagConfig] = []
        for node_type, num_nodes in self._node_type_to_num_nodes.items():
            tables.append(
                EmbeddingBagConfig(
                    name=f"node_embedding_{node_type}",
                    embedding_dim=embedding_dim,
                    num_embeddings=num_nodes,
                    feature_names=[f"{node_type}_id"],
                )
            )

        self._embedding_bag_collection = EmbeddingBagCollection(tables=tables, device=self._device)

        # Construct LightGCN propagation layers (LGConv = Ā X)
        self._convs = nn.ModuleList([LGConv() for _ in range(self._num_layers)])  # K layers

    def forward(
        self,
        data: Union[Data, HeteroData],
        device: torch.device,
        output_node_types: Optional[list[NodeType]] = None,
        anchor_node_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, dict[NodeType, torch.Tensor]]:
        """
        Forward pass of the LightGCN model.

        Args:
            data (Union[Data, HeteroData]): Graph data (homogeneous or heterogeneous).
            device (torch.device): Device to run the computation on.
            output_node_types (Optional[List[NodeType]]): List of node types to return
                embeddings for. Required for heterogeneous graphs. Default: None.
            anchor_node_ids (Optional[torch.Tensor]): Local node indices to return
                embeddings for. If None, returns embeddings for all nodes. Default: None.

        Returns:
            Union[torch.Tensor, Dict[NodeType, torch.Tensor]]: Node embeddings.
                For homogeneous graphs, returns tensor of shape [num_nodes, embedding_dim].
                For heterogeneous graphs, returns dict mapping node types to embeddings.
        """
        if isinstance(data, HeteroData):
            raise NotImplementedError("HeteroData is not yet supported for LightGCN")
            output_node_types = output_node_types or list(data.node_types)
            return self._forward_heterogeneous(data, device, output_node_types, anchor_node_ids)
        else:
            return self._forward_homogeneous(data, device, anchor_node_ids)

    def _forward_homogeneous(self, data: Data, device: torch.device, anchor_node_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        edge_index = data.edge_index.to(device)  # shape [2, E], where E is the number of edges

        assert hasattr(data, "node"), "Subgraph must include .node to map local→global IDs."
        global_ids = data.node.to(device).long() # shape [N_sub], maps local 0..N_sub-1 → global ids

        embeddings_0 = self._lookup_embeddings_for_single_node_type(key, global_ids)   # shape [N_sub, D], where N_sub is number of nodes in subgraph and D is embedding_dim

        all_layer_embeddings: list[torch.Tensor] = [embeddings_0]
        embeddings_k = embeddings_0

        for conv in self._convs:
            embeddings_k = conv(embeddings_k, edge_index)     # shape [N_sub, D], normalized neighbor averaging over *subgraph* edges
            all_layer_embeddings.append(embeddings_k)

        final_embeddings = self._weighted_layer_sum(all_layer_embeddings)  # shape [N_sub, D], weighted sum of all layer embeddings

        # If anchor node ids are provided, return the embeddings for the anchor nodes only
        if anchor_node_ids is not None:
            anchors_local = anchor_node_ids.to(device).long()  # shape [num_anchors]
            return final_embeddings[anchors_local]      # shape [num_anchors, D], embeddings for anchor nodes only

        # Otherwise, return the embeddings for all nodes in the subgraph
        return final_embeddings                         # shape [N_sub, D], embeddings for all nodes in subgraph

    def _lookup_embeddings_for_single_node_type(self, node_type: str, ids: torch.Tensor) -> torch.Tensor:
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
            raise KeyError(f"Unknown feature key '{node_type}'. Valid keys: {self._feature_keys}")

        # Number of examples (one ID per "bag")
        batch_size = int(ids.numel()) # B is the number of node IDs to lookup
        device = ids.device

        # Build lengths in key-major order: for each key, we give B lengths.
        # - requested key: ones (each example has 1 id)
        # - other keys: zeros (each example has 0 ids)
        lengths_per_key: list[torch.Tensor] = []
        for nt in self._feature_keys:
            if nt == node_type:
                lengths_per_key.append(torch.ones(batch_size, dtype=torch.long, device=device))  # shape [B], all ones for requested key
            else:
                lengths_per_key.append(torch.zeros(batch_size, dtype=torch.long, device=device))  # shape [B], all zeros for other keys

        lengths = torch.cat(lengths_per_key, dim=0)  # shape [batch_size * num_keys], concatenated lengths for all keys

        # Values only contain the requested key's ids (sum of other lengths is 0)
        kjt = KeyedJaggedTensor(
            keys=self._feature_keys,       # include ALL keys known by EBC
            values=ids.long(),             # shape [batch_size], only batch_size values for the requested key
            lengths=lengths,               # shape [batch_size * num_keys], batch_size lengths per key, concatenated key-major
        )

        out = self._embedding_bag_collection(kjt)              # KeyedTensor (dict-like): out[key] -> [batch_size, D]
        return out[node_type]                   # shape [batch_size, D], embeddings for the requested key

    def _weighted_layer_sum(self, all_layer_embeddings: list[torch.Tensor]) -> torch.Tensor:
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
        stacked = torch.stack(all_layer_embeddings, dim=0)                             # shape [K+1, N, D]
        w = self._layer_weights.to(stacked.device)                                      # shape [K+1], ensure on same device
        out = (stacked * w.view(-1, 1, 1)).sum(dim=0)                                  # shape [N, D], w_0*X_0 + w_1*X_1 + ...

        return out
