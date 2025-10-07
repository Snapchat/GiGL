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

# TODO(swong3): Split into different files
class LightGCN(nn.Module):
    """
    LightGCN model with TorchRec integration for distributed ID embeddings.

    https://arxiv.org/pdf/2002.02126

    This class extends the basic LightGCN implementation to use TorchRec's
    distributed embedding tables for handling large-scale ID embeddings.

    Args:
        node_type_to_num_nodes (Dict[NodeType, int]): map node types to counts
        embedding_dim (int): Dimension of node embeddings D (default 64)
        num_layers (int): K LightGCN propagation hops (default 2)
        layer_weights (Optional[List[float]]): weights for [e^(0), e^(1), ..., e^(K)]
            If None, uses uniform 1/(K+1).
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

        if isinstance(node_type_to_num_nodes, int):
            node_type_to_num_nodes = {NodeType("default_node_type"): node_type_to_num_nodes}

        self._node_type_to_num_nodes = node_type_to_num_nodes
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._device = device

        # Construct LightGCN α weights: include e^(0) + K propagated layers ==> K+1 weights
        if layer_weights is None:
            self._layer_weights = [1.0 / (num_layers + 1)] * (num_layers + 1)
        else:
            if len(layer_weights) != (num_layers + 1):
                raise ValueError(
                    f"layer_weights must have length K+1={num_layers+1}, got {len(layer_weights)}"
                )
            self._layer_weights = layer_weights
        self.register_buffer(
            "_layer_weights_tensor", torch.tensor(self._layer_weights, dtype=torch.float32, device=device)  # shape [K+1], where K is num_layers
        )

        # Build TorchRec EBC (one table per node type)
        # feature key naming convention: f"{node_type}_id"
        self._feature_keys: list[str] = [f"{nt}_id" for nt in node_type_to_num_nodes.keys()]
        tables: list[EmbeddingBagConfig] = []
        for nt, n in node_type_to_num_nodes.items():
            tables.append(
                EmbeddingBagConfig(
                    name=f"node_embedding_{nt}",
                    embedding_dim=embedding_dim,
                    num_embeddings=int(n),
                    feature_names=[f"{nt}_id"],
                )
            )

        self._embedding_bag_collection = EmbeddingBagCollection(tables=tables, device=self._device)

        # Construct LightGCN propagation layers (LGConv = Ā X)
        self._convs = nn.ModuleList([LGConv() for _ in range(self._num_layers)])  # K layers

        self._is_sharded = False

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
            data: Graph data (homogeneous or heterogeneous)
            device: Device to run the computation on
            output_node_types: List of node types to return embeddings for (for heterogeneous graphs)
            anchor_node_ids: Node IDs to return embeddings for (for homogeneous graphs)

        Returns:
            Node embeddings for the specified node types
        """
        if isinstance(data, HeteroData):
            output_node_types = output_node_types or list(data.node_types)
            return self._forward_heterogeneous(data, device, output_node_types)
        else:
            return self._forward_homogeneous(data, device, anchor_node_ids)

    def _forward_homogeneous(self, data: Data, device: torch.device, anchor_node_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Check if model is setup to be homogeneous
        if len(self._feature_keys) != 1:
            raise ValueError(
                "Homogeneous path expects exactly one node type; got "
                f"{len(self._feature_keys)} types: {self._feature_keys}"
            )
        key = self._feature_keys[0]
        edge_index = data.edge_index.to(device)  # shape [2, E], where E is the number of edges

        if hasattr(data, "n_id"):
            global_ids = data.n_id.to(device).long() # shape [N_sub], maps local 0..N_sub-1 → global ids
        elif hasattr(data, "node_id"):
            global_ids = data.node_id.to(device).long() # shape [N_sub], maps local 0..N_sub-1 → global ids
        else:
            raise ValueError("Subgraph must include .n_id (or .node_id) to map local→global IDs.")

        x0_sub = self._lookup_single_key(key, global_ids)   # shape [N_sub, D], where N_sub is number of nodes in subgraph and D is embedding_dim

        xs: list[torch.Tensor] = [x0_sub]
        x = x0_sub

        for conv in self._convs:
            x = conv(x, edge_index)     # shape [N_sub, D], normalized neighbor averaging over *subgraph* edges
            xs.append(x)

        z_sub = self._weighted_layer_sum(xs)  # shape [N_sub, D], weighted sum of all layer embeddings

        if anchor_node_ids is not None:
            anchors_local = anchor_node_ids.to(device).long()  # shape [num_anchors]
            return z_sub[anchors_local]      # shape [num_anchors, D], embeddings for anchor nodes only

        return z_sub                         # shape [N_sub, D], embeddings for all nodes in subgraph

    def _forward_heterogeneous(
        self,
        data: HeteroData,
        device: torch.device,
        output_node_types: list[NodeType],
    ) -> dict[NodeType, torch.Tensor]:

        raise NotImplementedError("Heterogeneous forward pass is not implemented")
        node_type_to_ids: dict[str, torch.Tensor] = {}
        for nt in data.node_types:
            if hasattr(data[nt], "node_id"):
                node_type_to_ids[nt] = data[nt].node_id.to(device)
            else:
                node_type_to_ids[nt] = torch.arange(
                    data[nt].num_nodes, device=device, dtype=torch.long
                )

        node_type_to_x0: dict[str, torch.Tensor] = {}
        for nt, ids in node_type_to_ids.items():
            node_type_to_x0[nt] = self._lookup_single_key(f"{nt}_id", ids)

        print(node_type_to_x0)

        node_type_to_xs: dict[str, list[torch.Tensor]] = {nt: [x0] for nt, x0 in node_type_to_x0.items()}
        node_type_to_x = {nt: x0 for nt, x0 in node_type_to_x0.items()}

        print(node_type_to_xs)
        print(node_type_to_x)

        for _layer in range(self._num_layers):
            accum = {nt: torch.zeros_like(node_type_to_x[nt]) for nt in node_type_to_x.keys()}
            degs = {nt: 0 for nt in node_type_to_x.keys()}

            for (src_nt, _rel, dst_nt), eidx in data.edge_index_dict.items():
                conv = LGConv()
                accum[dst_nt] = accum[dst_nt] + conv(node_type_to_x[src_nt], eidx.to(device))
                degs[dst_nt] += 1
                accum[src_nt] = accum[src_nt] + conv(node_type_to_x[dst_nt], eidx.flip(0).to(device))
                degs[src_nt] += 1

            for nt in accum.keys():
                if degs[nt] > 0:
                    node_type_to_x[nt] = accum[nt] / float(degs[nt])
                node_type_to_xs[nt].append(node_type_to_x[nt])

        final_embeddings: dict[str, torch.Tensor] = {}
        for nt in output_node_types:
            if nt not in node_type_to_xs:
                final_embeddings[nt] = torch.empty(0, self._embedding_dim, device=device)
                continue
            final_embeddings[nt] = self._weighted_layer_sum(node_type_to_xs[nt])

        return final_embeddings

    def _lookup_single_key(self, key: str, ids: torch.Tensor) -> torch.Tensor:
        """
        Fetch per-id embeddings for a single feature key using EBC and KJT.

        Construct a KJT that includes *all* EBC keys so the forward path is
        consistent. For the requested key, we create B bags of length 1 (each ID).
        For all other keys, we create B bags of length 0. SUM pooling then makes
        non-requested keys contribute zeros, and the requested key is identity.
        """
        if key not in self._feature_keys:
            raise KeyError(f"Unknown feature key '{key}'. Valid keys: {self._feature_keys}")

        # Number of examples (one ID per "bag")
        B = int(ids.numel())  # B is the number of node IDs to lookup
        device = ids.device

        # Build lengths in key-major order: for each key, we give B lengths.
        # - requested key: ones (each example has 1 id)
        # - other keys: zeros (each example has 0 ids)
        lengths_per_key: list[torch.Tensor] = []
        for k in self._feature_keys:
            if k == key:
                lengths_per_key.append(torch.ones(B, dtype=torch.long, device=device))  # shape [B], all ones for requested key
            else:
                lengths_per_key.append(torch.zeros(B, dtype=torch.long, device=device))  # shape [B], all zeros for other keys

        lengths = torch.cat(lengths_per_key, dim=0)  # shape [B * num_keys], concatenated lengths for all keys

        # Values only contain the requested key's ids (sum of other lengths is 0)
        kjt = KeyedJaggedTensor(
            keys=self._feature_keys,       # include ALL keys known by EBC
            values=ids.long(),             # shape [B], only B values for the requested key
            lengths=lengths,               # shape [B * num_keys], B lengths per key, concatenated key-major
        )

        out = self._embedding_bag_collection(kjt)              # KeyedTensor (dict-like): out[key] -> [B, D]
        return out[key]                   # shape [B, D], embeddings for the requested key

    def _weighted_layer_sum(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """
        xs: [e^(0), e^(1), ..., e^(K)]  each of shape [N, D]
        returns: [N, D]
        """
        if len(xs) != int(self._layer_weights_tensor.numel()):
            raise ValueError(
                f"Got {len(xs)} layer tensors but {self._layer_weights_tensor.numel()} weights."
            )

        # Ensure weights match device/dtype of embeddings
        w = self._layer_weights_tensor.to(device=xs[0].device, dtype=xs[0].dtype)      # shape [K+1], layer weights
        stacked = torch.stack(xs, dim=0)                                              # shape [K+1, N, D], stack all layer embeddings
        out = (stacked * w.view(-1, 1, 1)).sum(dim=0)                                  # shape [N, D], weighted sum across layers

        return out
