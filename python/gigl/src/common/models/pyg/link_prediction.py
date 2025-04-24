from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch_geometric

from gigl.src.common.models.layers.decoder import LinkPredictionDecoder
from gigl.src.common.models.layers.task import NodeAnchorBasedLinkPredictionTasks
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.model import GraphBackend


class LinkPredictionGNN(nn.Module):
    """
    Link Prediction GNN model for both homogeneous and heterogeneous use cases
    Args:
        encoder (nn.Module): Either BasicGNN or Heterogeneous GNN for generating embeddings
        decoder (nn.Module): LinkPredictionDecoder for transforming embeddings into scores
        tasks (NodeAnchorBasedLinkPredictionTasks): Learning tasks for training (i.e. Retrieval, Margin, SSL, ...)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: LinkPredictionDecoder,
        tasks: Optional[NodeAnchorBasedLinkPredictionTasks] = None,
    ) -> None:
        super().__init__()
        self.__encoder = encoder
        self.__decoder = decoder
        self.__tasks = tasks

    def forward(
        self,
        data: Union[
            torch_geometric.data.Data, torch_geometric.data.hetero_data.HeteroData
        ],
        output_node_types: List[NodeType],
        device: torch.device,
    ) -> Dict[NodeType, torch.Tensor]:
        if isinstance(data, torch_geometric.data.hetero_data.HeteroData):
            return self.__encoder(
                data=data, output_node_types=output_node_types, device=device
            )
        else:
            if len(output_node_types) > 1:
                raise NotImplementedError(
                    f"Found {len(output_node_types)} output node types for homogeneous data, which must have one node type"
                )
            return {output_node_types[0]: self.__encoder(data=data, device=device)}

    def decode(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.__decoder(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
        )

    @property
    def tasks(self) -> NodeAnchorBasedLinkPredictionTasks:
        return self.__tasks  # type: ignore

    @property
    def graph_backend(self) -> GraphBackend:
        return GraphBackend.PYG
