from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, Tuple

import torch
import torchrec

from gigl.experimental.knowledge_graph_embedding.common.torchrec.batch import (
    DataclassBatch,
)
from gigl.experimental.knowledge_graph_embedding.lib.config.dataloader import (
    DataloaderConfig,
)
from gigl.experimental.knowledge_graph_embedding.lib.config.sampling import (
    SamplingConfig,
)
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    NodeType,
)
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper


@dataclass
class NodeBatch(DataclassBatch):
    """
    A class for representing a batch of nodes in a heterogeneous graph.
    These nodes share the same condensed_node_type, and inference is
    being run in the context of a single condensed edge type.
    """

    nodes: torchrec.KeyedJaggedTensor
    # node_ids: torch.Tensor  # Denormalized for convenience. Same as `nodes.values()`.
    condensed_node_type: torch.Tensor
    condensed_edge_type: torch.Tensor

    @staticmethod
    def from_node_tensors(
        nodes: torch.Tensor,
        condensed_node_type: torch.Tensor,
        condensed_edge_type: torch.Tensor,
        condensed_node_type_to_node_type_map: Dict[CondensedNodeType, NodeType],
    ) -> NodeBatch:
        """
        Creates a NodeBatch from a range of nodes. Each batch will contain
        nodes of a single condensed node type.  This is useful for inference
        when we want to collect embeddings for a range of nodes.

        Args:
            nodes: torch.Tensor: A tensor containing the node IDs.
            condensed_node_type: torch.Tensor: A tensor representing the condensed node type.
            condensed_edge_type: torch.Tensor: A tensor representing the condensed edge type.
            condensed_node_type_to_node_type_map: Dict[CondensedNodeType, NodeType]: A mapping from condensed node types to node types.

        Returns:
            NodeBatch: The created NodeBatch.
        """

        num_nodes = nodes.size()  # Inclusive of start and end

        cnt_keys = sorted(list(condensed_node_type_to_node_type_map.keys()))
        lengths: Dict[CondensedNodeType, torch.Tensor] = dict()
        values: Dict[CondensedNodeType, torch.Tensor] = dict()

        for cnt_key in cnt_keys:
            lengths[cnt_key] = (
                torch.ones(num_nodes, dtype=torch.int32)
                if cnt_key == condensed_node_type.item()
                else torch.zeros(num_nodes, dtype=torch.int32)
            )
            values[cnt_key] = (
                nodes
                if cnt_key == condensed_node_type.item()
                else torch.empty(0, dtype=torch.int32)
            )

        nodes = torchrec.KeyedJaggedTensor(
            keys=cnt_keys,
            values=torch.cat([values[cnt] for cnt in cnt_keys], dim=0),
            lengths=torch.cat([lengths[cnt] for cnt in cnt_keys], dim=0),
        )
        return NodeBatch(
            nodes=nodes,
            condensed_node_type=condensed_node_type,
            condensed_edge_type=condensed_edge_type,
        )

    def to_node_tensors(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstructs the tensors comprising the NodeBatch.

        Args:
            condensed_node_type_to_node_type_map (Dict[CondensedNodeType, NodeType]): A mapping from condensed node types to node types.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - nodes: torch.Tensor: The node IDs.
                - condensed_node_type: torch.Tensor: The condensed node type.
                - condensed_edge_type: torch.Tensor: The condensed edge type.
        """
        lengths_per_key = torch.tensor(self.nodes.length_per_key())
        assert lengths_per_key.argwhere().ravel().numel() == 1
        assert lengths_per_key.argmax().item() == self.condensed_node_type.item()
        return self.nodes.values(), self.condensed_node_type, self.condensed_edge_type

    @staticmethod
    def build_data_loader(
        dataset: torch.utils.data.IterableDataset,
        condensed_node_type: CondensedNodeType,
        condensed_edge_type: CondensedEdgeType,
        graph_metadata: GraphMetadataPbWrapper,
        sampling_config: SamplingConfig,
        dataloader_config: DataloaderConfig,
        pin_memory: bool,
    ):
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=sampling_config.positive_edge_batch_size,  # todo(nshah): use inference batch size explicitly.
            num_workers=dataloader_config.num_workers,
            pin_memory=pin_memory,
            collate_fn=partial(
                collate_node_batch_from_range,
                condensed_node_type=condensed_node_type,
                condensed_edge_type=condensed_edge_type,
                condensed_node_type_to_node_type_map=graph_metadata.condensed_node_type_to_node_type_map,
            ),
        )


def collate_node_batch_from_range(
    nodes: Iterable[int],
    condensed_node_type: CondensedNodeType,
    condensed_edge_type: CondensedEdgeType,
    condensed_node_type_to_node_type_map: Dict[CondensedNodeType, NodeType],
) -> NodeBatch:
    """
    Collates a batch of nodes into a NodeBatch.
    This is used for inference when we want to collect embeddings for a range of nodes.

    Args:
        nodes (Iterable[int]): An iterable of node IDs.
        condensed_node_type (CondensedNodeType): The condensed node type for the batch.
        condensed_edge_type (CondensedEdgeType): The condensed edge type for the batch (relevant to inference).
        condensed_node_type_to_node_type_map (Dict[CondensedNodeType, NodeType]): A mapping from condensed node types to node types.
    """
    return NodeBatch.from_node_tensors(
        nodes=torch.tensor(nodes, dtype=torch.int32),
        condensed_node_type=torch.tensor(condensed_node_type, dtype=torch.int32),
        condensed_edge_type=torch.tensor(condensed_edge_type, dtype=torch.int32),
        condensed_node_type_to_node_type_map=condensed_node_type_to_node_type_map,
    )
