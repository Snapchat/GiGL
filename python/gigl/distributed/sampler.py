from typing import Any, Optional, Union

import torch
from graphlearn_torch.sampler import NodeSamplerInput

from gigl.src.common.types.graph_data import EdgeType, NodeType


class ABLPNodeSamplerInput(NodeSamplerInput):
    """
    Sampler input specific for ABLP use case. Contains additional information about positive labels, negative labels, and the corresponding
    supervision node type
    """

    def __init__(
        self,
        node: torch.Tensor,
        input_type: Optional[Union[str, NodeType]],
        # TODO (mkolodner-sc): Support multiple positive and negative label node types
        positive_label_by_edge_types: dict[EdgeType, torch.Tensor],
        negative_label_by_edge_types: Optional[dict[EdgeType, torch.Tensor]],
    ):
        """
        Args:
            node (torch.Tensor): Anchor nodes to fanout from
            input_type (Optional[Union[str, NodeType]]): Node type of the anchor nodes
            positive_labels (torch.Tensor): Positive label nodes to fanout from
            negative_labels (Optional[torch.Tensor]): Negative label nodes to fanout from
            supervision_node_types (Optional[list[Union[str, NodeType]]]): Node type of the positive and negative labels. GiGL
                currently only supports one supervision node type, this may be revisited in the future
        """
        super().__init__(node, input_type)

        self.positive_label_by_edge_types = positive_label_by_edge_types
        self.negative_label_by_edge_types = negative_label_by_edge_types

    def __len__(self) -> int:
        return self.node.shape[0]

    def __getitem__(self, index: Union[torch.Tensor, Any]) -> "ABLPNodeSamplerInput":
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index, dtype=torch.long)
        index = index.to(self.node.device)
        return ABLPNodeSamplerInput(
            node=self.node[index],
            input_type=self.input_type,
            positive_label_by_edge_types={
                edge_type: self.positive_label_by_edge_types[edge_type][index]
                for edge_type in self.positive_label_by_edge_types
            },
            negative_label_by_edge_types={
                edge_type: self.negative_label_by_edge_types[edge_type][index]
                for edge_type in self.negative_label_by_edge_types
            }
            if self.negative_label_by_edge_types is not None
            else None,
        )

    def __repr__(self) -> str:
        return f"ABLPNodeSamplerInput(\n\tnode={self.node},\n\tinput_type={self.input_type},\n\tpositive_label_by_edge_types={self.positive_label_by_edge_types},\n\tnegative_label_by_edge_types={self.negative_label_by_edge_types}\n)"
