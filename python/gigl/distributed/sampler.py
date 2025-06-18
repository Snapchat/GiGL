from typing import Any, Optional, Union

import torch
from graphlearn_torch.sampler import NodeSamplerInput

from gigl.src.common.types.graph_data import NodeType


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
        positive_labels: Optional[torch.Tensor],
        negative_labels: Optional[torch.Tensor],
        supervision_node_type: Optional[Union[str, NodeType]],
    ):
        """
        Args:
            node (torch.Tensor): Anchor nodes to fanout from
            input_type (Optional[Union[str, NodeType]]): Node type of the anchor nodes
            positive_labels (torch.Tensor): Positive label nodes to fanout from
            negative_labels (Optional[torch.Tensor]): Negative label nodes to fanout from
            supervision_node_type (Optional[Union[str, NodeType]]): Node type of the positive and negative labels. GiGL
                currently only supports one supervision node type, this may be revisited in the future
        """
        super().__init__(node, input_type)
        self.positive_labels = positive_labels
        self.negative_labels = negative_labels
        self.supervision_node_type = supervision_node_type

    def __len__(self) -> int:
        return self.node.shape[0]

    def __getitem__(self, index: Union[torch.Tensor, Any]) -> "ABLPNodeSamplerInput":
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index, dtype=torch.long)
        index = index.to(self.node.device)
        return ABLPNodeSamplerInput(
            node=self.node[index],
            input_type=self.input_type,
            positive_labels=self.positive_labels[index]
            if self.positive_labels is not None
            else None,
            negative_labels=self.negative_labels[index]
            if self.negative_labels is not None
            else None,
            supervision_node_type=self.supervision_node_type,
        )
