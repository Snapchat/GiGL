from typing import Any, Optional, Union

import torch
from graphlearn_torch.sampler import NodeSamplerInput

from gigl.src.common.types.graph_data import NodeType


class LabeledNodeSamplerInput(NodeSamplerInput):
    def __init__(
        self,
        node: torch.Tensor,
        input_type: Optional[Union[str, NodeType]],
        positive_labels: torch.Tensor,
        negative_labels: Optional[torch.Tensor],
        supervision_node_type: Optional[Union[str, NodeType]],
    ):
        super().__init__(node, input_type)
        self.positive_labels = positive_labels
        self.negative_labels = negative_labels
        self.supervision_node_type = supervision_node_type

    def __len__(self) -> int:
        return self.node.shape[0]

    def __getitem__(self, index: Union[torch.Tensor, Any]) -> "LabeledNodeSamplerInput":
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index, dtype=torch.long)
        index = index.to(self.node.device)
        return LabeledNodeSamplerInput(
            node=self.node[index],
            input_type=self.input_type,
            positive_labels=self.positive_labels[index],
            negative_labels=self.negative_labels[index]
            if self.negative_labels is not None
            else None,
            supervision_node_type=self.supervision_node_type,
        )
