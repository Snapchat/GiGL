from typing import Any, Final, Optional, Union

import torch
from gigl.src.common.types.graph_data import EdgeType, NodeType
from graphlearn_torch.sampler import NodeSamplerInput

POSITIVE_LABEL_METADATA_KEY: Final[str] = "gigl_positive_labels."
NEGATIVE_LABEL_METADATA_KEY: Final[str] = "gigl_negative_labels."


def metadata_key_with_prefix(key: str) -> str:
    """Prefixes the key with "#META
    Do this as GLT also does this.
    https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/python/distributed/dist_neighbor_sampler.py#L714
    """
    return f"#META.{key}"


class ABLPNodeSamplerInput(NodeSamplerInput):
    """
    Sampler input specific for ABLP use case. Contains additional information about positive labels, negative labels, and the corresponding
    supervision node type
    """

    def __init__(
        self,
        node: torch.Tensor,
        input_type: Optional[Union[str, NodeType]],
        positive_label_by_edge_types: dict[EdgeType, torch.Tensor],
        negative_label_by_edge_types: dict[EdgeType, torch.Tensor],
    ):
        """
        Args:
            node (torch.Tensor): Anchor nodes to fanout from
            input_type (Optional[Union[str, NodeType]]): Node type of the anchor nodes
            positive_label_by_edge_types (dict[EdgeType, torch.Tensor]): Positive label nodes to fanout from
            negative_label_by_edge_types (dict[EdgeType, torch.Tensor]): Negative label nodes to fanout from
        """
        super().__init__(node, input_type)

        self._positive_label_by_edge_types = positive_label_by_edge_types
        self._negative_label_by_edge_types = negative_label_by_edge_types

    @property
    def positive_label_by_edge_types(self) -> dict[EdgeType, torch.Tensor]:
        return self._positive_label_by_edge_types

    @property
    def negative_label_by_edge_types(self) -> dict[EdgeType, torch.Tensor]:
        return self._negative_label_by_edge_types

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
                edge_type: self._positive_label_by_edge_types[edge_type][index]
                for edge_type in self._positive_label_by_edge_types
            },
            negative_label_by_edge_types={
                edge_type: self._negative_label_by_edge_types[edge_type][index]
                for edge_type in self._negative_label_by_edge_types
            },
        )

    def __repr__(self) -> str:
        return f"ABLPNodeSamplerInput(\n\tnode={self.node},\n\tinput_type={self.input_type},\n\tpositive_label_by_edge_types={self._positive_label_by_edge_types},\n\tnegative_label_by_edge_types={self._negative_label_by_edge_types}\n)"
