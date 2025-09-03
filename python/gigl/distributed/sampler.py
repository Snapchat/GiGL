from typing import Any, Optional, Union
from dataclasses import asdict, dataclass
import ast
import json


import torch
from graphlearn_torch.sampler import NodeSamplerInput, RemoteSamplerInput

from gigl.common import Uri
from gigl.types.graph import FeatureInfo
from gigl.src.common.types.graph_data import NodeType, EdgeType
from gigl.src.common.utils.file_loader import FileLoader
from gigl.common.logger import Logger

logger = Logger()


@dataclass
class RemoteNodeInfo:
    node_type: Optional[NodeType]
    edge_types: list[tuple[NodeType, NodeType, NodeType]]
    node_tensor_uri: str
    node_feature_info: Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]]
    edge_feature_info: Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]]
    num_partitions: int
    edge_dir: str

    def dump(self) -> str:
        print(f"{asdict(self)=}")
        print(f"{json.dumps(asdict(self))=}")
        return str(asdict(self))

    @classmethod
    def load(cls, uri: Uri) -> "RemoteNodeInfo":
        logger.info(f"{uri=}")
        tf = FileLoader().load_to_temp_file(uri, should_create_symlinks_if_possible=False)
        with open(tf.name, "r") as f:
            s = f.read()
            logger.info(f"{s=}")
        tf.close()
        logger.info(f"{s=}")
        return cls(**ast.literal_eval(s))

class RemoteUriSamplerInput(RemoteSamplerInput):
    def __init__(self, uri: Uri, input_type: Optional[Union[str, NodeType]]):
        self._uri = uri
        self._input_type = input_type

    def to_local_sampler_input(self, dataset, **kwargs) -> NodeSamplerInput:
        file_loader = FileLoader()
        with file_loader.load_to_temp_file(self._uri) as temp_file:
            tensor = torch.load(temp_file)
        return NodeSamplerInput(node=tensor, input_type=self._input_type)


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
        positive_labels: torch.Tensor,
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
            positive_labels=self.positive_labels[index],
            negative_labels=self.negative_labels[index]
            if self.negative_labels is not None
            else None,
            supervision_node_type=self.supervision_node_type,
        )
