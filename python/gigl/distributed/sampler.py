import ast
import json
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from graphlearn_torch.sampler import NodeSamplerInput, RemoteSamplerInput

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.common.utils.file_loader import FileLoader
from gigl.types.graph import FeatureInfo

logger = Logger()


@dataclass
class RemoteNodeInfo:
    node_type: Optional[NodeType]
    edge_types: Optional[list[tuple[NodeType, NodeType, NodeType]]]
    node_tensor_uris: list[str]
    node_feature_info: Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]]
    edge_feature_info: Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]]
    num_partitions: int
    edge_dir: str
    master_addr: str
    master_port: int
    num_servers: int

    def serialize(self) -> str:
        """Serialize the RemoteNodeInfo to a JSON string."""
        out_dict = {}

        # Handle node_type (str or None)
        out_dict["node_type"] = self.node_type

        # Handle edge_types (list of EdgeType tuples -> list of lists)
        if self.edge_types is not None:
            out_dict["edge_types"] = [list(edge_type) for edge_type in self.edge_types]
        else:
            out_dict["edge_types"] = None

        # Handle simple fields
        out_dict["node_tensor_uris"] = self.node_tensor_uris
        out_dict["num_partitions"] = self.num_partitions
        out_dict["edge_dir"] = self.edge_dir
        out_dict["master_addr"] = self.master_addr
        out_dict["master_port"] = self.master_port
        out_dict["num_servers"] = self.num_servers

        def serialize_feature_info(feature_info: FeatureInfo) -> dict:
            """Serialize FeatureInfo with proper torch.dtype handling."""
            return {"dim": feature_info.dim, "dtype": str(feature_info.dtype)}

        # Handle node_feature_info (FeatureInfo, dict, or None)
        if self.node_feature_info is None:
            out_dict["node_feature_info"] = None
        elif isinstance(self.node_feature_info, dict):
            out_dict["node_feature_info"] = {
                k: serialize_feature_info(v) for k, v in self.node_feature_info.items()
            }
        else:
            out_dict["node_feature_info"] = serialize_feature_info(
                self.node_feature_info
            )

        # Handle edge_feature_info (FeatureInfo, dict, or None)
        if self.edge_feature_info is None:
            out_dict["edge_feature_info"] = None
        elif isinstance(self.edge_feature_info, dict):
            out_dict["edge_feature_info"] = {
                str(list(k)): serialize_feature_info(v)
                for k, v in self.edge_feature_info.items()
            }
        else:
            out_dict["edge_feature_info"] = serialize_feature_info(
                self.edge_feature_info
            )

        return json.dumps(out_dict, indent=2)

    def dump(self) -> str:
        """Legacy method name for backward compatibility."""
        return self.serialize()

    def save(self, uri: Uri) -> None:
        """Save RemoteNodeInfo to a URI."""
        json_str = self.serialize()
        file_loader = FileLoader()
        with file_loader.save_to_temp_file(json_str.encode(), uri) as temp_file:
            pass  # File is saved when context manager exits

    @classmethod
    def deserialize(cls, json_str: str) -> "RemoteNodeInfo":
        """Deserialize a JSON string to a RemoteNodeInfo instance."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")

        # Validate required fields
        required_fields = [
            "edge_types",
            "node_tensor_uris",
            "num_partitions",
            "edge_dir",
            "master_addr",
            "master_port",
            "num_servers",
        ]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        def deserialize_feature_info(feature_info_data: dict) -> FeatureInfo:
            """Deserialize FeatureInfo with proper torch.dtype handling."""
            if (
                not isinstance(feature_info_data, dict)
                or "dim" not in feature_info_data
                or "dtype" not in feature_info_data
            ):
                raise ValueError(f"Invalid FeatureInfo data: {feature_info_data}")

            dtype_str = feature_info_data["dtype"]
            # Convert string representation back to torch.dtype
            try:
                dtype = getattr(torch, dtype_str.split(".")[-1])
            except AttributeError:
                raise ValueError(f"Invalid torch dtype: {dtype_str}")

            return FeatureInfo(dim=feature_info_data["dim"], dtype=dtype)

        # Handle edge_types conversion from list of lists back to list of EdgeType tuples
        edge_types = []
        if data["edge_types"] is not None:
            for edge_type_list in data["edge_types"]:
                edge_type = EdgeType(
                    src_node_type=NodeType(edge_type_list[0]),
                    relation=Relation(edge_type_list[1]),
                    dst_node_type=NodeType(edge_type_list[2]),
                )
                edge_types.append(edge_type)
        else:
            edge_types = None

        # Handle node_feature_info deserialization
        node_feature_info = None
        if data["node_feature_info"] is not None:
            if (
                isinstance(data["node_feature_info"], dict)
                and "dim" in data["node_feature_info"]
            ):
                # Single FeatureInfo
                node_feature_info = deserialize_feature_info(data["node_feature_info"])
            else:
                # Dict of NodeType -> FeatureInfo
                node_feature_info = {
                    NodeType(k): deserialize_feature_info(v)
                    for k, v in data["node_feature_info"].items()
                }

        # Handle edge_feature_info deserialization
        edge_feature_info = None
        if data["edge_feature_info"] is not None:
            if (
                isinstance(data["edge_feature_info"], dict)
                and "dim" in data["edge_feature_info"]
            ):
                # Single FeatureInfo
                edge_feature_info = deserialize_feature_info(data["edge_feature_info"])
            else:
                # Dict of EdgeType -> FeatureInfo
                edge_feature_info = {}
                for k, v in data["edge_feature_info"].items():
                    # Parse the string representation back to list then to EdgeType
                    edge_type_list = ast.literal_eval(k)
                    if isinstance(edge_type_list, list) and len(edge_type_list) == 3:
                        edge_type = EdgeType(
                            src_node_type=NodeType(edge_type_list[0]),
                            relation=Relation(edge_type_list[1]),
                            dst_node_type=NodeType(edge_type_list[2]),
                        )
                        edge_feature_info[edge_type] = deserialize_feature_info(v)

        return cls(
            node_type=NodeType(data["node_type"])
            if data["node_type"] is not None
            else None,
            edge_types=edge_types,
            node_tensor_uris=data["node_tensor_uris"],
            node_feature_info=node_feature_info,
            edge_feature_info=edge_feature_info,
            num_partitions=data["num_partitions"],
            edge_dir=data["edge_dir"],
            master_addr=data["master_addr"],
            master_port=data["master_port"],
            num_servers=data["num_servers"],
        )

    @classmethod
    def load(cls, uri: Uri) -> "RemoteNodeInfo":
        """Load RemoteNodeInfo from a URI."""
        logger.info(f"{uri=}")
        tf = FileLoader().load_to_temp_file(
            uri, should_create_symlinks_if_possible=False
        )
        with open(tf.name, "r") as f:
            json_str = f.read()
            logger.info(f"Loaded JSON: {json_str}")
        tf.close()
        return cls.deserialize(json_str)


class RemoteUriSamplerInput(RemoteSamplerInput):
    def __init__(self, uri: Uri, input_type: Optional[Union[str, NodeType]]):
        self._uri = uri
        self._input_type = input_type

    @property
    def input_type(self) -> Optional[Union[str, NodeType]]:
        return self._input_type

    def to_local_sampler_input(self, dataset, **kwargs) -> NodeSamplerInput:
        file_loader = FileLoader()
        with file_loader.load_to_temp_file(self._uri) as temp_file:
            tensor = torch.load(temp_file.name)
        print(f"Loaded tensor: {tensor.shape}")
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
