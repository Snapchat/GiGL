from typing import Optional, Union

from gigl.distributed.dist_dataset import DistDataset
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import FeatureInfo

_dataset: Optional[DistDataset] = None


def register_dataset(dataset: DistDataset) -> None:
    global _dataset
    _dataset = dataset


def get_node_feature_info() -> Union[FeatureInfo, dict[NodeType, FeatureInfo], None]:
    if _dataset is None:
        raise ValueError(
            "Dataset not registered! Register the dataset first with `gigl.distributed.server_client.register_dataset`"
        )
    return _dataset.node_feature_info


def get_edge_feature_info() -> Union[FeatureInfo, dict[EdgeType, FeatureInfo], None]:
    if _dataset is None:
        raise ValueError(
            "Dataset not registered! Register the dataset first with `gigl.distributed.server_client.register_dataset`"
        )
    return _dataset.edge_feature_info
