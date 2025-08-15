from collections import defaultdict
from typing import Dict, List

from gigl.src.common.types.graph_data import EdgeUsageType, NodeId
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo


def sample_hydrate_user_def_edge(
    mocked_dataset_info: MockedDatasetInfo, edge_usage_type: EdgeUsageType
) -> Dict[NodeId, List]:
    """
    Samples all available pos/neg edges and hydrated these edges with their features.
    e.g. for positive edge the output will be {pos_edge_src: [pos_edge_dst, [f0, f1, ..., fn]]}
    """
    src_to_dst_map = defaultdict(list)

    assert (
        mocked_dataset_info.sample_edge_type is not None
    ), "sample_edge_type is missing in mocked_dataset_info"
    edge_index = (
        mocked_dataset_info.user_defined_edge_index[
            mocked_dataset_info.sample_edge_type  # type: ignore
        ][edge_usage_type]
        if mocked_dataset_info.user_defined_edge_index
        else None
    )
    edge_feats = (
        mocked_dataset_info.user_defined_edge_feats[
            mocked_dataset_info.sample_edge_type  # type: ignore
        ][edge_usage_type]
        if mocked_dataset_info.user_defined_edge_feats
        else None
    )

    if edge_feats is not None:
        for src, dst, feats in zip(
            edge_index[0].tolist(),  # type: ignore
            edge_index[1].tolist(),  # type: ignore
            edge_feats.tolist(),
        ):
            src_to_dst_map[src].append([dst, feats])
    else:
        for src, dst in zip(
            edge_index[0].tolist(),  # type: ignore
            edge_index[1].tolist(),  # type: ignore
        ):
            src_to_dst_map[src].append(dst)

    return src_to_dst_map
