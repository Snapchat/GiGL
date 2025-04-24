from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from gigl.common.env_config import get_available_cpus
from gigl.src.common.types.graph_data import (
    CondensedNodeType,
    EdgeType,
    EdgeUsageType,
    NodeId,
    NodeType,
)
from gigl.src.common.types.pb_wrappers.graph_data_types import GraphPbWrapper
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.utils.data.feature_serialization import FeatureSerializationUtils
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.user_defined_edge_sampling import sample_hydrate_user_def_edge
from snapchat.research.gbml import graph_schema_pb2, training_samples_schema_pb2

DEFAULT_NUM_HOPS_FOR_DATASETS = 1  # Number of hops to consider for each subgraph.
DEFAULT_NUM_NODES_PER_HOP = 5  # -1 means select all nodes at each hop.
DEFAULT_NUM_NEGATIVE_SAMPLES_PER_POS_EDGE = 1  # for samples taken from main edges

DEFAULT_PYG_NODE_ANCHOR_SPLIT_TRANSFORM = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=0,
)


def build_pyg_heterodata_from_mocked_dataset_info(
    mocked_dataset_info: MockedDatasetInfo,
) -> HeteroData:
    """
    Given a MockedDatasetInfo object, build a HeteroData object to use PyG convenience functions.
    """

    hetero_data = HeteroData()
    for node_type, node_feats in mocked_dataset_info.node_feats.items():
        hetero_data[node_type].x = node_feats
        hetero_data[node_type].n_id = torch.arange(hetero_data[node_type].num_nodes)

    if mocked_dataset_info.node_labels is not None:
        for node_type, node_labels in mocked_dataset_info.node_labels.items():
            hetero_data[node_type].y = node_labels

    for edge_type, edge_index in mocked_dataset_info.edge_index.items():
        hetero_data[
            edge_type.src_node_type, edge_type.relation, edge_type.dst_node_type
        ].edge_index = edge_index

    if mocked_dataset_info.edge_feats is not None:
        for edge_type, edge_attr in mocked_dataset_info.edge_feats.items():
            hetero_data[
                (edge_type.src_node_type, edge_type.relation, edge_type.dst_node_type)
            ].x = edge_attr

    return hetero_data


def _build_graph_pb_wrapper_from_hetero_data(
    hetero_data: HeteroData, graph_metadata_pb_wrapper: GraphMetadataPbWrapper
) -> GraphPbWrapper:
    khop_subgraph_edges: List[graph_schema_pb2.Edge] = list()
    khop_subgraph_nodes: List[graph_schema_pb2.Node] = list()

    for pyg_edge_type in hetero_data.edge_types:
        edge_type_metadata = hetero_data[pyg_edge_type]
        edge_index = edge_type_metadata.get("edge_index")
        edge_attr = edge_type_metadata.get("x")

        src_pyg_node_type = pyg_edge_type[0]
        dst_pyg_node_type = pyg_edge_type[2]
        edge_type = EdgeType(
            src_node_type=src_pyg_node_type,
            relation=pyg_edge_type[1],
            dst_node_type=dst_pyg_node_type,
        )
        condensed_edge_type = (
            graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[edge_type]
        )

        src_node_ids: torch.Tensor
        dst_node_ids: torch.Tensor
        src_node_ids, dst_node_ids = edge_index

        global_src_node_ids = torch.take(
            hetero_data[src_pyg_node_type].get("n_id"), src_node_ids
        )
        global_dst_node_ids = torch.take(
            hetero_data[dst_pyg_node_type].get("n_id"), dst_node_ids
        )

        for idx, (global_src_node_id, global_dst_node_id) in enumerate(
            zip(global_src_node_ids, global_dst_node_ids)
        ):
            edge_feature_value = (
                FeatureSerializationUtils.serialize_edge_features(
                    features=edge_attr[idx, :].numpy()
                )
                if edge_attr is not None
                else None
            )
            edge = graph_schema_pb2.Edge(
                src_node_id=global_src_node_id,
                dst_node_id=global_dst_node_id,
                condensed_edge_type=condensed_edge_type,
                feature_values=edge_feature_value,  # type: ignore
            )
            khop_subgraph_edges.append(edge)

    for pyg_node_type in hetero_data.node_types:
        node_type_metadata = hetero_data[pyg_node_type]
        node_attr = node_type_metadata.get("x")
        assert node_attr is not None

        node_type = NodeType(pyg_node_type)
        condensed_node_type = (
            graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[node_type]
        )

        global_node_ids = node_type_metadata.get("n_id")
        assert global_node_ids is not None

        for idx, global_node_id in enumerate(global_node_ids):
            node_feature_value = FeatureSerializationUtils.serialize_node_features(
                node_attr[idx, :].numpy()
            )

            node = graph_schema_pb2.Node(
                node_id=global_node_id,
                condensed_node_type=condensed_node_type,
                feature_values=node_feature_value,  # type: ignore
            )
            khop_subgraph_nodes.append(node)

    subgraph = GraphPbWrapper(
        pb=graph_schema_pb2.Graph(
            nodes=khop_subgraph_nodes,
            edges=khop_subgraph_edges,
        )
    )
    return subgraph


def build_k_hop_subgraphs_from_pyg_heterodata(
    hetero_data: HeteroData,
    graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
    root_node_type: NodeType,
    root_node_idxs: Optional[torch.Tensor] = None,
    num_hops: int = DEFAULT_NUM_HOPS_FOR_DATASETS,
    num_neighbors: int = DEFAULT_NUM_NODES_PER_HOP,
) -> Dict[NodeId, GraphPbWrapper]:
    """
    Given inputs, return a map of each root node of type `root_node_type` and index in `root_node_idxs'
    to GraphPbWrappers which describe the `num_hops` surrounding subgraph.
    """

    if root_node_idxs is None:
        root_node_idxs = torch.arange(hetero_data[str(root_node_type)].num_nodes)

    num_neighbors_dict = {
        edge_type: [num_neighbors] * num_hops for edge_type in hetero_data.edge_types
    }

    loader = NeighborLoader(
        data=hetero_data,
        num_neighbors=num_neighbors_dict,
        input_nodes=(str(root_node_type), root_node_idxs),
        batch_size=1,
        num_workers=get_available_cpus()
        - 1,  # use all available CPUs except one, for this task.
    )

    k_hop_subgraphs: Dict[NodeId, GraphPbWrapper] = dict()

    sample: HeteroData
    for root_node_idx, sample in zip(root_node_idxs.tolist(), loader):
        graph_pb_wrapper = _build_graph_pb_wrapper_from_hetero_data(
            hetero_data=sample, graph_metadata_pb_wrapper=graph_metadata_pb_wrapper
        )
        k_hop_subgraphs[NodeId(root_node_idx)] = graph_pb_wrapper

    return k_hop_subgraphs


def _get_random_negative_samples_for_pos_edges(
    edge_index: torch.LongTensor,
    num_nodes: int,
    num_negative_samples_per_pos_edge: int = 1,
) -> torch.LongTensor:
    """
    Given an "positive" edge index (edges which exist), we return a "negative" edge
    index (edges which likely don't) of an equal size.  We effectively sample the
    endpoints of these negative edges randomly from the node-set.
    """

    pos_node_ids = edge_index[0].repeat(num_negative_samples_per_pos_edge)
    neg_node_ids = torch.randint(low=0, high=num_nodes, size=[pos_node_ids.numel()])
    return torch.vstack((pos_node_ids, neg_node_ids))  # type: ignore


def _build_rooted_node_neighborhood_samples_from_subgraphs(
    subgraph_dict: Dict[NodeId, GraphPbWrapper], condensed_node_type: CondensedNodeType
) -> List[training_samples_schema_pb2.RootedNodeNeighborhood]:
    samples: List[training_samples_schema_pb2.RootedNodeNeighborhood] = list()

    for root_node_id, subgraph in subgraph_dict.items():
        sample = training_samples_schema_pb2.RootedNodeNeighborhood(
            root_node=graph_schema_pb2.Node(
                node_id=int(root_node_id),
                condensed_node_type=condensed_node_type,
                feature_values=None,  # type: ignore
            ),
            neighborhood=subgraph.pb,
        )
        samples.append(sample)

    return samples


def build_supervised_node_classification_samples_from_pyg_heterodata(
    hetero_data: HeteroData,
    root_node_type: NodeType,
    graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
) -> List[training_samples_schema_pb2.SupervisedNodeClassificationSample]:
    samples: List[
        training_samples_schema_pb2.SupervisedNodeClassificationSample
    ] = list()

    assert (
        hetero_data[str(root_node_type)].get("y") is not None
    )  # ensure labels exist for this node type (else we cannot have a supervised task)
    node_labels = hetero_data[str(root_node_type)].y

    k_hop_subgraphs_for_root_node_type = build_k_hop_subgraphs_from_pyg_heterodata(
        hetero_data=hetero_data,
        graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
        root_node_type=root_node_type,
        num_hops=DEFAULT_NUM_HOPS_FOR_DATASETS,
    )

    condensed_node_type = (
        graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[root_node_type]
    )

    for root_node_id, subgraph in k_hop_subgraphs_for_root_node_type.items():
        sample = training_samples_schema_pb2.SupervisedNodeClassificationSample(
            root_node=graph_schema_pb2.Node(
                node_id=int(root_node_id),
                condensed_node_type=condensed_node_type,
                feature_values=None,  # type: ignore
            ),
            neighborhood=subgraph.pb,
            root_node_labels=[
                training_samples_schema_pb2.Label(
                    label_type="classification",
                    label=node_labels[int(root_node_id)],
                )
            ],
        )
        samples.append(sample)

    return samples


def build_node_anchor_link_prediction_samples_from_pyg_heterodata(
    hetero_data: HeteroData,
    sample_edge_type: EdgeType,
    graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
    mocked_dataset_info: MockedDatasetInfo,
) -> Tuple[
    List[training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample],
    List[training_samples_schema_pb2.RootedNodeNeighborhood],
    List[training_samples_schema_pb2.RootedNodeNeighborhood],
]:
    src_node_id_to_k_hop_subgraph = build_k_hop_subgraphs_from_pyg_heterodata(
        hetero_data=hetero_data,
        graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
        root_node_type=sample_edge_type.src_node_type,
        num_hops=DEFAULT_NUM_HOPS_FOR_DATASETS,
    )

    if sample_edge_type.src_node_type == sample_edge_type.dst_node_type:
        # If the source and destination node types are the same, we can reuse the same subgraphs.
        dst_node_id_to_k_hop_subgraph = src_node_id_to_k_hop_subgraph
    else:
        # Otherwise, we need to build a separate set of subgraphs for the destination node type.
        dst_node_id_to_k_hop_subgraph = build_k_hop_subgraphs_from_pyg_heterodata(
            hetero_data=hetero_data,
            graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            root_node_type=sample_edge_type.dst_node_type,
            num_hops=DEFAULT_NUM_HOPS_FOR_DATASETS,
        )

    condensed_src_node_type = (
        graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
            sample_edge_type.src_node_type
        ]
    )
    condensed_dst_node_type = (
        graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
            sample_edge_type.dst_node_type
        ]
    )
    condensed_sample_edge_type = (
        graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[sample_edge_type]
    )

    # Create RootedNodeNeighborhood samples
    rooted_neighborhoods_for_src_node_type = (
        _build_rooted_node_neighborhood_samples_from_subgraphs(
            subgraph_dict=src_node_id_to_k_hop_subgraph,
            condensed_node_type=condensed_src_node_type,
        )
    )
    rooted_neighborhoods_for_dst_node_type = (
        _build_rooted_node_neighborhood_samples_from_subgraphs(
            subgraph_dict=dst_node_id_to_k_hop_subgraph,
            condensed_node_type=condensed_dst_node_type,
        )
    )

    user_defined_pos_edges = (
        mocked_dataset_info.user_defined_edge_index[sample_edge_type][
            EdgeUsageType.POSITIVE
        ]
        if mocked_dataset_info.user_defined_edge_index
        else None
    )
    user_def_pos_edge_feats = (
        mocked_dataset_info.user_defined_edge_feats[sample_edge_type][
            EdgeUsageType.POSITIVE
        ]
        if mocked_dataset_info.user_defined_edge_feats
        else None
    )

    if user_defined_pos_edges is not None:
        pos_node_map = sample_hydrate_user_def_edge(
            mocked_dataset_info=mocked_dataset_info,
            edge_usage_type=EdgeUsageType.POSITIVE,
        )
    else:
        pos_node_map = defaultdict(list)
        # Create map to track each node's candidate neighbors.
        edge_label_index = hetero_data[
            (
                str(sample_edge_type.src_node_type),
                str(sample_edge_type.relation),
                str(sample_edge_type.dst_node_type),
            )
        ].edge_label_index
        for src, dst in zip(edge_label_index[0].tolist(), edge_label_index[1].tolist()):
            pos_node_map[src].append(dst)

    user_defined_neg_edges = (
        mocked_dataset_info.user_defined_edge_index[sample_edge_type][
            EdgeUsageType.NEGATIVE
        ]
        if mocked_dataset_info.user_defined_edge_index
        else None
    )
    user_def_neg_edge_feats = (
        mocked_dataset_info.user_defined_edge_feats[sample_edge_type][
            EdgeUsageType.NEGATIVE
        ]
        if mocked_dataset_info.user_defined_edge_feats
        else None
    )

    if user_defined_neg_edges is not None:
        hard_neg_node_map = sample_hydrate_user_def_edge(
            mocked_dataset_info=mocked_dataset_info,
            edge_usage_type=EdgeUsageType.NEGATIVE,
        )
    else:
        hard_neg_node_map = defaultdict(list)
        # Create map to track each node's negatives
        hard_neg_edge_index = _get_random_negative_samples_for_pos_edges(
            edge_index=edge_label_index,
            num_nodes=hetero_data[str(sample_edge_type.dst_node_type)].num_nodes,
            num_negative_samples_per_pos_edge=DEFAULT_NUM_NEGATIVE_SAMPLES_PER_POS_EDGE,
        )
        for src, dst in zip(
            hard_neg_edge_index[0].tolist(), hard_neg_edge_index[1].tolist()
        ):
            hard_neg_node_map[src].append(dst)

    unsup_node_anchor_samples: List[
        training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample
    ] = list()

    # Create UnsupNodeAnchor samples for each node with at least 1 positive edge.
    unique_nodes = list(pos_node_map.keys())
    for root_node_id in unique_nodes:
        pos_edge_pbs: List[graph_schema_pb2.Edge] = list()
        hard_neg_edge_pbs: List[graph_schema_pb2.Edge] = list()
        subgraphs_to_merge: List[GraphPbWrapper] = list()

        root_node_pb = graph_schema_pb2.Node(
            node_id=root_node_id,
            condensed_node_type=condensed_src_node_type,
            feature_values=None,  # type: ignore
        )
        subgraphs_to_merge.append(src_node_id_to_k_hop_subgraph[root_node_id])

        for pos_sample in pos_node_map[root_node_id]:
            if (
                user_def_pos_edge_feats is not None
            ):  # pos_node_map={root_node_id: [pos_node_id, edge_feats]}
                pos_node_id = pos_sample[0]
                pos_edge_feats = pos_sample[1]
                edge_pb = graph_schema_pb2.Edge(
                    src_node_id=root_node_id,
                    dst_node_id=pos_node_id,
                    condensed_edge_type=condensed_sample_edge_type,
                    feature_values=pos_edge_feats,
                )
            else:
                pos_node_id = pos_sample
                edge_pb = graph_schema_pb2.Edge(
                    src_node_id=root_node_id,
                    dst_node_id=pos_node_id,
                    condensed_edge_type=condensed_sample_edge_type,
                )

            pos_edge_pbs.append(edge_pb)

            subgraphs_to_merge.append(dst_node_id_to_k_hop_subgraph[pos_node_id])

        for hard_neg_sample in hard_neg_node_map[root_node_id]:
            if (
                user_def_neg_edge_feats is not None
            ):  # neg_node_map={root_node_id: [hard_neg_node_id, edge_feats]}:
                hard_neg_node_id = hard_neg_sample[0]
                hard_neg_edge_feats = hard_neg_sample[1]
                edge_pb = graph_schema_pb2.Edge(
                    src_node_id=root_node_id,
                    dst_node_id=hard_neg_node_id,
                    condensed_edge_type=condensed_sample_edge_type,
                    feature_values=hard_neg_edge_feats,
                )
            else:
                hard_neg_node_id = hard_neg_sample
                edge_pb = graph_schema_pb2.Edge(
                    src_node_id=root_node_id,
                    dst_node_id=hard_neg_node_id,
                    condensed_edge_type=condensed_sample_edge_type,
                )
            hard_neg_edge_pbs.append(edge_pb)

            subgraphs_to_merge.append(dst_node_id_to_k_hop_subgraph[hard_neg_node_id])

        neighborhood_pb = GraphPbWrapper.merge_subgraphs(
            subgraphs=subgraphs_to_merge
        ).pb

        sample = training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample(
            root_node=root_node_pb,
            pos_edges=pos_edge_pbs,
            hard_neg_edges=hard_neg_edge_pbs,
            neighborhood=neighborhood_pb,
        )
        unsup_node_anchor_samples.append(sample)

    return (
        unsup_node_anchor_samples,
        rooted_neighborhoods_for_src_node_type,
        rooted_neighborhoods_for_dst_node_type,
    )
