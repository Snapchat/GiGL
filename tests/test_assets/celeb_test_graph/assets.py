import numpy as np
import torch
from google.protobuf.json_format import ParseDict

from gigl.src.common.graph_builder.pyg_graph_builder import PygGraphBuilder
from gigl.src.common.types.graph_data import Edge, Node, NodeId, NodeType, Relation
from gigl.src.common.utils.data.feature_serialization import FeatureSerializationUtils
from snapchat.research.gbml import (
    graph_schema_pb2,
    preprocessed_metadata_pb2,
    training_samples_schema_pb2,
)

__node_1_feature_value = np.array([1, 1, 1], dtype="float32").tolist()
__node_2_feature_value = np.array([2, 2, 2], dtype="float32").tolist()
__node_3_feature_value = np.array([3, 3, 3], dtype="float32").tolist()

__edge_1_2_1_feature_value = np.array([10], dtype="float32").tolist()
__edge_1_3_2_feature_value = np.array([20], dtype="float32").tolist()
__edge_2_3_2_feature_value = np.array([30], dtype="float32").tolist()


def get_celeb_graph_metadata_pb2():
    user_node_type = "user"
    celebrity_node_type = "celebrity"
    user_user_edge_type = graph_schema_pb2.EdgeType(
        relation="is_friend", src_node_type="user", dst_node_type="user"
    )
    user_celebrity_edge_type = graph_schema_pb2.EdgeType(
        relation="is_friend",
        src_node_type="user",
        dst_node_type="celebrity",
    )

    return graph_schema_pb2.GraphMetadata(
        node_types=[user_node_type, celebrity_node_type],
        edge_types=[user_user_edge_type, user_celebrity_edge_type],
        condensed_edge_type_map={
            1: user_user_edge_type,
            2: user_celebrity_edge_type,
        },
        condensed_node_type_map={1: user_node_type, 2: celebrity_node_type},
    )


def get_celeb_preprocessed_metadata():
    return {
        "condensed_node_type_to_preprocessed_metadata": {
            1: {"feature_dim": 3},
            2: {"feature_dim": 3},
        },
        "condensed_edge_type_to_preprocessed_metadata": {
            1: {"main_edge_info": {"feature_dim": 1}},
            2: {"main_edge_info": {"feature_dim": 1}},
        },
    }


def get_celeb_preprocessed_metadata_pb2():
    return ParseDict(
        js_dict=get_celeb_preprocessed_metadata(),
        message=preprocessed_metadata_pb2.PreprocessedMetadata(),
    )


def get_celeb_khop_subgraph_for_node1():
    # Node1 --(1)--> Node2
    # Node2 --(2)--> Node3
    # Node1 --(1)--> Node3
    node_1 = graph_schema_pb2.Node(
        node_id=1,
        condensed_node_type=1,
        feature_values=FeatureSerializationUtils.serialize_node_features(
            __node_1_feature_value
        ),
    )
    node_2 = graph_schema_pb2.Node(
        node_id=2,
        condensed_node_type=1,
        feature_values=FeatureSerializationUtils.serialize_node_features(
            __node_2_feature_value
        ),
    )
    node_3 = graph_schema_pb2.Node(
        node_id=3,
        condensed_node_type=2,
        feature_values=FeatureSerializationUtils.serialize_node_features(
            __node_3_feature_value
        ),
    )

    edge_1_2_1 = graph_schema_pb2.Edge(
        src_node_id=1,
        dst_node_id=2,
        condensed_edge_type=1,
        feature_values=FeatureSerializationUtils.serialize_edge_features(
            __edge_1_2_1_feature_value
        ),
    )
    edge_1_3_2 = graph_schema_pb2.Edge(
        src_node_id=1,
        dst_node_id=3,
        condensed_edge_type=2,
        feature_values=FeatureSerializationUtils.serialize_edge_features(
            __edge_1_3_2_feature_value
        ),
    )
    edge_2_3_2 = graph_schema_pb2.Edge(
        src_node_id=2,
        dst_node_id=3,
        condensed_edge_type=2,
        feature_values=FeatureSerializationUtils.serialize_edge_features(
            __edge_2_3_2_feature_value
        ),
    )

    khop_subgraph = graph_schema_pb2.Graph(
        nodes=[node_1, node_2, node_3],
        edges=[edge_1_2_1, edge_1_3_2, edge_2_3_2],
    )
    return node_1, khop_subgraph


def get_celeb_supervised_sample():
    root_node, subgraph = get_celeb_khop_subgraph_for_node1()
    return training_samples_schema_pb2.SupervisedNodeClassificationSample(
        root_node=root_node,  # node_1
        neighborhood=subgraph,
        root_node_labels=[
            training_samples_schema_pb2.Label(label_type="classification", label=1)
        ],
    )


def get_celeb_rooted_node_neighborhood_sample() -> (
    training_samples_schema_pb2.RootedNodeNeighborhood
):
    root_node, subgraph = get_celeb_khop_subgraph_for_node1()
    return training_samples_schema_pb2.RootedNodeNeighborhood(
        root_node=root_node, neighborhood=subgraph
    )


def get_celeb_expected_pyg_graph():
    pyg_graph_builder = PygGraphBuilder()
    expected_node_1 = Node(type=NodeType("user"), id=NodeId(1))
    expected_node_2 = Node(type=NodeType("user"), id=NodeId(2))
    expected_node_3 = Node(type=NodeType("celebrity"), id=NodeId(3))
    expected_edge_1_2_1 = Edge.from_nodes(
        src_node=expected_node_1,
        dst_node=expected_node_2,
        relation=Relation("is_friend"),
    )
    expected_edge_1_3_2 = Edge.from_nodes(
        src_node=expected_node_1,
        dst_node=expected_node_3,
        relation=Relation("is_friend"),
    )
    expected_edge_2_3_2 = Edge.from_nodes(
        src_node=expected_node_2,
        dst_node=expected_node_3,
        relation=Relation("is_friend"),
    )
    pyg_graph_builder.reset()
    pyg_graph_builder.add_node(
        node=expected_node_1, feature_values=torch.tensor(__node_1_feature_value)
    )
    pyg_graph_builder.add_node(
        node=expected_node_2, feature_values=torch.tensor(__node_2_feature_value)
    )
    pyg_graph_builder.add_node(
        node=expected_node_3, feature_values=torch.tensor(__node_3_feature_value)
    )
    pyg_graph_builder.add_edge(
        edge=expected_edge_1_2_1,
        feature_values=torch.tensor(__edge_1_2_1_feature_value),
    )
    pyg_graph_builder.add_edge(
        edge=expected_edge_1_3_2,
        feature_values=torch.tensor(__edge_1_3_2_feature_value),
    )
    pyg_graph_builder.add_edge(
        edge=expected_edge_2_3_2,
        feature_values=torch.tensor(__edge_2_3_2_feature_value),
    )

    return pyg_graph_builder.build()
