import argparse

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf

from gigl.common import UriFactory
from gigl.common.utils.proto_utils import ProtoUtils
from snapchat.research.gbml import gbml_config_pb2, training_samples_schema_pb2

"""
Usage:
( cd python && \
python -m tests.test_assets.dataset_mocking.visualization_test.visualize_sgs_output \
--preprocessed_metadata_uri gs://TEMP DEV GBML PLACEHOLDER/toy_graph/config_populator/frozen_gbml_config.yaml \
)
"""


class SGSVisualizer:
    """
    Used to visualize user specified subsampled subgraphs
    """

    def __init__(self, frozen_config):
        self.__proto_utils = ProtoUtils()
        self._load_configs(frozen_config)

    def _load_configs(self, frozen_config):
        uri = UriFactory.create_uri(frozen_config)
        frozen_config_pb = self.__proto_utils.read_proto_from_yaml(
            uri=uri, proto_cls=gbml_config_pb2.GbmlConfig
        )
        node_type = frozen_config_pb.graph_metadata.node_types[0]

        self.random_negative_uri = frozen_config_pb.shared_config.flattened_graph_metadata.node_anchor_based_link_prediction_output.node_type_to_random_negative_tfrecord_uri_prefix[
            node_type
        ]
        self.node_anchor_based_uri = (
            frozen_config_pb.shared_config.flattened_graph_metadata.node_anchor_based_link_prediction_output.tfrecord_uri_prefix
        )

    def _plot_graph(self, pb, is_negative=True):
        output_graph = nx.DiGraph()
        nodes = {}

        for node in pb.neighborhood.nodes:
            node_id = node.node_id
            nodes[node_id] = {
                "condensed_node_type": node.condensed_node_type,
                "feature_values": node.feature_values,
            }
            output_graph.add_node(node_id)

        for edge in pb.neighborhood.edges:
            src_node_id = edge.src_node_id
            dst_node_id = edge.dst_node_id
            output_graph.add_edge(
                src_node_id,
                dst_node_id,
                condensed_edge_type=edge.condensed_edge_type,
                color="black",
            )

        node_border_colors = ["black" for node_id in output_graph.nodes()]
        node_colors = ["lightgrey" for node_id in output_graph.nodes()]

        if not is_negative:
            (
                root_node_random_neg_nodes,
                root_node_random_neg_edges,
            ) = self.get_rooted_neighbourhood_info(pb.root_node.node_id)

            for pos_edge in pb.pos_edges:
                src_node_id = pos_edge.src_node_id
                dst_node_id = pos_edge.dst_node_id

                (
                    dst_node_random_neg_nodes,
                    dst_node_random_neg_edges,
                ) = self.get_rooted_neighbourhood_info(dst_node_id)

                for edge in dst_node_random_neg_edges:
                    if edge not in root_node_random_neg_edges:
                        src_node_id, dst_node_id = edge
                        output_graph.add_edge(src_node_id, dst_node_id, color="black")

                node_colors = [
                    (
                        "lightgreen"
                        if node_id in dst_node_random_neg_nodes
                        else "lightgrey"
                    )
                    for node_id in output_graph.nodes()
                ]
                node_border_colors = [
                    "blue" if node_id in root_node_random_neg_nodes else "black"
                    for node_id in output_graph.nodes()
                ]
                edges = output_graph.edges()
                colors = [
                    (
                        "red"
                        if (u == src_node_id and v == dst_node_id)
                        else output_graph[u][v]["color"]
                    )
                    for u, v in edges
                ]

        else:
            edges = output_graph.edges()
            colors = [output_graph[u][v]["color"] for u, v in edges]

        plt.clf()
        nx.draw(
            output_graph,
            with_labels=True,
            node_color=node_colors,
            edge_color=colors,
            edgecolors=node_border_colors,
            linewidths=[3 if color == "red" else 2 for color in colors],
        )

        root_node_patch = plt.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="blue",
            markeredgewidth=2,
        )
        pos_edge_node_patch = plt.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markersize=10,
            markerfacecolor="green",
            markeredgecolor="none",
        )
        pos_edge_patch = mlines.Line2D([], [], color="red", linewidth=2, linestyle="-")
        plt.legend(
            [root_node_patch, pos_edge_node_patch, pos_edge_patch],
            ["Root Node Neighbourhood", "Pos Edge Node Neighbourhood", "Pos Edge"],
            loc="upper right",
        )

        plt.show()
        return plt

    def get_rooted_neighbourhood_info(self, node_id):
        uri = self.random_negative_uri + "*.tfrecord"
        ds = tf.data.TFRecordDataset(tf.io.gfile.glob(uri)).as_numpy_iterator()

        edges = set()
        nodes = set()
        while True:
            try:
                bytestr = next(iter(ds))
                pb = training_samples_schema_pb2.RootedNodeNeighborhood()
                pb.ParseFromString(bytestr)
                if pb.root_node.node_id == node_id:
                    for edge in pb.neighborhood.edges:
                        src_node_id = edge.src_node_id
                        dst_node_id = edge.dst_node_id
                        edges.add((src_node_id, dst_node_id))

                    for node in pb.neighborhood.nodes:
                        nodes.add(node.node_id)
                    return nodes, edges
            except StopIteration:
                break

        return None

    def visualize_random_negative_sample(self, root_node):
        uri = self.random_negative_uri + "*.tfrecord"
        ds = tf.data.TFRecordDataset(tf.io.gfile.glob(uri)).as_numpy_iterator()

        while True:
            try:
                bytestr = next(iter(ds))
                pb = training_samples_schema_pb2.RootedNodeNeighborhood()
                pb.ParseFromString(bytestr)
                if pb.root_node.node_id == root_node:
                    self._plot_graph(pb)
                    return "Done Visualizing Random Negative Sample"

            except StopIteration:
                break
        return None

    def visualize_node_anchor_prediction_sample(self, root_node):
        uri = self.node_anchor_based_uri + "*.tfrecord"
        ds = tf.data.TFRecordDataset(tf.io.gfile.glob(uri)).as_numpy_iterator()

        while True:
            try:
                bytestr = next(iter(ds))
                pb = training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample()
                pb.ParseFromString(bytestr)
                if pb.root_node.node_id == root_node:
                    self._plot_graph(pb, is_negative=False)
                    return "Done Visualizing Node Prediction Sample"
            except StopIteration:
                break

        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed_metadata_uri",
        required=True,
        type=str,
        help="GCS URI of the YAML file containing preprocessed metadata",
    )
    args = parser.parse_args()

    vis = SGSVisualizer(args.preprocessed_metadata_uri)
    vis.visualize_node_anchor_prediction_sample(5)
