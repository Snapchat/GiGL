import argparse

import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf

from gigl.common import GcsUri, UriFactory
from gigl.common.utils.proto_utils import ProtoUtils
from snapchat.research.gbml import gbml_config_pb2, training_samples_schema_pb2

"""
( cd python && \
python -m tests.test_assets.dataset_mocking.visualization_test.visualize_sgn_output \
--frozen_config gs://TEMP DEV GBML PLACEHOLDER/toy_graph/config_populator/frozen_gbml_config.yaml \
)
"""


class SGNVisualizer:
    """
    Used to visualize train, test and val splits obtained from Split Generator (SGN)
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

        self.test_main_uri = UriFactory.create_uri(
            uri=frozen_config_pb.shared_config.dataset_metadata.node_anchor_based_link_prediction_dataset.test_main_data_uri
        )
        self.test_random_neg_uri = UriFactory.create_uri(
            uri=frozen_config_pb.shared_config.dataset_metadata.node_anchor_based_link_prediction_dataset.test_node_type_to_random_negative_data_uri[
                node_type
            ]
        )
        self.train_main_uri = UriFactory.create_uri(
            uri=frozen_config_pb.shared_config.dataset_metadata.node_anchor_based_link_prediction_dataset.train_main_data_uri
        )
        self.train_random_neg_uri = UriFactory.create_uri(
            uri=frozen_config_pb.shared_config.dataset_metadata.node_anchor_based_link_prediction_dataset.train_node_type_to_random_negative_data_uri[
                node_type
            ]
        )
        self.val_main_uri = UriFactory.create_uri(
            frozen_config_pb.shared_config.dataset_metadata.node_anchor_based_link_prediction_dataset.val_main_data_uri
        )
        self.val_random_neg_uri = UriFactory.create_uri(
            frozen_config_pb.shared_config.dataset_metadata.node_anchor_based_link_prediction_dataset.val_node_type_to_random_negative_data_uri[
                node_type
            ]
        )

        self.split_names = ["Train", "Test", "Val"]
        self.main_uris = [self.test_main_uri, self.train_main_uri, self.val_main_uri]
        self.random_neg_uris = [
            self.test_random_neg_uri,
            self.train_random_neg_uri,
            self.val_random_neg_uri,
        ]

    def _plot_graph(self, pb, root_node, index):
        g = nx.DiGraph()
        nodes = {}
        for node in pb.neighborhood.nodes:
            node_id = node.node_id
            nodes[node_id] = {
                "condensed_node_type": node.condensed_node_type,
                "feature_values": node.feature_values,
            }
            g.add_node(node_id)

        for edge in pb.neighborhood.edges:
            src_node_id = edge.src_node_id
            dst_node_id = edge.dst_node_id
            g.add_edge(
                src_node_id,
                dst_node_id,
                condensed_edge_type=edge.condensed_edge_type,
            )

        nx.draw(g, with_labels=True)
        plt.title(f"Root Node: {root_node}\n {self.split_names[index]} (Sample)")
        plt.show()
        # plt.savefig(f"{self.split_names[index]}.png")
        plt.clf()

    def visualize_main_data_output(self, root_node):
        for index, uri in enumerate(self.main_uris):
            curr_uri = GcsUri.join(uri, "*.tfrecord").uri
            ds = tf.data.TFRecordDataset(tf.io.gfile.glob(curr_uri)).as_numpy_iterator()

            while True:
                try:
                    bytestr = next(iter(ds))
                    pb = (
                        training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample()
                    )
                    pb.ParseFromString(bytestr)

                    if pb.root_node.node_id == root_node:
                        self._plot_graph(pb, root_node, index)
                except StopIteration:
                    break

    def visualize_random_negative_output(self, root_node):
        for index, uri in enumerate(self.random_neg_uris):
            curr_uri = GcsUri.join(uri, "*.tfrecord").uri
            ds = tf.data.TFRecordDataset(tf.io.gfile.glob(curr_uri)).as_numpy_iterator()

            while True:
                try:
                    bytestr = next(iter(ds))
                    pb = training_samples_schema_pb2.RootedNodeNeighborhood()
                    pb.ParseFromString(bytestr)

                    if pb.root_node.node_id == root_node:
                        self._plot_graph(pb, root_node, index)
                except StopIteration:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen_config",
        required=True,
        type=str,
        help="GCS URI of the YAML file containing preprocessed metadata",
    )
    args = parser.parse_args()
