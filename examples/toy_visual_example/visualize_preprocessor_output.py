import argparse
import os

import pandas as pd
import tensorflow as tf
import yaml

from gigl.common import GcsUri, LocalUri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.src.training.v1.lib.data_loaders.tf_records_iterable_dataset import (
    get_np_iterator_from_tfrecords,
)

"""
Usage:
( cd python && \
python -m tests.test_assets.dataset_mocking.visualization_test.visualize_preprocessor_output \
--metadata_uri path_to_preprocessed_metadata.yaml \
)
"""


def visualize_preprocessed_graph(metadata_uri):
    """
    Used to visualize data preprocesser output

    Input: metadata_uri: str -> gcs uri for where preprocessed_metadata.yaml is located
    """
    logger = Logger()

    # Parse the metadata from the YAML file stored on GCS
    gcs_utils = GcsUtils()
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    gcs_utils.download_file_from_gcs(
        GcsUri(metadata_uri), LocalUri(path + "/temp_metadata_dict.yaml")
    )
    with open(path + "/temp_metadata_dict.yaml", "r") as f:
        metadata_dict = yaml.load(f, Loader=yaml.Loader)

    # Extract the condensed node and edge type maps
    condensed_node_type_map = metadata_dict["graphMetadata"]["condensedNodeTypeMap"]
    condensed_edge_type_map = metadata_dict["graphMetadata"]["condensedEdgeTypeMap"]

    node_data = []

    for condensed_node_type, node_metadata in metadata_dict[
        "condensedNodeTypeToPreprocessedMetadata"
    ].items():
        logger.info("Displaying Node Metadata:\n")

        feature_keys = node_metadata["featureKeys"]
        node_id_key = node_metadata["nodeIdKey"]
        tfrecord_files = tf.io.gfile.glob(node_metadata["tfrecordUriPrefix"] + "*")
        dataset_sampler = tf.data.TFRecordDataset(tfrecord_files)

        for tfrecord_file in tfrecord_files:
            logger.info(f"Reading TFRecord file: {tfrecord_file}\n")
            for record in dataset_sampler.take(1):
                logger.info(
                    f"One Sample Straight From TFRecord (Unformatted):\n {tf.train.Example.FromString(record.numpy())}",
                )

        node_dataset = get_np_iterator_from_tfrecords(
            UriFactory.create_uri(node_metadata["schemaUri"]), tfrecord_files
        )

        for serialized_example in node_dataset:
            row = {node_id_key: serialized_example[node_id_key][0]}
            row.update({"Node Type": condensed_node_type_map[condensed_node_type]})
            row.update(
                {k: v[0] for k, v in serialized_example.items() if k in feature_keys}
            )
            node_data.append(row)

        node_df = pd.DataFrame(node_data)
        node_df.set_index(node_id_key, inplace=True)
        node_df = node_df.reindex(sorted(node_df.columns), axis=1)

        print(node_df)

    edge_data = []

    # Processing edge metadata for each condensed edge type
    for condensed_edge_type, edge_metadata in metadata_dict[
        "condensedEdgeTypeToPreprocessedMetadata"
    ].items():
        logger.info("Displaying Edge Metadata: \n")

        tfrecord_files = tf.io.gfile.glob(edge_metadata["tfrecordUriPrefix"] + "*")
        dataset_sampler = tf.data.TFRecordDataset(tfrecord_files)

        for tfrecord_file in tfrecord_files:
            logger.info(f"Reading TFRecord file: {tfrecord_file}\n")
            for record in dataset_sampler.take(1):
                logger.info(
                    f"One Sample Straight From TFRecord (Unformatted):\n {tf.train.Example.FromString(record.numpy())}",
                )

        edge_dataset = get_np_iterator_from_tfrecords(
            UriFactory.create_uri(edge_metadata["schemaUri"]), tfrecord_files
        )

        for serialized_example in edge_dataset:
            dst_node_type = condensed_edge_type_map[condensed_edge_type]["dstNodeType"]
            dst_node_value = serialized_example["dst"]
            src_node_type = condensed_edge_type_map[condensed_edge_type]["srcNodeType"]
            src_node_value = serialized_example["src"]
            relation = condensed_edge_type_map[condensed_edge_type]["relation"]

            row_data = [
                condensed_edge_type,
                dst_node_type,
                dst_node_value,
                src_node_type,
                src_node_value,
                relation,
            ]

            edge_data.append(row_data)

        edge_df = pd.DataFrame(
            edge_data,
            columns=[
                "Condensed Edge Type",
                "dst node type",
                "dst node value",
                "src node type",
                "src node value",
                "relation",
            ],
        )
        edge_df.style.set_caption("Formatted Output of Edge TFRecord")

        os.remove(path + "/temp_metadata_dict.yaml")

        return node_df, edge_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_uri",
        required=True,
        type=str,
        help="GCS URI of the YAML file containing preprocessed metadata",
    )
    args = parser.parse_args()

    visualize_preprocessed_graph(args.metadata_uri)
