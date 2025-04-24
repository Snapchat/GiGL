from typing import List

import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.mocking.lib import pyg_to_training_samples, tfrecord_io
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from snapchat.research.gbml import gbml_config_pb2, training_samples_schema_pb2

logger = Logger()


def split_and_write_supervised_node_classification_subgraph_samples_from_mocked_dataset_info(
    mocked_dataset_info: MockedDatasetInfo,
    root_node_type: NodeType,
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
    hetero_data: HeteroData,
):
    transductive_split_cls = T.RandomNodeSplit(
        split="train_rest",
        num_val=0.3,
        num_test=0.3,
    )

    # Return result HeteroData with train_mask, val_mask, test_mask defined
    split_data: HeteroData = transductive_split_cls(hetero_data)

    # Build all SNC samples from dataset.
    samples: List[
        training_samples_schema_pb2.SupervisedNodeClassificationSample
    ] = pyg_to_training_samples.build_supervised_node_classification_samples_from_pyg_heterodata(
        hetero_data=split_data,
        root_node_type=root_node_type,
        graph_metadata_pb_wrapper=mocked_dataset_info.graph_metadata_pb_wrapper,
    )

    # Separate into train / val / test sets according to mask.
    root_node_data_view = split_data[str(root_node_type)]
    train_idxs = set(torch.where(root_node_data_view.train_mask)[0].tolist())
    val_idxs = set(torch.where(root_node_data_view.val_mask)[0].tolist())
    test_idxs = set(torch.where(root_node_data_view.test_mask)[0].tolist())

    train_samples: List[
        training_samples_schema_pb2.SupervisedNodeClassificationSample
    ] = list()
    val_samples: List[
        training_samples_schema_pb2.SupervisedNodeClassificationSample
    ] = list()
    test_samples: List[
        training_samples_schema_pb2.SupervisedNodeClassificationSample
    ] = list()

    for sample in samples:
        node_id = sample.root_node.node_id
        if node_id in train_idxs:
            train_samples.append(sample)
        elif node_id in val_idxs:
            val_samples.append(sample)
        elif node_id in test_idxs:
            test_samples.append(sample)
        else:
            raise ValueError(
                f"Found node id {node_id} which is unassigned to train / val / test."
            )

    # Write out to GbmlConfig-specified paths
    output_paths = (
        gbml_config_pb.shared_config.dataset_metadata.supervised_node_classification_dataset
    )

    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=train_samples,
        uri_prefix=UriFactory.create_uri(uri=output_paths.train_data_uri),
        sample_type_for_logging="train SNC",
    )
    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=val_samples,
        uri_prefix=UriFactory.create_uri(uri=output_paths.val_data_uri),
        sample_type_for_logging="val SNC",
    )
    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=test_samples,
        uri_prefix=UriFactory.create_uri(uri=output_paths.test_data_uri),
        sample_type_for_logging="test SNC",
    )


def split_and_write_node_anchor_link_prediction_subgraph_samples_from_mocked_dataset_info(
    mocked_dataset_info: MockedDatasetInfo,
    sample_edge_type: EdgeType,
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
    hetero_data: HeteroData,
):
    transductive_split_cls = T.RandomLinkSplit(
        num_val=0.3,
        num_test=0.3,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=0,
        edge_types=[(sample_edge_type[0], sample_edge_type[1], sample_edge_type[2])],
    )

    train_data, val_data, test_data = transductive_split_cls(hetero_data)

    # NOTE (Tong): the code above always split on message passing edges even when UDL edges exist.
    #       Hence, when mocking UDL datasets, the following code will result with the same label edges
    #       for train/val/test splits. This is fine right now as they are for functionality testing
    #       purposes, but it needs to be revisited if we want to enforce the correctness of the
    #       mocked datasets in future.

    # Build samples for train split.
    (
        train_na_samples,
        train_rnn_src_samples,
        train_rnn_dst_samples,
    ) = pyg_to_training_samples.build_node_anchor_link_prediction_samples_from_pyg_heterodata(
        hetero_data=train_data,
        sample_edge_type=sample_edge_type,
        graph_metadata_pb_wrapper=mocked_dataset_info.graph_metadata_pb_wrapper,
        mocked_dataset_info=mocked_dataset_info,
    )

    # Build samples for val split.
    (
        val_na_samples,
        val_rnn_src_samples,
        val_rnn_dst_samples,
    ) = pyg_to_training_samples.build_node_anchor_link_prediction_samples_from_pyg_heterodata(
        hetero_data=val_data,
        sample_edge_type=sample_edge_type,
        graph_metadata_pb_wrapper=mocked_dataset_info.graph_metadata_pb_wrapper,
        mocked_dataset_info=mocked_dataset_info,
    )

    #  Build samples for test split.
    (
        test_na_samples,
        test_rnn_src_samples,
        test_rnn_dst_samples,
    ) = pyg_to_training_samples.build_node_anchor_link_prediction_samples_from_pyg_heterodata(
        hetero_data=test_data,
        sample_edge_type=sample_edge_type,
        graph_metadata_pb_wrapper=mocked_dataset_info.graph_metadata_pb_wrapper,
        mocked_dataset_info=mocked_dataset_info,
    )

    # Write out to GbmlConfig-specified paths
    output_paths = (
        gbml_config_pb.shared_config.dataset_metadata.node_anchor_based_link_prediction_dataset
    )

    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=train_na_samples,
        uri_prefix=UriFactory.create_uri(uri=output_paths.train_main_data_uri),
        sample_type_for_logging="train NA",
    )

    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=val_na_samples,
        uri_prefix=UriFactory.create_uri(output_paths.val_main_data_uri),
        sample_type_for_logging="val NA",
    )

    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=test_na_samples,
        uri_prefix=UriFactory.create_uri(output_paths.test_main_data_uri),
        sample_type_for_logging="test NA",
    )

    # We only need to generate appropriately split RNNs for the target (dst) nodes.
    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=train_rnn_dst_samples,
        uri_prefix=UriFactory.create_uri(
            output_paths.train_node_type_to_random_negative_data_uri[
                sample_edge_type.dst_node_type
            ]
        ),
        sample_type_for_logging="train RNN (dst)",
    )

    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=val_rnn_dst_samples,
        uri_prefix=UriFactory.create_uri(
            output_paths.val_node_type_to_random_negative_data_uri[
                sample_edge_type.dst_node_type
            ]
        ),
        sample_type_for_logging="val RNN (dst)",
    )

    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=test_rnn_dst_samples,
        uri_prefix=UriFactory.create_uri(
            output_paths.test_node_type_to_random_negative_data_uri[
                sample_edge_type.dst_node_type
            ]
        ),
        sample_type_for_logging="test RNN (dst)",
    )
