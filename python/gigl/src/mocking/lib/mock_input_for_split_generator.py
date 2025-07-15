from typing import List

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.mocking.lib import pyg_to_training_samples, tfrecord_io
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from snapchat.research.gbml import gbml_config_pb2, training_samples_schema_pb2

logger = Logger()


def build_and_write_supervised_node_classification_subgraph_samples_from_mocked_dataset_info(
    mocked_dataset_info: MockedDatasetInfo,
    root_node_type: NodeType,
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> HeteroData:
    hetero_data = pyg_to_training_samples.build_pyg_heterodata_from_mocked_dataset_info(
        mocked_dataset_info=mocked_dataset_info
    )

    samples: List[
        training_samples_schema_pb2.SupervisedNodeClassificationSample
    ] = pyg_to_training_samples.build_supervised_node_classification_samples_from_pyg_heterodata(
        hetero_data=hetero_data,
        root_node_type=root_node_type,
        graph_metadata_pb_wrapper=mocked_dataset_info.graph_metadata_pb_wrapper,
    )

    # Write out to GbmlConfig-specified paths
    output_paths = (
        gbml_config_pb.shared_config.flattened_graph_metadata.supervised_node_classification_output
    )
    labeled_sample_tfrecord_uri = UriFactory.create_uri(
        output_paths.labeled_tfrecord_uri_prefix
    )
    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=samples,
        uri_prefix=labeled_sample_tfrecord_uri,
        sample_type_for_logging="labeled SNC",
    )

    return hetero_data


def build_and_write_node_anchor_link_prediction_subgraph_samples_from_mocked_dataset_info(
    mocked_dataset_info: MockedDatasetInfo,
    sample_edge_type: EdgeType,
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> HeteroData:
    hetero_data = pyg_to_training_samples.build_pyg_heterodata_from_mocked_dataset_info(
        mocked_dataset_info=mocked_dataset_info
    )

    # This doesn't really "split" the data, it is just a trick to help generate
    # positive / hard negative for each node, similar to what Subgraph Sampler does.
    dummy_split_cls = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=0,
        edge_types=[(sample_edge_type[0], sample_edge_type[1], sample_edge_type[2])],
    )
    unsplit_hetero_data, _, _ = dummy_split_cls(hetero_data)

    na_samples: List[training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample]
    rnn_src_samples: List[training_samples_schema_pb2.RootedNodeNeighborhood]
    rnn_dst_samples: List[training_samples_schema_pb2.RootedNodeNeighborhood]
    (
        na_samples,
        rnn_src_samples,
        rnn_dst_samples,
    ) = pyg_to_training_samples.build_node_anchor_link_prediction_samples_from_pyg_heterodata(
        hetero_data=unsplit_hetero_data,
        sample_edge_type=sample_edge_type,
        graph_metadata_pb_wrapper=mocked_dataset_info.graph_metadata_pb_wrapper,
        mocked_dataset_info=mocked_dataset_info,
    )

    # Write out to GbmlConfig-specified paths
    output_paths = (
        gbml_config_pb.shared_config.flattened_graph_metadata.node_anchor_based_link_prediction_output
    )
    main_sample_tfrecord_uri_prefix = UriFactory.create_uri(
        output_paths.tfrecord_uri_prefix
    )

    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=na_samples,
        uri_prefix=main_sample_tfrecord_uri_prefix,
        sample_type_for_logging="UNA",
    )

    # TODO(Tong): storing only the src and dst samples for a single sample_edge_type for the 1st stage of HGS support
    #             as defined in the design doc. Should be update to support multiple sample_edge_types in the future.
    tfrecord_io.write_pb_tfrecord_shards_to_uri(
        pb_samples=rnn_dst_samples,
        uri_prefix=UriFactory.create_uri(
            output_paths.node_type_to_random_negative_tfrecord_uri_prefix[
                sample_edge_type.dst_node_type
            ]
        ),
        sample_type_for_logging="RNN (dst)",
    )

    if sample_edge_type.src_node_type != sample_edge_type.dst_node_type:
        tfrecord_io.write_pb_tfrecord_shards_to_uri(
            pb_samples=rnn_src_samples,
            uri_prefix=UriFactory.create_uri(
                output_paths.node_type_to_random_negative_tfrecord_uri_prefix[
                    sample_edge_type.src_node_type
                ]
            ),
            sample_type_for_logging="RNN (src)",
        )

    return hetero_data
