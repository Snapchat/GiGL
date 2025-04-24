from typing import Optional, Set

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.task_metadata import TaskMetadataPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.common.utils.file_loader import FileLoader
from snapchat.research.gbml import gbml_config_pb2

logger = Logger()
file_loader = FileLoader()


def assert_asset_exists(
    resource_name: str,
    uri: Uri,
    file_name_suffix: Optional[str] = None,
) -> None:
    logger.info(
        f"Config validation check: if {resource_name} at {uri}*{file_name_suffix} exists."
    )
    if file_loader.count_assets(uri_prefix=uri, suffix=file_name_suffix) < 1:
        raise ValueError(
            f"Required resource does not exist, "
            f"file path specified in frozen config: sharedConfig - {resource_name}. "
            f"'{resource_name}' provided: {uri} "
        )


def assert_preprocessed_metadata_exists(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if preprocessed metadata file exists.
    """
    assert_asset_exists(
        resource_name="preprocessedMetadataUri",
        uri=UriFactory.create_uri(
            gbml_config_pb.shared_config.preprocessed_metadata_uri
        ),
    )


def assert_trained_model_exists(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if trained model file exists.
    """
    gbml_config_pb_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)
    if gbml_config_pb_wrapper.should_use_experimental_glt_backend:
        logger.warning(
            "Skipping trained model check since GLT Backend is being used."
            + "Currently it is not expected that model be piped in through gigl specific configs. "
            + "This will be updated in the future."
        )
        return

    assert_asset_exists(
        resource_name="trainedModelUri",
        uri=UriFactory.create_uri(
            gbml_config_pb.shared_config.trained_model_metadata.trained_model_uri
        ),
    )


def assert_split_generator_output_exists(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if split generator output files exist.
    """
    gbml_config_pb_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)
    if gbml_config_pb_wrapper.should_use_experimental_glt_backend:
        logger.warning(
            "Skipping split generator output check since GLT Backend is being used."
        )
        return
    task_metadata_type = (
        gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_type
    )
    dataset_metadata_pb = gbml_config_pb.shared_config.dataset_metadata
    if task_metadata_type == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK:
        # node types for which random negative samples are generated
        # Only target node types are considered for random negative samples in Split Genenator
        random_negative_node_types = gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_node_types(
            should_include_src_nodes=False,
            should_include_dst_nodes=True,
        )

        if not gbml_config_pb.shared_config.should_skip_training:
            assert_asset_exists(
                resource_name="trainMainDataUri",
                uri=UriFactory.create_uri(
                    dataset_metadata_pb.node_anchor_based_link_prediction_dataset.train_main_data_uri
                ),
                file_name_suffix=".tfrecord",
            )
            assert_asset_exists(
                resource_name="valMainDataUri",
                uri=UriFactory.create_uri(
                    dataset_metadata_pb.node_anchor_based_link_prediction_dataset.val_main_data_uri
                ),
                file_name_suffix=".tfrecord",
            )
            for node_type in random_negative_node_types:
                assert_asset_exists(
                    resource_name="trainRandomNegativeDataUri",
                    uri=UriFactory.create_uri(
                        dataset_metadata_pb.node_anchor_based_link_prediction_dataset.train_node_type_to_random_negative_data_uri[
                            node_type
                        ]
                    ),
                    file_name_suffix=".tfrecord",
                )
                assert_asset_exists(
                    resource_name="valRandomNegativeDataUri",
                    uri=UriFactory.create_uri(
                        dataset_metadata_pb.node_anchor_based_link_prediction_dataset.val_node_type_to_random_negative_data_uri[
                            node_type
                        ]
                    ),
                    file_name_suffix=".tfrecord",
                )
        assert_asset_exists(
            resource_name="testMainDataUri",
            uri=UriFactory.create_uri(
                dataset_metadata_pb.node_anchor_based_link_prediction_dataset.test_main_data_uri
            ),
            file_name_suffix=".tfrecord",
        )
        for node_type in random_negative_node_types:
            assert_asset_exists(
                resource_name="testRandomNegativeDataUri",
                uri=UriFactory.create_uri(
                    dataset_metadata_pb.node_anchor_based_link_prediction_dataset.test_node_type_to_random_negative_data_uri[
                        node_type
                    ]
                ),
                file_name_suffix=".tfrecord",
            )
    elif task_metadata_type == TaskMetadataType.NODE_BASED_TASK:
        if not gbml_config_pb.shared_config.should_skip_training:
            assert_asset_exists(
                resource_name="trainDataUri",
                uri=UriFactory.create_uri(
                    dataset_metadata_pb.supervised_node_classification_dataset.train_data_uri
                ),
                file_name_suffix=".tfrecord",
            )
            assert_asset_exists(
                resource_name="valDataUri",
                uri=UriFactory.create_uri(
                    dataset_metadata_pb.supervised_node_classification_dataset.val_data_uri
                ),
                file_name_suffix=".tfrecord",
            )
        assert_asset_exists(
            resource_name="testDataUri",
            uri=UriFactory.create_uri(
                dataset_metadata_pb.supervised_node_classification_dataset.test_data_uri
            ),
            file_name_suffix=".tfrecord",
        )
    elif task_metadata_type == TaskMetadataType.LINK_BASED_TASK:
        if not gbml_config_pb.shared_config.should_skip_training:
            assert_asset_exists(
                resource_name="trainDataUri",
                uri=UriFactory.create_uri(
                    dataset_metadata_pb.supervised_link_based_task_dataset.train_data_uri
                ),
                file_name_suffix=".tfrecord",
            )
            assert_asset_exists(
                resource_name="valDataUri",
                uri=UriFactory.create_uri(
                    dataset_metadata_pb.supervised_link_based_task_dataset.val_data_uri
                ),
                file_name_suffix=".tfrecord",
            )
        assert_asset_exists(
            resource_name="testDataUri",
            uri=UriFactory.create_uri(
                dataset_metadata_pb.supervised_link_based_task_dataset.test_data_uri
            ),
            file_name_suffix=".tfrecord",
        )


def assert_subgraph_sampler_output_exists(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if subgraph sampler output files exist.
    """
    gbml_config_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)
    if gbml_config_wrapper.should_use_experimental_glt_backend:
        logger.warning(
            "Skipping subgraph sampler output check since GLT Backend is being used."
        )
        return

    task_metadata_wrapper = TaskMetadataPbWrapper(gbml_config_pb.task_metadata)
    flattened_graph_metadata_pb = gbml_config_pb.shared_config.flattened_graph_metadata
    if (
        task_metadata_wrapper.task_metadata_type
        == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
    ):
        assert_asset_exists(
            resource_name="tfrecordUriPrefix",
            uri=UriFactory.create_uri(
                flattened_graph_metadata_pb.node_anchor_based_link_prediction_output.tfrecord_uri_prefix
            ),
            file_name_suffix=".tfrecord",
        )

        assert isinstance(
            task_metadata_wrapper.task_metadata,
            gbml_config_pb2.GbmlConfig.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata,
        )
        random_negative_node_types: Set[str] = set()
        for (
            supervision_edge_type
        ) in task_metadata_wrapper.task_metadata.supervision_edge_types:
            random_negative_node_types.add(supervision_edge_type.src_node_type)
            random_negative_node_types.add(supervision_edge_type.dst_node_type)

        for node_type in random_negative_node_types:
            assert_asset_exists(
                resource_name="randomNegativeTfrecordUriPrefix",
                uri=UriFactory.create_uri(
                    flattened_graph_metadata_pb.node_anchor_based_link_prediction_output.node_type_to_random_negative_tfrecord_uri_prefix[
                        node_type
                    ]
                ),
                file_name_suffix=".tfrecord",
            )
    elif task_metadata_wrapper.task_metadata_type == TaskMetadataType.NODE_BASED_TASK:
        assert_asset_exists(
            resource_name="labeledTfrecordUriPrefix",
            uri=UriFactory.create_uri(
                flattened_graph_metadata_pb.supervised_node_classification_output.labeled_tfrecord_uri_prefix
            ),
            file_name_suffix=".tfrecord",
        )
        assert_asset_exists(
            resource_name="unlabeledTfrecordUriPrefix",
            uri=UriFactory.create_uri(
                flattened_graph_metadata_pb.supervised_node_classification_output.unlabeled_tfrecord_uri_prefix
            ),
            file_name_suffix=".tfrecord",
        )
    elif task_metadata_wrapper.task_metadata_type == TaskMetadataType.LINK_BASED_TASK:
        assert_asset_exists(
            resource_name="labeledTfrecordUriPrefix",
            uri=UriFactory.create_uri(
                flattened_graph_metadata_pb.supervised_link_based_task_output.labeled_tfrecord_uri_prefix
            ),
            file_name_suffix=".tfrecord",
        )
        assert_asset_exists(
            resource_name="unlabeledTfrecordUriPrefix",
            uri=UriFactory.create_uri(
                flattened_graph_metadata_pb.supervised_link_based_task_output.unlabeled_tfrecord_uri_prefix
            ),
            file_name_suffix=".tfrecord",
        )
