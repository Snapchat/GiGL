import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from gigl.common import GcsUri, LocalUri, UriFactory
from gigl.common.constants import (
    SPARK_31_TFRECORD_JAR_LOCAL_PATH,
    SPARK_35_TFRECORD_JAR_LOCAL_PATH,
)
from gigl.common.logger import Logger
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.constants.local_fs import get_project_root_directory
from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    EdgeType,
    EdgeUsageType,
    NodeType,
)
from gigl.src.common.types.pb_wrappers.dataset_metadata_utils import (
    read_training_sample_protos_from_tfrecords,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.graph_data_types import (
    EdgePbWrapper,
    NodePbWrapper,
)
from gigl.src.common.types.pb_wrappers.graph_data_types_utils import (
    get_dehydrated_node_pb_wrappers_from_edge_wrapper,
)
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.training.v1.lib.data_loaders.tf_records_iterable_dataset import (
    get_np_iterator_from_tfrecords,
)
from snapchat.research.gbml import (
    gbml_config_pb2,
    graph_schema_pb2,
    preprocessed_metadata_pb2,
    training_samples_schema_pb2,
)

logger = Logger()


@dataclass
class EdgeMetadataInfo:
    feasible_adjacency_list_map: Dict[NodePbWrapper, List[EdgePbWrapper]]
    edge_type_to_edge_to_features_map: Dict[EdgeType, Dict[EdgePbWrapper, List[float]]]


@dataclass
class ExpectedGraphFromPreprocessor:
    """
    Represents the expected graph structure and information from Data Preprocessor output.

    This includes:
    - A map from NodeType to a map from NodePbWrapper to a list of features for that node.
    - Edge metadata for main graph edges.
    - Edge metadata for positive user-defined label edges.
    - Edge metadata for negative user-defined label edges.
    """

    node_type_to_node_to_features_map: Dict[NodeType, Dict[NodePbWrapper, List[float]]]
    main_edge_info: EdgeMetadataInfo
    pos_edge_info: EdgeMetadataInfo
    neg_edge_info: EdgeMetadataInfo


def read_output_nablp_samples_from_subgraph_sampler(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> Tuple[
    Dict[NodeType, List[training_samples_schema_pb2.RootedNodeNeighborhood]],
    List[training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample],
]:
    """
    Reads the output RNN samples keyed by NodeType, as well as the output training samples.

    Note that this function is only recommended for small graphs.
    """

    node_anchor_based_link_prediction_output = (
        gbml_config_pb_wrapper.gbml_config_pb.shared_config.flattened_graph_metadata.node_anchor_based_link_prediction_output
    )

    node_type_to_rooted_neighborhood_samples: Dict[
        NodeType, List[training_samples_schema_pb2.RootedNodeNeighborhood]
    ] = defaultdict(list)
    for (
        node_type,
        random_negative_tfrecord_uri_prefix,
    ) in (
        node_anchor_based_link_prediction_output.node_type_to_random_negative_tfrecord_uri_prefix.items()
    ):
        node_type_to_rooted_neighborhood_samples[NodeType(node_type)].extend(
            read_training_sample_protos_from_tfrecords(
                uri_prefix=UriFactory.create_uri(
                    uri=random_negative_tfrecord_uri_prefix
                ),
                proto_cls=training_samples_schema_pb2.RootedNodeNeighborhood,
            )
        )
    samples: List[
        training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample
    ] = read_training_sample_protos_from_tfrecords(
        uri_prefix=UriFactory.create_uri(
            uri=node_anchor_based_link_prediction_output.tfrecord_uri_prefix
        ),
        proto_cls=training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
    )
    return (node_type_to_rooted_neighborhood_samples, samples)


def read_output_node_based_task_samples_from_subgraph_sampler(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> Tuple[
    List[training_samples_schema_pb2.RootedNodeNeighborhood],
    List[training_samples_schema_pb2.RootedNodeNeighborhood],
]:
    """
    Reads the output RNN samples for both labeled and unlabeled data.

    Note that this function is only recommended for small graphs.
    """

    node_based_task_output = (
        gbml_config_pb_wrapper.gbml_config_pb.shared_config.flattened_graph_metadata.supervised_node_classification_output
    )

    labeled_rooted_neighborhood_samples = read_training_sample_protos_from_tfrecords(
        uri_prefix=UriFactory.create_uri(
            uri=node_based_task_output.labeled_tfrecord_uri_prefix
        ),
        proto_cls=training_samples_schema_pb2.RootedNodeNeighborhood,
    )

    unlabeled_rooted_neighborhood_samples = read_training_sample_protos_from_tfrecords(
        uri_prefix=UriFactory.create_uri(
            uri=node_based_task_output.unlabeled_tfrecord_uri_prefix
        ),
        proto_cls=training_samples_schema_pb2.RootedNodeNeighborhood,
    )

    return (labeled_rooted_neighborhood_samples, unlabeled_rooted_neighborhood_samples)


def _build_node_features_map(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> Dict[NodeType, Dict[NodePbWrapper, List[float]]]:
    """
    Builds a map from NodeType to a map from NodePbWrapper to a list of features for that node, for all NodeTypes encountered in preprocessed output.
    """

    preprocessed_metadata_pb = (
        gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb
    )

    node_type_to_node_to_features_map: Dict[
        NodeType, Dict[NodePbWrapper, List[float]]
    ] = {}
    for (
        condensed_node_type,
        node_metadata_output,
    ) in preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata.items():
        node_type: NodeType = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
            CondensedNodeType(condensed_node_type)
        ]
        assert node_metadata_output is not None
        tfrecord_files = tf.io.gfile.glob(
            f"{node_metadata_output.tfrecord_uri_prefix}*.tfrecord"
        )
        nodes_records = get_np_iterator_from_tfrecords(
            schema_path=UriFactory.create_uri(node_metadata_output.schema_uri),
            tfrecord_files=tfrecord_files,
        )
        node_type_to_node_to_features_map[node_type] = {}
        sorted_feature_key_list = list(node_metadata_output.feature_keys)
        for record in nodes_records:
            node_pbw = NodePbWrapper(
                pb=graph_schema_pb2.Node(
                    node_id=record[node_metadata_output.node_id_key][0],
                    condensed_node_type=condensed_node_type,
                )
            )
            features = [record[key][0] for key in sorted_feature_key_list]
            node_type_to_node_to_features_map[node_type][node_pbw] = features

    return node_type_to_node_to_features_map


def _build_edge_features_map(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    edge_usage_type: EdgeUsageType = EdgeUsageType.MAIN,
) -> Dict[EdgeType, Dict[EdgePbWrapper, List[float]]]:
    """
    Builds a map from EdgeType to a map from EdgePbWrapper to a list of features for that edge, for all EdgeTypes encountered in preprocessed output.
    """

    preprocessed_metadata_pb = (
        gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb
    )

    edge_type_to_edge_to_features_map: Dict[
        EdgeType, Dict[EdgePbWrapper, List[float]]
    ] = {}
    for (
        condensed_edge_type,
        edge_metadata_output,
    ) in preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata.items():
        edge_type: EdgeType = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map[
            CondensedEdgeType(condensed_edge_type)
        ]
        assert edge_metadata_output is not None
        edge_metadata_info: Optional[
            preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo
        ] = getattr(edge_metadata_output, f"{edge_usage_type.value}_edge_info", None)
        if not edge_metadata_info or not edge_metadata_info.tfrecord_uri_prefix:
            continue
        tfrecord_files = tf.io.gfile.glob(
            f"{edge_metadata_info.tfrecord_uri_prefix}*.tfrecord"
        )
        edges_records = get_np_iterator_from_tfrecords(
            schema_path=UriFactory.create_uri(edge_metadata_info.schema_uri),
            tfrecord_files=tfrecord_files,
        )
        edge_type_to_edge_to_features_map[edge_type] = {}
        sorted_feature_key_list = list(edge_metadata_info.feature_keys)
        for record in edges_records:
            edge_pbw = EdgePbWrapper(
                pb=graph_schema_pb2.Edge(
                    src_node_id=record[edge_metadata_output.src_node_id_key][0],
                    dst_node_id=record[edge_metadata_output.dst_node_id_key][0],
                    condensed_edge_type=condensed_edge_type,
                )
            )
            features = [record[key][0] for key in sorted_feature_key_list]
            edge_type_to_edge_to_features_map[edge_type][edge_pbw] = features

    return edge_type_to_edge_to_features_map


def _build_feasible_adjacency_list_map(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    edge_usage_type: EdgeUsageType = EdgeUsageType.MAIN,
) -> Dict[NodePbWrapper, List[EdgePbWrapper]]:
    """
    Builds a map from NodePbWrapper to a list of EdgePbWrappers, representing the adjacency list for each src node,
    for all nodes encountered in Data Preprocessor output.  This will be used to test feasibility of edges which
    exist in SGS output.
    """

    preprocessed_metadata_pb = (
        gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb
    )
    graph_metadata_pb_wrapper = gbml_config_pb_wrapper.graph_metadata_pb_wrapper

    src_node_to_edge_map: Dict[NodePbWrapper, List[EdgePbWrapper]] = defaultdict(list)

    for (
        condensed_edge_type,
        edge_metadata_output,
    ) in preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata.items():
        assert edge_metadata_output is not None
        edge_metadata_info: Optional[
            preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo
        ] = getattr(edge_metadata_output, f"{edge_usage_type.value}_edge_info", None)
        if not edge_metadata_info or not edge_metadata_info.tfrecord_uri_prefix:
            continue
        tfrecord_files = tf.io.gfile.glob(
            f"{edge_metadata_info.tfrecord_uri_prefix}*.tfrecord"
        )
        edge_records = get_np_iterator_from_tfrecords(
            schema_path=UriFactory.create_uri(edge_metadata_info.schema_uri),
            tfrecord_files=tfrecord_files,
        )

        (
            condensed_src_node_type,
            _,
        ) = graph_metadata_pb_wrapper.condensed_edge_type_to_condensed_node_types[
            CondensedEdgeType(condensed_edge_type)
        ]
        for record in edge_records:
            src_node_int_id = record[edge_metadata_output.src_node_id_key][0]
            dst_node_int_id = record[edge_metadata_output.dst_node_id_key][0]
            src_node_pbw = NodePbWrapper(
                pb=graph_schema_pb2.Node(
                    node_id=src_node_int_id,
                    condensed_node_type=condensed_src_node_type,
                )
            )
            edge_pbw = EdgePbWrapper(
                pb=graph_schema_pb2.Edge(
                    src_node_id=src_node_int_id,
                    dst_node_id=dst_node_int_id,
                    condensed_edge_type=condensed_edge_type,
                )
            )
            src_node_to_edge_map[src_node_pbw].append(edge_pbw)

    return src_node_to_edge_map


def bidirectionalize_feasible_adjacency_list_map(
    src_node_to_edge_map: Dict[NodePbWrapper, List[EdgePbWrapper]],
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> Dict[NodePbWrapper, List[EdgePbWrapper]]:
    """
    Given an adjacency list map from NodePbWrapper to a list of EdgePbWrappers, this function
    returns a bidirectional adjacency list map applied to main graph edges.

    That is, for any edge (src=A, dst=B, edge_type=C, features=D) in the original map,
    the bidirectional map will contain two edges:
    - (src=A, dst=B, edge_type=C, features=D)
    - (src=B, dst=A, edge_type=C, features=D)

    This is used in cases where the SGS output is expected to contain bidirectional edges.
    Note that bidirectionalization logic is only valid for homogeneous graphs.
    """

    assert (
        not gbml_config_pb_wrapper.graph_metadata_pb_wrapper.is_heterogeneous
    ), "Bidirectionalizing adjacency list map is only supported for homogeneous graphs."

    bidirectional_adjacency_list_map: Dict[
        NodePbWrapper, List[EdgePbWrapper]
    ] = defaultdict(list)
    for _, edge_pbws in src_node_to_edge_map.items():
        for edge_pbw in edge_pbws:
            (
                src_node_pbw,
                dst_node_pbw,
            ) = get_dehydrated_node_pb_wrappers_from_edge_wrapper(
                edge_pb_wrapper=edge_pbw,
                graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            )
            bidirectional_adjacency_list_map[src_node_pbw].append(edge_pbw)
            bidirectional_adjacency_list_map[dst_node_pbw].append(edge_pbw.flip_edge())

    return bidirectional_adjacency_list_map


def bidirectionalize_edge_type_to_edge_to_features_map(
    edge_type_to_edge_to_features_map: Dict[EdgeType, Dict[EdgePbWrapper, List[float]]],
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> Dict[EdgeType, Dict[EdgePbWrapper, List[float]]]:
    """
    Given a map from EdgeType to a map from EdgePbWrapper to a list of features for that edge, this function
    returns a bidirectional map.

    That is, for any edge (src=A, dst=B, edge_type=C, features=D) in the original map,
    the bidirectional map will contain two edges:
    - (src=A, dst=B, edge_type=C, features=D)
    - (src=B, dst=A, edge_type=C, features=D)

    This is used in cases where the SGS output is expected to contain bidirectional edges.
    Note that bidirectionalization logic is only valid for homogeneous graphs.
    """

    assert (
        not gbml_config_pb_wrapper.graph_metadata_pb_wrapper.is_heterogeneous
    ), "Bidirectionalizing edge type to edge to features map is only supported for homogeneous graphs."

    bidirectional_edge_type_to_edge_to_features_map: Dict[
        EdgeType, Dict[EdgePbWrapper, List[float]]
    ] = {}
    for edge_type, edge_to_features_map in edge_type_to_edge_to_features_map.items():
        bidirectional_edge_to_features_map: Dict[EdgePbWrapper, List[float]] = {}
        for edge_pbw, features in edge_to_features_map.items():
            bidirectional_edge_pbw = edge_pbw.flip_edge()
            bidirectional_edge_to_features_map[edge_pbw] = features
            bidirectional_edge_to_features_map[bidirectional_edge_pbw] = features
        bidirectional_edge_type_to_edge_to_features_map[
            edge_type
        ] = bidirectional_edge_to_features_map

    return bidirectional_edge_type_to_edge_to_features_map


def reconstruct_graph_information_from_preprocessor_output(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> ExpectedGraphFromPreprocessor:
    """
    This function uses the preprocessed output to construct the following maps which represent "truth" about the graph:
    - A map from NodeType to a map from NodePbWrapper to a list of features for that node.
    - Edge metadata for main graph edges.
    - Edge metadata for positive user-defined label edges.
    - Edge metadata for negative user-defined label edges.

    Note that this amounts to fully reconstructing the graph from the preprocessed output,
    and hence is only recommended for small graphs.
    """

    node_type_to_node_to_features_map = _build_node_features_map(
        gbml_config_pb_wrapper=gbml_config_pb_wrapper
    )

    main_edge_metadata_info = EdgeMetadataInfo(
        feasible_adjacency_list_map=_build_feasible_adjacency_list_map(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            edge_usage_type=EdgeUsageType.MAIN,
        ),
        edge_type_to_edge_to_features_map=_build_edge_features_map(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            edge_usage_type=EdgeUsageType.MAIN,
        ),
    )

    pos_edge_metadata_info = EdgeMetadataInfo(
        feasible_adjacency_list_map=_build_feasible_adjacency_list_map(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            edge_usage_type=EdgeUsageType.POSITIVE,
        ),
        edge_type_to_edge_to_features_map=_build_edge_features_map(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            edge_usage_type=EdgeUsageType.POSITIVE,
        ),
    )

    neg_edge_metadata_info = EdgeMetadataInfo(
        feasible_adjacency_list_map=_build_feasible_adjacency_list_map(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            edge_usage_type=EdgeUsageType.NEGATIVE,
        ),
        edge_type_to_edge_to_features_map=_build_edge_features_map(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            edge_usage_type=EdgeUsageType.NEGATIVE,
        ),
    )

    return ExpectedGraphFromPreprocessor(
        node_type_to_node_to_features_map=node_type_to_node_to_features_map,
        main_edge_info=main_edge_metadata_info,
        pos_edge_info=pos_edge_metadata_info,
        neg_edge_info=neg_edge_metadata_info,
    )


def overwrite_subgraph_sampler_output_paths_to_local(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    subgraph_sampler_config_pb: gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig,
) -> LocalUri:
    # Override the frozen config with configurable test params
    subgraph_sampler_config = (
        gbml_config_pb_wrapper.gbml_config_pb.dataset_config.subgraph_sampler_config
    )
    subgraph_sampler_config.CopyFrom(subgraph_sampler_config_pb)

    # Overwrite output paths.
    tmp_dir_path = tempfile.mkdtemp()
    tmp_subgraph_sampler_dir = LocalUri.join(tmp_dir_path, "subgraph_sampler")

    if (
        gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_type
        == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
    ):
        flattened_nablp_output_dataset = (
            gbml_config_pb_wrapper.gbml_config_pb.shared_config.flattened_graph_metadata.node_anchor_based_link_prediction_output
        )
        for (
            node_type
        ) in (
            flattened_nablp_output_dataset.node_type_to_random_negative_tfrecord_uri_prefix
        ):
            flattened_nablp_output_dataset.node_type_to_random_negative_tfrecord_uri_prefix[
                node_type
            ] = LocalUri.join(
                tmp_subgraph_sampler_dir,
                "random_negative_rooted_neighborhood_samples",
                node_type,
                "samples/",
            ).uri

        # TODO(nshah-sc): This should be revisited when NABLP Output proto supports multiple supervision edge types.
        supervision_edge_types = (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_pb.node_anchor_based_link_prediction_task_metadata.supervision_edge_types
        )
        assert (
            len(supervision_edge_types) == 1
        ), "Only one supervision edge type is currently supported."
        supervision_edge_type = GbmlProtosTranslator.edge_type_from_EdgeTypePb(
            edge_type_pb=supervision_edge_types[0]
        )
        flattened_nablp_output_dataset.tfrecord_uri_prefix = LocalUri.join(
            tmp_subgraph_sampler_dir,
            "node_anchor_based_link_prediction_samples",
            str(supervision_edge_type),
            "samples/",
        ).uri
        frozen_gbml_config_uri = LocalUri.join(
            tmp_subgraph_sampler_dir, "sgs_input_node_anchor_link_prediction.yaml"
        )
    elif (
        gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_type
        == TaskMetadataType.NODE_BASED_TASK
    ):
        flattened_node_based_task_output_dataset = (
            gbml_config_pb_wrapper.gbml_config_pb.shared_config.flattened_graph_metadata.supervised_node_classification_output
        )

        # TODO(nshah-sc): This should be revisited when SNC Output proto supports multiple supervision node types.
        supervision_node_types = [
            NodeType(node_type)
            for node_type in gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_pb.node_based_task_metadata.supervision_node_types
        ]
        assert (
            len(supervision_node_types) == 1
        ), "Only one supervision node type is currently supported."
        supervision_node_type = supervision_node_types[0]
        flattened_node_based_task_output_dataset.labeled_tfrecord_uri_prefix = (
            LocalUri.join(
                tmp_subgraph_sampler_dir,
                "labeled_rooted_neighborhood_samples",
                str(supervision_node_type),
                "samples/",
            ).uri
        )
        flattened_node_based_task_output_dataset.unlabeled_tfrecord_uri_prefix = (
            LocalUri.join(
                tmp_subgraph_sampler_dir,
                "unlabeled_rooted_neighborhood_samples",
                str(supervision_node_type),
                "samples/",
            ).uri
        )
        frozen_gbml_config_uri = LocalUri.join(
            tmp_subgraph_sampler_dir, "sgs_input_node_based_task.yaml"
        )
    else:
        raise NotImplementedError(
            f"Task metadata type {gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_type} is not supported."
        )

    proto_utils = ProtoUtils()
    proto_utils.write_proto_to_yaml(
        proto=gbml_config_pb_wrapper.gbml_config_pb, uri=frozen_gbml_config_uri
    )
    return frozen_gbml_config_uri


def overwrite_subgraph_sampler_downloaded_assets_paths_to_local(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    gcs_dir: GcsUri,
    local_dir: LocalUri,
):
    """
    First overwrites the preprocessed_metadata_uri referenced in gbml_config_pb to local.
    Then, opens the file at preprocessed_metadata_uri and overwrites those paths to local too.
    """

    preprocessed_metadata_uri = (
        gbml_config_pb_wrapper.gbml_config_pb.shared_config.preprocessed_metadata_uri
    )
    gbml_config_pb_wrapper.gbml_config_pb.shared_config.preprocessed_metadata_uri = (
        preprocessed_metadata_uri.replace(gcs_dir.uri, local_dir.uri)
    )
    with open(
        gbml_config_pb_wrapper.gbml_config_pb.shared_config.preprocessed_metadata_uri,
        "r+",
    ) as f:
        content = f.read()
        f.seek(0)
        new_content = content.replace(gcs_dir.uri, local_dir.uri)
        f.write(new_content)
        f.truncate()


def compile_and_run_sgs_pipeline_locally(
    frozen_gbml_config_uri: LocalUri,
    resource_config_uri: LocalUri,
    use_spark35: bool = False,
):
    _PATH_TO_SGS_SCALA_ROOT = os.path.join(get_project_root_directory(), "scala")
    _PATH_TO_SGS_SCALA_SPARK_35_ROOT = os.path.join(
        get_project_root_directory(), "scala_spark35"
    )
    _SCALA_TOOLS_PATH = os.path.join(get_project_root_directory(), "tools", "scala")
    _PATH_TO_SPARK_SUBMIT = os.path.join(
        _SCALA_TOOLS_PATH, "spark-3.1.3-bin-hadoop3.2", "bin", "spark-submit"
    )
    _PATH_TO_SPARK35_SUBMIT = os.path.join(
        _SCALA_TOOLS_PATH, "spark-3.5.0-bin-hadoop3", "bin", "spark-submit"
    )
    _SGS_SCALA_PROJECT_NAME = "subgraph_sampler"

    commands: List[str]
    if use_spark35:
        commands = [
            f"cd {_PATH_TO_SGS_SCALA_SPARK_35_ROOT} && sbt {_SGS_SCALA_PROJECT_NAME}/assembly",
            f"""{_PATH_TO_SPARK35_SUBMIT} \\
            --class Main \\
            --master local \\
            --jars {SPARK_35_TFRECORD_JAR_LOCAL_PATH} \\
            {_PATH_TO_SGS_SCALA_SPARK_35_ROOT}/{_SGS_SCALA_PROJECT_NAME}/target/scala-2.12/subgraph_sampler-assembly-1.0.jar\\
            {frozen_gbml_config_uri.uri} \\
            sgs_integration_test_{current_formatted_datetime()} \\
            {resource_config_uri.uri}""",
        ]

    else:
        commands = [
            f"cd {_PATH_TO_SGS_SCALA_ROOT} && sbt {_SGS_SCALA_PROJECT_NAME}/assembly",
            f"""{_PATH_TO_SPARK_SUBMIT} \\
            --class Main \\
            --master local \\
            --jars {SPARK_31_TFRECORD_JAR_LOCAL_PATH} \\
            {_PATH_TO_SGS_SCALA_ROOT}/{_SGS_SCALA_PROJECT_NAME}/target/scala-2.12/subgraph_sampler-assembly-1.0.jar \\
            {frozen_gbml_config_uri.uri} \\
            sgs_integration_test_{current_formatted_datetime()} \\
            {resource_config_uri.uri}""",
        ]

    for command in commands:
        logger.info(f"Running following command: {command}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        output = output or b""
        error = error or b""

        logger.info(f"STDOUT:")
        logger.info(output.decode(encoding="utf-8"))
        logger.info(f"STDERR:")
        logger.info(error.decode(encoding="utf-8"))
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with return code {process.returncode}")
