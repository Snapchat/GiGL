import argparse
import concurrent.futures
import sys
import threading
from collections import defaultdict
from itertools import chain, repeat
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
from apache_beam.runners.dataflow.dataflow_runner import DataflowPipelineResult
from apache_beam.runners.runner import PipelineState

import gigl.common.utils.dataflow
import gigl.src.common.constants.gcs as gcs_constants
import gigl.src.common.constants.local_fs as local_fs_constants
import gigl.src.data_preprocessor.lib.transform.utils as transform_utils
from gigl.analytics.graph_validation import BQGraphValidator
from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.metrics.decorators import flushes_metrics, profileit
from gigl.common.utils import os_utils
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.constants.metrics import TIMER_PREPROCESSOR_S
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.features import FeatureTypes
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    EdgeType,
    EdgeUsageType,
    NodeType,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.common.utils.metrics_service_provider import (
    get_metrics_service_instance,
    initialize_metrics,
)
from gigl.src.data_preprocessor.lib.data_preprocessor_config import (
    DataPreprocessorConfig,
)
from gigl.src.data_preprocessor.lib.enumerate.utils import (
    Enumerator,
    EnumeratorEdgeTypeMetadata,
    EnumeratorNodeTypeMetadata,
)
from gigl.src.data_preprocessor.lib.ingest.bigquery import (
    BigqueryEdgeDataReference,
    BigqueryNodeDataReference,
)
from gigl.src.data_preprocessor.lib.ingest.reference import (
    DataReference,
    EdgeDataReference,
    NodeDataReference,
)
from gigl.src.data_preprocessor.lib.transform.transformed_features_info import (
    TransformedFeaturesInfo,
)
from gigl.src.data_preprocessor.lib.types import (
    DEFAULT_TF_INT_DTYPE,
    EdgeDataPreprocessingSpec,
    EdgeOutputIdentifier,
    FeatureSpecDict,
    NodeDataPreprocessingSpec,
    NodeOutputIdentifier,
)
from snapchat.research.gbml import preprocessed_metadata_pb2

logger = Logger()


class PreprocessedMetadataReferences(NamedTuple):
    node_data: Dict[NodeDataReference, TransformedFeaturesInfo]
    edge_data: Dict[EdgeDataReference, TransformedFeaturesInfo]


class DataPreprocessor:
    """
    GiGL Component to read node, edge and respective feature data from multiple data sources, and produce preprocessed / transformed versions of all this data, for subsequent components to use.
    """

    __gbml_config_pb_wrapper: GbmlConfigPbWrapper
    __data_preprocessor_config: DataPreprocessorConfig
    __custom_worker_image_uri: Optional[str]

    def __init__(self) -> None:
        self.__proto_utils = ProtoUtils()

    @property
    def gbml_config_pb_wrapper(self) -> GbmlConfigPbWrapper:
        if not self.__gbml_config_pb_wrapper:
            raise ValueError(f"gbml_config_pb_wrapper is not initialized before use.")
        return self.__gbml_config_pb_wrapper

    @property
    def applied_task_identifier(self) -> AppliedTaskIdentifier:
        if not self.__applied_task_identifier:
            raise ValueError(f"applied_task_identifier is not initialized before use.")
        return self.__applied_task_identifier

    @property
    def data_preprocessor_config(self) -> DataPreprocessorConfig:
        if not self.__data_preprocessor_config:
            raise ValueError(f"data_preprocessor_config is not initialized before use.")
        return self.__data_preprocessor_config

    @property
    def custom_worker_image_uri(self) -> Optional[str]:
        return self.__custom_worker_image_uri

    def __prepare_env(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        custom_worker_image_uri: Optional[str],
    ):
        """
        Reads config from YAML and sets self.__config.
        :param task_config_uri:
        :return:
        """
        self.__applied_task_identifier = applied_task_identifier
        self.__gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=task_config_uri
            )
        )
        self.__data_preprocessor_config = self.__import_data_preprocessor_config()
        self.__custom_worker_image_uri = custom_worker_image_uri
        self.__prepare_staging_paths()

    def __prepare_staging_paths(self) -> None:
        """
        Clean up paths that Data Preprocessor would be writing to in order to avoid clobbering of data.
        These paths are inferred from the GbmlConfig, and the AppliedTaskIdentifier.
        :return:
        """
        logger.info("Preparing staging paths for Data Preprocessor...")
        paths_to_delete: List[Uri] = [
            local_fs_constants.get_gbml_task_local_tmp_path(
                applied_task_identifier=self.applied_task_identifier
            ),
            gcs_constants.get_data_preprocessor_assets_temp_gcs_path(
                applied_task_identifier=self.applied_task_identifier
            ),
            gcs_constants.get_data_preprocessor_assets_perm_gcs_path(
                applied_task_identifier=self.applied_task_identifier
            ),
            UriFactory.create_uri(
                uri=self.gbml_config_pb_wrapper.shared_config.preprocessed_metadata_uri
            ),
        ]
        file_loader = FileLoader()
        file_loader.delete_files(uris=paths_to_delete)
        logger.info("Staging paths for Data Preprocessor prepared.")

    def __import_data_preprocessor_config(self) -> DataPreprocessorConfig:
        """
        Parse DataPreprocessorConfig object from GbmlConfig proto, create an instance, and return it.
        :return:
        """

        data_preprocessor_cls_str: str = (
            self.gbml_config_pb_wrapper.dataset_config.data_preprocessor_config.data_preprocessor_config_cls_path
        )
        data_preprocessor_cls = os_utils.import_obj(data_preprocessor_cls_str)
        kwargs = self.gbml_config_pb_wrapper.dataset_config.data_preprocessor_config.data_preprocessor_args  # type: ignore

        try:
            data_preprocessor_config: DataPreprocessorConfig = data_preprocessor_cls(
                **kwargs
            )
            assert isinstance(data_preprocessor_config, DataPreprocessorConfig)
        except Exception as e:
            logger.error(
                f"Could not instantiate class {data_preprocessor_cls_str}: {e}"
            )
            raise e
        return data_preprocessor_config

    def __cleanup_env(self):
        """
        No-op for now.
        :return:
        """

    def __preprocess_single_data_reference(
        self,
        data_reference: Union[NodeDataReference, EdgeDataReference],
        preprocessing_spec: Union[NodeDataPreprocessingSpec, EdgeDataPreprocessingSpec],
        lock: threading.Lock,
    ) -> TransformedFeaturesInfo:
        """
        Ingests data using a data reference, and subsequently runs the associated TFTransform logic.
        :return:
        """

        feature_type: FeatureTypes
        entity_type: Union[NodeType, EdgeType]

        custom_identifier: str = ""
        if isinstance(data_reference, NodeDataReference):
            feature_type = FeatureTypes.NODE
            entity_type = data_reference.node_type
        elif isinstance(data_reference, EdgeDataReference):
            feature_type = FeatureTypes.EDGE
            entity_type = data_reference.edge_type
            custom_identifier = str(data_reference.edge_usage_type.value)
        else:
            raise TypeError(
                f"Data reference must be of type "
                f"{NodeDataReference.__name__} or {EdgeDataReference.__name__}.  "
                f"Got {type(data_reference)}."
            )

        transformed_features_info = TransformedFeaturesInfo(
            applied_task_identifier=self.applied_task_identifier,
            feature_type=feature_type,
            entity_type=entity_type,
            custom_identifier=custom_identifier,
        )

        def __get_feature_preprocessing_job_msgs(
            is_start: bool,
        ) -> str:
            verb = "Started" if is_start else "Finished"
            return f"[{entity_type}] {verb} Dataflow job to transform {feature_type} features."

        with lock:
            logger.debug(f"[{feature_type}:{entity_type}] acquiring lock.")
            # We wait for each pipeline to start running to avoid thread-safety issues while kicking off multiple jobs.
            p = transform_utils.get_load_data_and_transform_pipeline_component(
                applied_task_identifier=self.applied_task_identifier,
                data_reference=data_reference,
                preprocessing_spec=preprocessing_spec,
                transformed_features_info=transformed_features_info,
                num_shards=int(
                    self.gbml_config_pb_wrapper.gbml_config_pb.feature_flags.get(
                        "data_preprocessor_num_shards", "0"
                    )
                ),
                custom_worker_image_uri=self.custom_worker_image_uri,
            )
            feature_transform_pipeline_result = p.run()
            logger.debug(f"[{feature_type}:{entity_type}] releasing lock.")

        logger.info(
            __get_feature_preprocessing_job_msgs(
                is_start=True,
            )
        )
        feature_transform_pipeline_result.wait_until_finish()
        logger.info(
            __get_feature_preprocessing_job_msgs(
                is_start=False,
            )
        )

        def __get_feature_dimension_for_single_data_reference(
            schema_path: Uri, feature_outputs: List[str]
        ) -> int:
            schema = tfdv.load_schema_text(schema_path.uri)
            feature_spec = tft.tf_metadata.schema_utils.schema_as_feature_spec(
                schema
            ).feature_spec
            feature_dimension = 0
            for feature in feature_spec:
                if feature in feature_outputs:
                    feature_shape = feature_spec[feature].shape
                    if len(feature_shape) == 0:
                        feature_dimension += 1
                    else:
                        feature_dimension += feature_shape[0]
            return feature_dimension

        # Find and save the feature dimension if there is any
        if preprocessing_spec.features_outputs is not None:
            transformed_features_info.feature_dim_output = __get_feature_dimension_for_single_data_reference(
                schema_path=transformed_features_info.transformed_features_schema_path,
                feature_outputs=preprocessing_spec.features_outputs,
            )

        # Carry forward the identifier, features and label outputs from the preprocessing spec.
        transformed_features_info.identifier_output = (
            preprocessing_spec.identifier_output
        )
        transformed_features_info.features_outputs = preprocessing_spec.features_outputs
        transformed_features_info.label_outputs = preprocessing_spec.labels_outputs

        if isinstance(feature_transform_pipeline_result, DataflowPipelineResult):
            pipeline_state: str = feature_transform_pipeline_result.state
            if pipeline_state != PipelineState.DONE:
                raise RuntimeError(
                    f"A dataflow pipeline potentiall failed, has state {pipeline_state}: {feature_transform_pipeline_result}"
                )

            transformed_features_info.dataflow_console_uri = (
                gigl.common.utils.dataflow.get_console_uri_from_pipeline_result(
                    pipeline_result=feature_transform_pipeline_result
                )
            )

        logger.info(f"Transformed features written to {transformed_features_info}")

        return transformed_features_info

    def __preprocess_all_data_references(
        self,
        node_ref_to_preprocessing_spec: Dict[
            NodeDataReference, NodeDataPreprocessingSpec
        ],
        edge_ref_to_preprocessing_spec: Dict[
            EdgeDataReference, EdgeDataPreprocessingSpec
        ],
    ) -> PreprocessedMetadataReferences:
        """
        Kicks off multiple jobs in parallel to apply Tensorflow Transform to all relevant node and edge data specified
        for preprocessing according to `data_preprocessor_config`.  There will be k total jobs, where
        k = (# of NodeDataReference) + (# of EdgeDataReference). Each job will return a single instance of
        TransformedFeaturesInfo, which houses information about the job's outputs, schema and associated assets.
        :param data_preprocessor_config: A concrete instantiation of DataPreprocessorConfig written by the user.
        :return: An instance of PreprocessedDataReferences which maps all references to post-TFTransform outputs.
        """

        def __build_data_reference_str(references: Iterable[DataReference]) -> str:
            ret_str = ""
            for ref in references:
                ret_str += f"\t{ref}\n"
            return ret_str

        logger.info(
            f"Node data reference to preprocessing spec has {len(node_ref_to_preprocessing_spec)} items:\n"
            f"{__build_data_reference_str(references=node_ref_to_preprocessing_spec.keys())}"
        )

        logger.info(
            f"Edge data reference to preprocessing spec has {len(edge_ref_to_preprocessing_spec)} items:\n"
            f"{__build_data_reference_str(references=edge_ref_to_preprocessing_spec.keys())}"
        )

        node_refs_and_results: Dict[NodeDataReference, TransformedFeaturesInfo] = dict()
        edge_refs_and_results: Dict[EdgeDataReference, TransformedFeaturesInfo] = dict()

        # We kick off multiple Dataflow pipelines, each of which kicks off a setup.py sdist run.
        # sdist has race-condition issues for simultaneous runs: https://github.com/pypa/setuptools/issues/1222
        # We have each thread take a lock when kicking off the pipelines to avoid this issue.
        dataflow_setup_lock = threading.Lock()

        num_dataflow_jobs = len(node_ref_to_preprocessing_spec) + len(
            edge_ref_to_preprocessing_spec
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_dataflow_jobs
        ) as executor:
            logger.info(f"Launching {num_dataflow_jobs} dataflow jobs in parallel.")
            futures: Dict[
                concurrent.futures.Future[TransformedFeaturesInfo],
                Tuple[Union[NodeDataReference, EdgeDataReference], FeatureTypes],
            ] = dict()

            data_ref_and_prep_specs: Iterable[
                Union[
                    Tuple[NodeDataReference, NodeDataPreprocessingSpec],
                    Tuple[EdgeDataReference, EdgeDataPreprocessingSpec],
                ]
            ] = chain(
                node_ref_to_preprocessing_spec.items(),
                edge_ref_to_preprocessing_spec.items(),
            )

            feature_types: Iterable[FeatureTypes] = chain(
                repeat(FeatureTypes.NODE, len(node_ref_to_preprocessing_spec)),
                repeat(FeatureTypes.EDGE, len(edge_ref_to_preprocessing_spec)),
            )

            for data_ref_and_prep_spec, feature_type in zip(
                data_ref_and_prep_specs, feature_types
            ):
                data_ref: Union[
                    NodeDataReference, EdgeDataReference
                ] = data_ref_and_prep_spec[0]
                prep_spec: Union[
                    NodeDataPreprocessingSpec, EdgeDataPreprocessingSpec
                ] = data_ref_and_prep_spec[1]

                future = executor.submit(
                    self.__preprocess_single_data_reference,
                    data_reference=data_ref,
                    preprocessing_spec=prep_spec,
                    lock=dataflow_setup_lock,
                )
                futures.update({future: (data_ref, feature_type)})

            # Collect results from node / edge jobs and error on failure.
            for future in concurrent.futures.as_completed(futures):
                data_ref, feature_type = futures[future]
                try:
                    preprocessed_features_info: TransformedFeaturesInfo = (
                        future.result()
                    )
                    if isinstance(data_ref, NodeDataReference):
                        node_refs_and_results[data_ref] = preprocessed_features_info
                    elif isinstance(data_ref, EdgeDataReference):
                        edge_refs_and_results[data_ref] = preprocessed_features_info
                except Exception as e:
                    logger.exception(
                        f"[{feature_type}: {(data_ref)}] preprocessing job failed due to a raised exception: {e}"
                    )
                    raise e

        return PreprocessedMetadataReferences(
            node_data=node_refs_and_results, edge_data=edge_refs_and_results
        )

    def _generate_edge_metadata_info_pb(
        self,
        transformed_features_info: TransformedFeaturesInfo,
        enumerated_edge_metadata: EnumeratorEdgeTypeMetadata,
    ) -> preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo:
        return preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(
            tfrecord_uri_prefix=transformed_features_info.transformed_features_file_prefix.uri,
            schema_uri=transformed_features_info.transformed_features_schema_path.uri,
            feature_keys=transformed_features_info.features_outputs,
            label_keys=transformed_features_info.label_outputs,
            enumerated_edge_data_bq_table=enumerated_edge_metadata.enumerated_edge_data_reference.reference_uri,
            feature_dim=transformed_features_info.feature_dim_output,
            transform_fn_assets_uri=transformed_features_info.transformed_features_transform_fn_assets_path.uri,
        )

    def generate_preprocessed_metadata_pb(
        self,
        preprocessed_metadata_references: PreprocessedMetadataReferences,
        enumerator_node_type_metadata: List[EnumeratorNodeTypeMetadata],
        enumerator_edge_type_metadata: List[EnumeratorEdgeTypeMetadata],
    ) -> preprocessed_metadata_pb2.PreprocessedMetadata:
        preprocessed_metadata_pb = preprocessed_metadata_pb2.PreprocessedMetadata()

        enumerator_node_type_metadata_map: Dict[
            NodeType, EnumeratorNodeTypeMetadata
        ] = {
            node_type_metadata.enumerated_node_data_reference.node_type: node_type_metadata
            for node_type_metadata in enumerator_node_type_metadata
        }

        # Populate all node data.
        logger.info("Populating preprocessed metadata with node data.")
        node_info: Tuple[NodeDataReference, TransformedFeaturesInfo]
        for node_info in preprocessed_metadata_references.node_data.items():
            node_data_ref: NodeDataReference
            node_transformed_features_info: TransformedFeaturesInfo
            node_data_ref, node_transformed_features_info = node_info

            node_type: NodeType = node_data_ref.node_type
            enumerated_node_metadata = enumerator_node_type_metadata_map[node_type]

            logger.info(
                f"Adding to preprocessed metadata pb: [{node_data_ref}: {node_transformed_features_info}]"
            )

            condensed_node_type: CondensedNodeType = self.gbml_config_pb_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
                node_type
            ]
            node_identifier_output = node_transformed_features_info.identifier_output
            assert isinstance(
                node_identifier_output, NodeOutputIdentifier
            ), f"Identifier output should be of class {NodeOutputIdentifier.__name__}."

            features_outputs = node_transformed_features_info.features_outputs
            label_outputs = node_transformed_features_info.label_outputs
            feature_dim_output = node_transformed_features_info.feature_dim_output

            node_metadata_output_pb = preprocessed_metadata_pb2.PreprocessedMetadata.NodeMetadataOutput(
                tfrecord_uri_prefix=node_transformed_features_info.transformed_features_file_prefix.uri,
                schema_uri=node_transformed_features_info.transformed_features_schema_path.uri,
                node_id_key=str(node_identifier_output),
                feature_keys=features_outputs,
                label_keys=label_outputs,
                enumerated_node_ids_bq_table=enumerated_node_metadata.bq_unique_node_ids_enumerated_table_name,
                enumerated_node_data_bq_table=enumerated_node_metadata.enumerated_node_data_reference.reference_uri,
                feature_dim=feature_dim_output,
                transform_fn_assets_uri=node_transformed_features_info.transformed_features_transform_fn_assets_path.uri,
            )
            preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata[
                int(condensed_node_type)
            ].CopyFrom(node_metadata_output_pb)

        # Populate all edge data.
        logger.info("Populating preprocessed metadata with edge data.")

        enumerator_edge_type_metadata_map: Dict[
            EdgeType, Dict[EdgeUsageType, EnumeratorEdgeTypeMetadata]
        ] = defaultdict(dict)
        for edge_type_metadata in enumerator_edge_type_metadata:
            enumerator_edge_type_metadata_map[
                edge_type_metadata.enumerated_edge_data_reference.edge_type
            ][
                edge_type_metadata.enumerated_edge_data_reference.edge_usage_type
            ] = edge_type_metadata

        preprocessed_metadata_references_map: Dict[
            EdgeType, Dict[EdgeUsageType, TransformedFeaturesInfo]
        ] = defaultdict(dict)
        edge_info: Tuple[EdgeDataReference, TransformedFeaturesInfo]
        for edge_info in preprocessed_metadata_references.edge_data.items():
            edge_data_ref: EdgeDataReference
            edge_transformed_features_info: TransformedFeaturesInfo
            edge_data_ref, edge_transformed_features_info = edge_info
            preprocessed_metadata_references_map[edge_data_ref.edge_type][
                edge_data_ref.edge_usage_type
            ] = edge_transformed_features_info

        edge_type: EdgeType
        edge_transformed_features_info_map: Dict[EdgeUsageType, TransformedFeaturesInfo]
        for (
            edge_type,
            edge_transformed_features_info_map,
        ) in preprocessed_metadata_references_map.items():
            positive_transformed_features_info: Optional[
                TransformedFeaturesInfo
            ] = edge_transformed_features_info_map.get(EdgeUsageType.POSITIVE, None)
            negative_transformed_features_info: Optional[
                TransformedFeaturesInfo
            ] = edge_transformed_features_info_map.get(EdgeUsageType.NEGATIVE, None)
            main_transformed_features_info: Optional[
                TransformedFeaturesInfo
            ] = edge_transformed_features_info_map.get(EdgeUsageType.MAIN, None)
            assert (
                main_transformed_features_info is not None
            ), f"Main edge data must be present for edge type {edge_type}."

            positive_enumerated_edge_metadata: Optional[
                EnumeratorEdgeTypeMetadata
            ] = None
            negative_enumerated_edge_metadata: Optional[
                EnumeratorEdgeTypeMetadata
            ] = None
            main_enumerated_edge_metadata: Optional[EnumeratorEdgeTypeMetadata] = None
            if positive_transformed_features_info:
                positive_enumerated_edge_metadata = enumerator_edge_type_metadata_map[
                    edge_type
                ][EdgeUsageType.POSITIVE]
            if negative_transformed_features_info:
                negative_enumerated_edge_metadata = enumerator_edge_type_metadata_map[
                    edge_type
                ][EdgeUsageType.NEGATIVE]
            main_enumerated_edge_metadata = enumerator_edge_type_metadata_map[
                edge_type
            ][EdgeUsageType.MAIN]

            condensed_edge_type: CondensedEdgeType = self.gbml_config_pb_wrapper.graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[
                edge_type
            ]
            assert isinstance(
                main_transformed_features_info.identifier_output, EdgeOutputIdentifier
            ), f"Identifier output should be of class {EdgeOutputIdentifier.__name__}."
            edge_output_identifier: EdgeOutputIdentifier = (
                main_transformed_features_info.identifier_output
            )
            assert isinstance(
                edge_output_identifier, EdgeOutputIdentifier
            ), f"Identifier output should be of class {EdgeOutputIdentifier.__name__}."

            positive_edge_metadata_info_pb: Optional[
                preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo
            ] = None
            negative_edge_metadata_info_pb: Optional[
                preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo
            ] = None
            main_edge_metadata_info_pb: (
                preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo
            ) = self._generate_edge_metadata_info_pb(
                transformed_features_info=main_transformed_features_info,
                enumerated_edge_metadata=main_enumerated_edge_metadata,
            )

            if positive_transformed_features_info:
                assert isinstance(
                    positive_enumerated_edge_metadata, EnumeratorEdgeTypeMetadata
                )
                positive_edge_metadata_info_pb = self._generate_edge_metadata_info_pb(
                    transformed_features_info=positive_transformed_features_info,
                    enumerated_edge_metadata=positive_enumerated_edge_metadata,
                )

            if negative_transformed_features_info:
                assert isinstance(
                    negative_enumerated_edge_metadata, EnumeratorEdgeTypeMetadata
                )
                negative_edge_metadata_info_pb = self._generate_edge_metadata_info_pb(
                    transformed_features_info=negative_transformed_features_info,
                    enumerated_edge_metadata=negative_enumerated_edge_metadata,
                )

            edge_metadata_output_pb = (
                preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataOutput(
                    src_node_id_key=str(edge_output_identifier.src_node),
                    dst_node_id_key=str(edge_output_identifier.dst_node),
                    main_edge_info=main_edge_metadata_info_pb,
                    positive_edge_info=positive_edge_metadata_info_pb,
                    negative_edge_info=negative_edge_metadata_info_pb,
                )
            )
            preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata[
                int(condensed_edge_type)
            ].CopyFrom(edge_metadata_output_pb)

        return preprocessed_metadata_pb

    def __validate_data_references_map_to_graph_metadata(self) -> None:
        """
        Validates that all node and edge data references reference node and edge types that are present in the graph.
        """

        node_data_refs = (
            self.data_preprocessor_config.get_nodes_preprocessing_spec().keys()
        )
        edge_data_refs = (
            self.data_preprocessor_config.get_edges_preprocessing_spec().keys()
        )
        for node_data_ref in node_data_refs:
            if (
                node_data_ref.node_type
                not in self.gbml_config_pb_wrapper.graph_metadata_pb_wrapper.node_types
            ):
                raise ValueError(
                    f"Node type {node_data_ref.node_type} from {node_data_ref} not found in graph metadata."
                )
        for edge_data_ref in edge_data_refs:
            if (
                edge_data_ref.edge_type
                not in self.gbml_config_pb_wrapper.graph_metadata_pb_wrapper.edge_types
            ):
                raise ValueError(
                    f"Edge type {edge_data_ref.edge_type} from {edge_data_ref} not found in graph metadata."
                )

    def __patch_preprocessing_specs(
        self,
        node_data_reference_to_preprocessing_spec: Dict[
            NodeDataReference, NodeDataPreprocessingSpec
        ],
        edge_data_reference_to_preprocessing_spec: Dict[
            EdgeDataReference, EdgeDataPreprocessingSpec
        ],
        enumerator_node_type_metadata: List[EnumeratorNodeTypeMetadata],
        enumerator_edge_type_metadata: List[EnumeratorEdgeTypeMetadata],
    ) -> Tuple[
        Dict[NodeDataReference, NodeDataPreprocessingSpec],
        Dict[EdgeDataReference, EdgeDataPreprocessingSpec],
    ]:
        """
        Patches the preprocessing specs for enumerated node and edge data references.
        This is necessary because the enumerated node and edge data references have different identifiers than the original
        node and edge data references.  We need to update the preprocessing specs to use the enumerated identifiers.

        Args:
            enumerator_node_type_metadata: List of enumerated node type metadata.
            enumerator_edge_type_metadata: List of enumerated edge type metadata.
        Returns:
            Tuple of dictionaries mapping enumerated node and edge data references to their preprocessing specs.
        """

        # First, we patch the node data references.
        enumerated_node_refs_to_preprocessing_specs: Dict[
            NodeDataReference, NodeDataPreprocessingSpec
        ] = {}

        def feature_spec_fn(
            feature_spec: FeatureSpecDict,
        ) -> Callable[[], FeatureSpecDict]:
            # We do this in order to bind the value of feature_spec to the returned function.
            # This is a common pattern in Python to create a closure.
            def inner() -> FeatureSpecDict:
                return feature_spec

            return inner

        for enumerated_node_metadata in enumerator_node_type_metadata:
            input_node_preprocessing_spec = node_data_reference_to_preprocessing_spec[
                enumerated_node_metadata.input_node_data_reference
            ]

            feature_spec = input_node_preprocessing_spec.feature_spec_fn()
            assert (
                input_node_preprocessing_spec.identifier_output in feature_spec
            ), f"identifier_output: {input_node_preprocessing_spec.identifier_output} must be in feature_spec: {feature_spec}"

            # We expect the user to give us the actual feature spec for the node id; i.e. it might be string.
            # By the end of this function, we will finish enumerated the node id to an integer; thus we update
            # the feature spec respectively.
            feature_spec[
                input_node_preprocessing_spec.identifier_output
            ] = tf.io.FixedLenFeature(shape=[], dtype=DEFAULT_TF_INT_DTYPE)

            enumerated_node_data_preprocessing_spec = NodeDataPreprocessingSpec(
                feature_spec_fn=feature_spec_fn(feature_spec),
                preprocessing_fn=input_node_preprocessing_spec.preprocessing_fn,
                identifier_output=input_node_preprocessing_spec.identifier_output,
                pretrained_tft_model_uri=input_node_preprocessing_spec.pretrained_tft_model_uri,
                features_outputs=input_node_preprocessing_spec.features_outputs,
                labels_outputs=input_node_preprocessing_spec.labels_outputs,
            )
            enumerated_node_refs_to_preprocessing_specs[
                enumerated_node_metadata.enumerated_node_data_reference
            ] = enumerated_node_data_preprocessing_spec

        # Now we do the same for edges.
        enumerated_edge_refs_to_preprocessing_specs: Dict[
            EdgeDataReference, EdgeDataPreprocessingSpec
        ] = {}
        for enumerated_edge_metadata in enumerator_edge_type_metadata:
            input_edge_preprocessing_spec = edge_data_reference_to_preprocessing_spec[
                enumerated_edge_metadata.input_edge_data_reference
            ]

            feature_spec = input_edge_preprocessing_spec.feature_spec_fn()
            assert (
                input_edge_preprocessing_spec.identifier_output.src_node in feature_spec
            ), f"identifier_output: {input_edge_preprocessing_spec.identifier_output.src_node} must be in feature_spec: {feature_spec}"
            assert (
                input_edge_preprocessing_spec.identifier_output.dst_node in feature_spec
            ), f"identifier_output: {input_edge_preprocessing_spec.identifier_output.dst_node} must be in feature_spec: {feature_spec}"

            # We expect the user to give us the actual feature spec for the node id; i.e. it might be string.
            # By the end of this function, we will finish enumerated the node id to an integer; thus we update
            # the feature spec respectively.
            feature_spec[
                input_edge_preprocessing_spec.identifier_output.src_node
            ] = tf.io.FixedLenFeature(shape=[], dtype=DEFAULT_TF_INT_DTYPE)
            feature_spec[
                input_edge_preprocessing_spec.identifier_output.dst_node
            ] = tf.io.FixedLenFeature(shape=[], dtype=DEFAULT_TF_INT_DTYPE)

            enumerated_edge_data_preprocessing_spec = EdgeDataPreprocessingSpec(
                feature_spec_fn=feature_spec_fn(feature_spec),
                preprocessing_fn=input_edge_preprocessing_spec.preprocessing_fn,
                identifier_output=input_edge_preprocessing_spec.identifier_output,
                pretrained_tft_model_uri=input_edge_preprocessing_spec.pretrained_tft_model_uri,
                features_outputs=input_edge_preprocessing_spec.features_outputs,
                labels_outputs=input_edge_preprocessing_spec.labels_outputs,
            )
            enumerated_edge_refs_to_preprocessing_specs[
                enumerated_edge_metadata.enumerated_edge_data_reference
            ] = enumerated_edge_data_preprocessing_spec

        return (
            enumerated_node_refs_to_preprocessing_specs,
            enumerated_edge_refs_to_preprocessing_specs,
        )

    def __run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        custom_worker_image_uri: Optional[str] = None,
    ) -> Uri:
        # Prepare environment
        self.__prepare_env(
            applied_task_identifier=applied_task_identifier,
            task_config_uri=task_config_uri,
            custom_worker_image_uri=custom_worker_image_uri,
        )

        # Any custom preparation work before running the pipeline
        self.data_preprocessor_config.prepare_for_pipeline(
            applied_task_identifier=applied_task_identifier
        )

        # Validate the node and edge data references.
        self.__validate_data_references_map_to_graph_metadata()

        bq_gcp_project = get_resource_config().project
        logger.info(f"Using implicit GCP project {bq_gcp_project} for BigQuery.")

        # Update the node and edge data references to include identifiers. In current configuration setup,
        # these identifiers are piped in from the DataPreprocessorConfig.
        node_refs_to_specs: Dict[NodeDataReference, NodeDataPreprocessingSpec] = {}
        for (
            node_data_reference,
            node_data_preprocessing_spec,
        ) in self.data_preprocessor_config.get_nodes_preprocessing_spec().items():
            assert isinstance(
                node_data_reference, BigqueryNodeDataReference
            ), f"Only {BigqueryNodeDataReference.__name__} references are currently supported."
            node_data_ref_with_identifier = BigqueryNodeDataReference(
                reference_uri=node_data_reference.reference_uri,
                node_type=node_data_reference.node_type,
                identifier=node_data_preprocessing_spec.identifier_output,
            )
            node_refs_to_specs[
                node_data_ref_with_identifier
            ] = node_data_preprocessing_spec

        edge_refs_to_specs: Dict[EdgeDataReference, EdgeDataPreprocessingSpec] = {}
        for (
            edge_data_reference,
            edge_data_preprocessing_spec,
        ) in self.data_preprocessor_config.get_edges_preprocessing_spec().items():
            assert isinstance(
                edge_data_reference, BigqueryEdgeDataReference
            ), f"Only {BigqueryEdgeDataReference.__name__} references are currently supported."
            edge_data_ref_with_identifier = BigqueryEdgeDataReference(
                reference_uri=edge_data_reference.reference_uri,
                edge_type=edge_data_reference.edge_type,
                edge_usage_type=edge_data_reference.edge_usage_type,
                src_identifier=edge_data_preprocessing_spec.identifier_output.src_node,
                dst_identifier=edge_data_preprocessing_spec.identifier_output.dst_node,
            )
            edge_refs_to_specs[
                edge_data_ref_with_identifier
            ] = edge_data_preprocessing_spec

        # Enumerate all graph data.
        enumerator = Enumerator()
        enumerator_results: Tuple[
            List[EnumeratorNodeTypeMetadata], List[EnumeratorEdgeTypeMetadata]
        ] = enumerator.run(
            applied_task_identifier=self.applied_task_identifier,
            node_data_references=list(node_refs_to_specs.keys()),
            edge_data_references=list(edge_refs_to_specs.keys()),
            gcp_project=bq_gcp_project,
        )

        (
            enumerator_node_type_metadata,
            enumerator_edge_type_metadata,
        ) = enumerator_results

        # Now that we've enumerated all the node and edge data, we need to update
        # the preprocessing specs to use the enumerated node and edge data references.
        (
            enumerated_node_refs_to_preprocessing_specs,
            enumerated_edge_refs_to_preprocessing_specs,
        ) = self.__patch_preprocessing_specs(
            node_data_reference_to_preprocessing_spec=node_refs_to_specs,
            edge_data_reference_to_preprocessing_spec=edge_refs_to_specs,
            enumerator_node_type_metadata=enumerator_node_type_metadata,
            enumerator_edge_type_metadata=enumerator_edge_type_metadata,
        )

        # Validating Enumerated Edge Tables that were generated
        # We perform this check on the enumerated table, meaning that for nodes that exist in the
        # edge table that are not in the node table, the node will be enumerated to NULL.
        # Thus having a check for dangling edges i.e. checking if there is any NULL node id,
        # in turn is just cheking whether or not the source data provided has edges with nodes
        # that are not present in the node data.
        logger.info(
            "Validating that all enumerated edge data references have no dangling edges."
        )
        resource_labels = get_resource_config().get_resource_labels(
            component=GiGLComponents.DataPreprocessor
        )
        for enumerated_edge_metadata in enumerator_edge_type_metadata:
            src_node_column_name = (
                enumerated_edge_metadata.enumerated_edge_data_reference.src_identifier
            )
            dst_node_column_name = (
                enumerated_edge_metadata.enumerated_edge_data_reference.dst_identifier
            )
            assert (src_node_column_name is not None) and (
                dst_node_column_name is not None
            ), f"Missing src/dst dentifiers in enumerated edge data reference: {enumerated_edge_metadata.enumerated_edge_data_reference}"
            edge_table = (
                enumerated_edge_metadata.enumerated_edge_data_reference.reference_uri
            )

            has_dangling_edges = BQGraphValidator.does_edge_table_have_dangling_edges(
                edge_table=edge_table,
                src_node_column_name=src_node_column_name,
                dst_node_column_name=dst_node_column_name,
                query_labels=resource_labels,
                bq_gcp_project=bq_gcp_project,
            )
            if has_dangling_edges:
                raise ValueError(
                    f"""
                    ERROR: The enumerated edge table {edge_table} has dangling edges. Meaning that at least one
                    edge exists where either src_node ({src_node_column_name}) and/or
                    dst_node ({dst_node_column_name}) is null. This is usually because of input data having
                    edges containing nodes which are not present in the input node data. Please look into the
                    input data and fix the issue.
                """
                )

        # Run Dataflow jobs to transform data references as per DataPreprocessorConfig.
        preprocessed_metadata_references: PreprocessedMetadataReferences = self.__preprocess_all_data_references(
            node_ref_to_preprocessing_spec=enumerated_node_refs_to_preprocessing_specs,
            edge_ref_to_preprocessing_spec=enumerated_edge_refs_to_preprocessing_specs,
        )

        logger.info("All preprocessed NODE results:\n")
        for (
            node_data_ref,
            node_transformed_features_info,
        ) in preprocessed_metadata_references.node_data.items():
            logger.info(f"\n{node_data_ref}\n" f"\t{node_transformed_features_info}\n")

        logger.info("All preprocessed EDGE results:\n")
        for (
            edge_data_ref,
            edge_transformed_features_info,
        ) in preprocessed_metadata_references.edge_data.items():
            logger.info(f"\n{edge_data_ref}\n" f"\t{edge_transformed_features_info}\n")

        # Generate PreprocessedMetadata result proto for other components to read.
        preprocessed_metadata_pb: preprocessed_metadata_pb2.PreprocessedMetadata = (
            self.generate_preprocessed_metadata_pb(
                preprocessed_metadata_references=preprocessed_metadata_references,
                enumerator_node_type_metadata=enumerator_node_type_metadata,
                enumerator_edge_type_metadata=enumerator_edge_type_metadata,
            )
        )
        preprocessed_metadata_output_uri = UriFactory.create_uri(
            self.gbml_config_pb_wrapper.shared_config.preprocessed_metadata_uri
        )
        self.__proto_utils.write_proto_to_yaml(
            proto=preprocessed_metadata_pb, uri=preprocessed_metadata_output_uri
        )
        logger.info(
            f"{preprocessed_metadata_pb.__class__.__name__} written to {preprocessed_metadata_output_uri.uri}"
        )

        # Cleanup environment.
        self.__cleanup_env()

        return preprocessed_metadata_output_uri

    @flushes_metrics(get_metrics_service_instance_fn=get_metrics_service_instance)
    @profileit(
        metric_name=TIMER_PREPROCESSOR_S,
        get_metrics_service_instance_fn=get_metrics_service_instance,
    )
    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        custom_worker_image_uri: Optional[str] = None,
    ) -> Uri:
        """
        Runs the DataPreprocessor, given a config file.
        :param applied_task_identifier:
        :param config_uri: YAML file representing GbmlConfig proto.
        :param resource_config_uri: YAML file representing GiGLResourceConfig proto
        :param custom_worker_image_uri: Optional URI for a custom Dataflow worker image.
        :return:
        """
        resource_config = get_resource_config(resource_config_uri=resource_config_uri)
        try:
            preprocessed_metadata_output_uri = self.__run(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                custom_worker_image_uri=custom_worker_image_uri,
            )
            return preprocessed_metadata_output_uri
        except Exception as e:
            logger.error(
                "DataPreprocessor failed due to a raised exception, which will follow"
            )
            logger.error(e)
            logger.info("Cleaning up DataPreprocessor environment...")
            self.__cleanup_env()
            sys.exit(f"System will now exit: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to preprocess node and edge data from an input graph"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="Unique identifier for the job name",
        required=True,
    )
    parser.add_argument(
        "--task_config_uri",
        type=str,
        help="Gbml config uri",
        required=True,
    )
    parser.add_argument(
        "--resource_config_uri",
        type=str,
        help="Runtime argument for resource and env specifications of each component",
        required=True,
    )
    parser.add_argument(
        "--custom_worker_image_uri",
        type=str,
        help="Docker image to use for the worker harness in dataflow",
        required=False,
    )
    args = parser.parse_args()

    ati = AppliedTaskIdentifier(args.job_name)
    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    resource_config_uri = UriFactory.create_uri(args.resource_config_uri)
    custom_worker_image_uri = args.custom_worker_image_uri

    initialize_metrics(task_config_uri=task_config_uri, service_name=args.job_name)

    data_preprocessor = DataPreprocessor()
    data_preprocessor.run(
        applied_task_identifier=ati,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        custom_worker_image_uri=custom_worker_image_uri,
    )
