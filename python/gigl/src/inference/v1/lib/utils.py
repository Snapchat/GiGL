from typing import List, Optional, Tuple

import apache_beam as beam
import tensorflow as tf
from apache_beam.pvalue import PCollection

from gigl.common import GcsUri, Uri
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.dataflow import init_beam_pipeline_options
from gigl.src.data_preprocessor.lib.enumerate.queries import (
    DEFAULT_ENUMERATED_NODE_ID_FIELD,
    DEFAULT_ORIGINAL_NODE_ID_FIELD,
)
from gigl.src.inference.v1.lib.base_inference_blueprint import (
    EMBEDDING_TAGGED_OUTPUT_KEY,
    PREDICTION_TAGGED_OUTPUT_KEY,
    BaseInferenceBlueprint,
)
from gigl.src.inference.v1.lib.inference_output_schema import (
    DEFAULT_EMBEDDING_FIELD,
    DEFAULT_NODE_ID_FIELD,
    DEFAULT_PREDICTION_FIELD,
)
from gigl.src.inference.v1.lib.transforms.batch_generator import BatchProcessorDoFn
from snapchat.research.gbml.gigl_resource_config_pb2 import DataflowResourceConfig

logger = Logger()

# TODO(svij-sc) adopt dynamic batching
DEFAULT_BATCH_SIZE = 3000


class UnenumerateAssets(beam.PTransform):
    def __init__(self, tagged_output_key: str):
        if tagged_output_key == PREDICTION_TAGGED_OUTPUT_KEY:
            self.field = DEFAULT_PREDICTION_FIELD
        elif tagged_output_key == EMBEDDING_TAGGED_OUTPUT_KEY:
            self.field = DEFAULT_EMBEDDING_FIELD
        else:
            raise NotImplementedError(
                "Only embedding or prediction outputs are supported"
            )

    def expand(self, pcolls: Tuple[PCollection, PCollection]) -> PCollection:
        """
        Performs unenumeration on two PCollections through a join between the two collections.
        The first PCollection should contain the DEFAULT_NODE_ID_FIELD and either DEFAULT_PREDICTION_FIELD or DEFAULT_EMBEDDING_FIELD columns.
        The second PCollection should contain the DEFAULT_ENUMERATED_NODE_ID_FIELD and DEFAULT_ORIGINAL_NODE_ID_FIELD columns.
        The two pcollections will be joined by the values in the DEFAULT_NODE_ID_FIELD and DEFAULT_ENUMERATED_NODE_ID_FIELD columns.
        """
        output, mapping = pcolls
        enumerated_assets = output | "Format outputs" >> beam.Map(
            lambda row: (
                row[DEFAULT_NODE_ID_FIELD],
                row[self.field],
            )
        )
        unenumerated_assets = (
            {"enumerated_assets": enumerated_assets, "mapping": mapping}
            | "Perform join" >> beam.CoGroupByKey()
            # CoGroupByKey joins by the first element of each tuple, in this case mapping.int_id and predictions.node_id
            | "Extract prediction and Format"
            >> beam.Map(
                lambda kv: {
                    DEFAULT_NODE_ID_FIELD: kv[1]["mapping"][0],
                    self.field: kv[1]["enumerated_assets"][0],
                }
            )
        )
        return unenumerated_assets


def get_inferencer_pipeline_component_for_single_node_type(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    inference_blueprint: BaseInferenceBlueprint,
    applied_task_identifier: AppliedTaskIdentifier,
    custom_worker_image_uri: Optional[str],
    node_type: NodeType,
    uri_prefix_list: List[Uri],
    temp_predictions_gcs_path: Optional[GcsUri],
    temp_embeddings_gcs_path: Optional[GcsUri],
) -> beam.Pipeline:
    """
    Gets the beam pipeline for running the inference dataflow job
    Args:
        gbml_config_pb_wrapper (GbmlConfigPbWrapper): GBML config wrapper for this inference run
        inference_blueprint (BaseInferenceBlueprint): Blueprint for running and saving inference for GBML pipelines
        applied_task_identifier (AppliedTaskIdentifier): Identifier for the GiGL job
        custom_worker_image_uri (Optional[str]): Uri to custom worker image
        node_type (NodeType): Node type being inferred
        uri_prefix_list (List[Uri]): List of prefixes for running inference for given node type
        temp_predictions_gcs_path (Optional[GcsUri]): Gcs uri for writing temp predictions
        temp_embeddings_gcs_path (Optional[GcsUri]): Gcs uri for writing temp embeddings
    Returns:
        pipeline (beam.Pipeline): Dataflow pipeline for running inference
    """
    # Launching one beam pipeline per node type
    inferencer_config = get_resource_config().inferencer_config
    assert isinstance(
        inferencer_config, DataflowResourceConfig
    ), f"Only Dataflow is supported for v1 inference, got: {type(inferencer_config)}"
    condensed_node_type_to_preprocessed_metadata = (
        gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata
    )
    batch_size = (
        gbml_config_pb_wrapper.inferencer_config.inference_batch_size
        or DEFAULT_BATCH_SIZE
    )
    options = init_beam_pipeline_options(
        applied_task_identifier=applied_task_identifier,
        job_name_suffix=f"{node_type}_inference",
        component=GiGLComponents.Inferencer,
        num_workers=inferencer_config.num_workers,
        max_num_workers=inferencer_config.max_num_workers,
        machine_type=inferencer_config.machine_type,
        disk_size_gb=inferencer_config.disk_size_gb,
        resource_config=get_resource_config().get_resource_config_uri,
        custom_worker_image_uri=custom_worker_image_uri,
    )
    condensed_node_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
        NodeType(node_type)
    ]
    mapping_table = condensed_node_type_to_preprocessed_metadata[
        condensed_node_type
    ].enumerated_node_ids_bq_table
    should_run_unenumeration = bool(mapping_table)
    pipeline = beam.Pipeline(options=options)
    record_pcolls = []
    for uri_prefix in uri_prefix_list:
        tfrecord_glob_str = f"{uri_prefix.uri}*.tfrecord"
        files = tf.io.gfile.glob(tfrecord_glob_str)
        if not files:
            logger.warning(
                f"Found no TFRecord files at {uri_prefix.uri} for node type {node_type}"
            )
            continue
        pcol = (
            pipeline
            | f"Read TFRecords from uri prefix {uri_prefix}"
            >> beam.io.ReadFromTFRecord(
                file_pattern=tfrecord_glob_str,
                coder=inference_blueprint.get_tf_record_coder(),
            )
        )
        record_pcolls.append(pcol)
    outputs = (
        record_pcolls
        | f"Flatten read pcollections" >> beam.Flatten()
        | "Batch Elements"
        >> beam.BatchElements(
            min_batch_size=batch_size,
            max_batch_size=batch_size,
            target_batch_duration_secs_including_fixed_cost=1,
        )
        | "Generate Batches"
        >> beam.ParDo(
            BatchProcessorDoFn(
                batch_generator_fn=inference_blueprint.get_batch_generator_fn(),
            )
        )
        | "Inference"
        >> beam.ParDo(inference_blueprint.get_inferer()).with_outputs(
            PREDICTION_TAGGED_OUTPUT_KEY, EMBEDDING_TAGGED_OUTPUT_KEY
        )
    )
    if should_run_unenumeration:
        mapping = (
            pipeline
            | "Read mapping"
            >> beam.io.gcp.bigquery.ReadFromBigQuery(table=mapping_table)
            | "Map mapping field for node type"
            >> beam.Map(
                lambda row: (
                    row[DEFAULT_ENUMERATED_NODE_ID_FIELD],
                    row[DEFAULT_ORIGINAL_NODE_ID_FIELD],
                )
            )
        )
        if temp_predictions_gcs_path is not None:
            predictions = (
                outputs[PREDICTION_TAGGED_OUTPUT_KEY],
                mapping,
            ) | "Unenumerate Predictions" >> UnenumerateAssets(
                tagged_output_key=PREDICTION_TAGGED_OUTPUT_KEY
            )
        if temp_embeddings_gcs_path is not None:
            embeddings = (
                outputs[EMBEDDING_TAGGED_OUTPUT_KEY],
                mapping,
            ) | "Unenumerate Embeddings" >> UnenumerateAssets(
                tagged_output_key=EMBEDDING_TAGGED_OUTPUT_KEY
            )
    else:
        logger.info(
            f"Skipping un-enumeration for node type {node_type} since no mapping table exists"
        )
        predictions = outputs[PREDICTION_TAGGED_OUTPUT_KEY]
        embeddings = outputs[EMBEDDING_TAGGED_OUTPUT_KEY]
    if temp_predictions_gcs_path is not None:
        logger.info(
            f"Writing node type {node_type} temp predictions to gcs path {temp_predictions_gcs_path.uri}"
        )
        (
            predictions
            | "Write temp predictions to gcs"
            >> beam.io.WriteToText(
                file_path_prefix=temp_predictions_gcs_path.uri,
                file_name_suffix=".json",
            )
        )
    if temp_embeddings_gcs_path is not None:
        logger.info(
            f"Writing node type {node_type} temp embeddings to gcs path {temp_embeddings_gcs_path.uri}"
        )
        (
            embeddings
            | "Write temp embeddings to gcs"
            >> beam.io.WriteToText(
                file_path_prefix=temp_embeddings_gcs_path.uri,
                file_name_suffix=".json",
            )
        )
    return pipeline
