import json
import tempfile
from functools import partial
from typing import List, Tuple

import tensorflow as tf
from google.cloud import bigquery

import gigl.src.common.utils.model as model_utils
from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.os_utils import import_obj
from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.graph_builder.pyg_graph_builder import PygGraphBuilder
from gigl.src.common.translators.training_samples_protos_translator import (
    RootedNodeNeighborhoodSample,
    SupervisedNodeClassificationSample,
)
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.common.utils.bq import BqUtils
from gigl.src.inference.v1.lib.base_inferencer import BaseInferencer, InferBatchResults
from gigl.src.inference.v1.lib.inference_output_schema import (
    DEFAULT_EMBEDDING_FIELD,
    DEFAULT_EMBEDDINGS_TABLE_SCHEMA,
    DEFAULT_NODE_ID_FIELD,
    DEFAULT_PREDICTION_FIELD,
    DEFAULT_PREDICTIONS_TABLE_SCHEMA,
)
from gigl.src.training.v1.lib.data_loaders.rooted_node_neighborhood_data_loader import (
    RootedNodeNeighborhoodBatch,
)
from gigl.src.training.v1.lib.data_loaders.supervised_node_classification_data_loader import (
    SupervisedNodeClassificationBatch,
)
from snapchat.research.gbml import (
    flattened_graph_metadata_pb2,
    gbml_config_pb2,
    training_samples_schema_pb2,
)

logger = Logger()


def _initialize_inferencer_with_gbml_config_pb(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> Tuple[BaseInferencer, GbmlConfigPbWrapper]:
    inferencer_cls = import_obj(gbml_config_pb.inferencer_config.inferencer_cls_path)
    kwargs = dict(gbml_config_pb.inferencer_config.inferencer_args)
    inferencer = inferencer_cls(**kwargs)

    gbml_config_pb_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)
    model_save_path_uri = UriFactory.create_uri(
        gbml_config_pb.shared_config.trained_model_metadata.trained_model_uri
    )
    logger.info(
        f"Loading model state dict from: {model_save_path_uri}, for inferencer: {inferencer}"
    )
    model_state_dict = model_utils.load_state_dict_from_uri(
        load_from_uri=model_save_path_uri
    )
    inferencer.init_model(
        gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        state_dict=model_state_dict,
    )

    return (
        inferencer,
        gbml_config_pb_wrapper,
    )


def infer_model(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
):
    (
        inferencer,
        gbml_config_pb_wrapper,
    ) = _initialize_inferencer_with_gbml_config_pb(gbml_config_pb=gbml_config_pb)

    task_metadata_pb_wrapper = gbml_config_pb_wrapper.task_metadata_pb_wrapper

    if task_metadata_pb_wrapper.task_metadata_type == TaskMetadataType.NODE_BASED_TASK:
        _infer_supervised_node_classification_model(
            inferencer=inferencer,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )
    elif (
        task_metadata_pb_wrapper.task_metadata_type
        == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
    ):
        _infer_node_anchor_based_link_prediction_model(
            inferencer=inferencer,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )
    else:
        raise NotImplementedError


def _infer_supervised_node_classification_model(
    inferencer: BaseInferencer,
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
):
    builder = PygGraphBuilder()

    def _collate_node_classification_batch(
        elements: List[SupervisedNodeClassificationSample],
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
        builder: GraphBuilder,
    ) -> SupervisedNodeClassificationBatch:
        batch = (
            SupervisedNodeClassificationBatch.collate_pyg_node_classification_minibatch(
                builder=builder,
                graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
                preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
                samples=elements,
            )
        )
        return batch

    translator = partial(
        SupervisedNodeClassificationBatch.preprocess_node_classification_sample_fn,
        builder=builder,
        gbml_config_pb_wrapper=gbml_config_pb_wrapper,
    )

    assert isinstance(
        gbml_config_pb_wrapper.flattened_graph_metadata_pb_wrapper.output_metadata,
        flattened_graph_metadata_pb2.SupervisedNodeClassificationOutput,
    ), f"Flattened graph metadata output of wrong type: expected {flattened_graph_metadata_pb2.SupervisedNodeClassificationOutput.__name__}"

    supervised_node_classification_output = (
        gbml_config_pb_wrapper.flattened_graph_metadata_pb_wrapper.output_metadata
    )
    labeled_tfrecord_files = tf.io.gfile.glob(
        f"{supervised_node_classification_output.labeled_tfrecord_uri_prefix}*"
    )
    unlabeled_tfrecord_files = tf.io.gfile.glob(
        f"{supervised_node_classification_output.unlabeled_tfrecord_uri_prefix}*"
    )
    all_tfrecord_files = labeled_tfrecord_files + unlabeled_tfrecord_files

    ds_iter = tf.data.TFRecordDataset(filenames=all_tfrecord_files).as_numpy_iterator()

    emb_tfh = tempfile.NamedTemporaryFile(delete=False, mode="w")
    emb_file = open(emb_tfh.name, "w")
    pred_tfh = tempfile.NamedTemporaryFile(delete=False, mode="w")
    pred_file = open(pred_tfh.name, "w")
    node_type: NodeType
    for sample_bytes in ds_iter:
        pb = training_samples_schema_pb2.SupervisedNodeClassificationSample()
        pb.ParseFromString(sample_bytes)
        training_sample = translator(pb)
        batch = _collate_node_classification_batch(
            elements=[training_sample],
            graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            preprocessed_metadata_pb_wrapper=gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
            builder=builder,
        )
        infer_batch_results: InferBatchResults = inferencer.infer_batch(batch=batch)
        node = batch.root_nodes[0]
        node_id = node.id
        node_type = node.type
        assert (
            infer_batch_results.embeddings is not None
            and infer_batch_results.predictions is not None
        )
        emb = infer_batch_results.embeddings[0].tolist()
        pred = infer_batch_results.predictions[0].tolist()
        emb_file.write(
            json.dumps(
                {
                    DEFAULT_NODE_ID_FIELD: node_id,
                    DEFAULT_EMBEDDING_FIELD: emb,
                }
            )
            + "\n"
        )
        pred_file.write(
            json.dumps(
                {
                    DEFAULT_NODE_ID_FIELD: node_id,
                    DEFAULT_PREDICTION_FIELD: pred,
                }
            )
            + "\n"
        )

    emb_file.close()
    pred_file.close()
    bq_utils = BqUtils()
    emb_path = gbml_config_pb_wrapper.shared_config.inference_metadata.node_type_to_inferencer_output_info_map[
        node_type
    ].embeddings_path
    assert DEFAULT_EMBEDDINGS_TABLE_SCHEMA.schema is not None
    bq_utils.load_file_to_bq(
        source_path=UriFactory.create_uri(emb_tfh.name),
        bq_path=emb_path,
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            schema=DEFAULT_EMBEDDINGS_TABLE_SCHEMA.schema["fields"],
        ),
        retry=True,
    )
    logger.info(f"Embeddings for {node_type} loaded to BQ table {emb_path}")

    pred_path = gbml_config_pb_wrapper.shared_config.inference_metadata.node_type_to_inferencer_output_info_map[
        node_type
    ].predictions_path
    assert DEFAULT_PREDICTIONS_TABLE_SCHEMA.schema is not None
    bq_utils.load_file_to_bq(
        source_path=UriFactory.create_uri(pred_tfh.name),
        bq_path=pred_path,
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            schema=DEFAULT_PREDICTIONS_TABLE_SCHEMA.schema["fields"],
        ),
        retry=True,
    )
    logger.info(f"Predictions for {node_type} loaded to BQ table {pred_path}")


def _infer_node_anchor_based_link_prediction_model(
    inferencer: BaseInferencer,
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
):
    def _collate_rooted_node_neighborhood_batch(
        elements: List[RootedNodeNeighborhoodSample],
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
        builder: GraphBuilder,
    ):
        dataloaded_elements = [
            {element.root_node.type: element} for element in elements
        ]
        batch = (
            RootedNodeNeighborhoodBatch.collate_pyg_rooted_node_neighborhood_minibatch(
                builder=builder,
                graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
                preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
                samples=dataloaded_elements,
            )
        )
        return batch

    builder = PygGraphBuilder()
    translator = partial(
        RootedNodeNeighborhoodBatch.preprocess_rooted_node_neighborhood_sample_fn,
        builder=builder,
        gbml_config_pb_wrapper=gbml_config_pb_wrapper,
    )

    assert isinstance(
        gbml_config_pb_wrapper.flattened_graph_metadata_pb_wrapper.output_metadata,
        flattened_graph_metadata_pb2.NodeAnchorBasedLinkPredictionOutput,
    ), f"Flattened graph metadata output of wrong type: expected {flattened_graph_metadata_pb2.NodeAnchorBasedLinkPredictionOutput.__name__}"

    node_anchor_output = (
        gbml_config_pb_wrapper.flattened_graph_metadata_pb_wrapper.output_metadata
    )
    bq_utils = BqUtils()
    for (
        node_type,
        random_negative_tfrecord_uri_prefix,
    ) in node_anchor_output.node_type_to_random_negative_tfrecord_uri_prefix.items():
        tfrecord_files = tf.io.gfile.glob(f"{random_negative_tfrecord_uri_prefix}*")

        ds_iter = tf.data.TFRecordDataset(filenames=tfrecord_files).as_numpy_iterator()

        emb_tfh = tempfile.NamedTemporaryFile(delete=False, mode="w")
        emb_file = open(emb_tfh.name, "w")

        for sample_bytes in ds_iter:
            pb = training_samples_schema_pb2.RootedNodeNeighborhood()
            pb.ParseFromString(sample_bytes)
            training_sample = translator(pb)
            batch = _collate_rooted_node_neighborhood_batch(
                elements=[training_sample],
                graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
                preprocessed_metadata_pb_wrapper=gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
                builder=builder,
            )
            infer_batch_results: InferBatchResults = inferencer.infer_batch(batch=batch)
            node = batch.root_nodes[0]
            node_id = node.id
            assert (
                NodeType(node_type) == node.type
            ), "Expected node type at this tfrecord_uri_prefix to match batch root node type"
            assert (
                infer_batch_results.embeddings is not None
            ), "Expected embeddings to be returned by inferencer"
            emb = infer_batch_results.embeddings[0].tolist()
            emb_file.write(
                json.dumps(
                    {
                        DEFAULT_NODE_ID_FIELD: node_id,
                        DEFAULT_EMBEDDING_FIELD: emb,
                    }
                )
                + "\n"
            )

        emb_file.close()

        emb_path = gbml_config_pb_wrapper.shared_config.inference_metadata.node_type_to_inferencer_output_info_map[
            node_type
        ].embeddings_path
        assert DEFAULT_EMBEDDINGS_TABLE_SCHEMA.schema is not None
        bq_utils.load_file_to_bq(
            source_path=UriFactory.create_uri(emb_tfh.name),
            bq_path=emb_path,
            job_config=bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                schema=DEFAULT_EMBEDDINGS_TABLE_SCHEMA.schema["fields"],
            ),
            retry=True,
        )
        logger.info(
            f"Embeddings for node type {node_type} loading to BQ Table {emb_path}"
        )
    logger.info("Finished loading all inferred embeddings to BQ")
