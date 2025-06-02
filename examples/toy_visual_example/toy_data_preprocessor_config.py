from typing import Any, Dict

import tensorflow as tf
import tensorflow_transform as tft

from gigl.common.logger import Logger
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.data_preprocessor.lib.data_preprocessor_config import (
    DataPreprocessorConfig,
    build_ingestion_feature_spec_fn,
    build_passthrough_transform_preprocessing_fn,
)
from gigl.src.data_preprocessor.lib.ingest.bigquery import (
    BigqueryEdgeDataReference,
    BigqueryNodeDataReference,
)
from gigl.src.data_preprocessor.lib.ingest.reference import (
    EdgeDataReference,
    NodeDataReference,
)
from gigl.src.data_preprocessor.lib.types import (
    EdgeDataPreprocessingSpec,
    EdgeOutputIdentifier,
    NodeDataPreprocessingSpec,
    NodeOutputIdentifier,
)

logger = Logger()


class ToyDataPreprocessorConfig(DataPreprocessorConfig):
    def __init__(self, **kwargs):
        self._bq_nodes_table_name = kwargs.get("bq_nodes_table_name", "")
        self._bq_edges_table_name = kwargs.get("bq_edges_table_name", "")

        if self._bq_nodes_table_name != "" and self._bq_edges_table_name != "":
            logger.info(
                f"Found BQ table reference."
                f"bq_nodes_table_name: {self._bq_nodes_table_name}, "
                f"bq_edges_table_name: {self._bq_edges_table_name}"
            )
        else:
            raise ValueError(
                f"Both 'bq_edges_table_name' and 'bq_nodes_table_name' must be provided for Toy Graph Data."
            )

        self._node_type = "user"
        self._relation_type = "is_friends_with"
        self._node_features_list = ["f0", "f1"]
        self._node_id_input = "node_id"
        self._src_node = "src"
        self._dst_node = "dst"

    # TODO: If no table name provided, call mock code to generate tables.
    def prepare_for_pipeline(
        self, applied_task_identifier: AppliedTaskIdentifier
    ) -> None:
        pass

    def get_nodes_preprocessing_spec(
        self,
    ) -> Dict[NodeDataReference, NodeDataPreprocessingSpec]:
        node_data_ref = BigqueryNodeDataReference(
            reference_uri=self._bq_nodes_table_name,
            node_type=NodeType(self._node_type),
        )

        feature_spec_fn = build_ingestion_feature_spec_fn(
            fixed_int_fields=[self._node_id_input],
            fixed_float_fields=self._node_features_list,
        )

        preprocessing_fn = build_passthrough_transform_preprocessing_fn()
        node_output_id = NodeOutputIdentifier(self._node_id_input)
        node_features_outputs = self._node_features_list

        return {
            node_data_ref: NodeDataPreprocessingSpec(
                identifier_output=node_output_id,
                features_outputs=node_features_outputs,
                feature_spec_fn=feature_spec_fn,
                preprocessing_fn=preprocessing_fn,
            )
        }

    def get_edges_preprocessing_spec(
        self,
    ) -> Dict[EdgeDataReference, EdgeDataPreprocessingSpec]:
        edge_data_ref = BigqueryEdgeDataReference(
            reference_uri=self._bq_edges_table_name,
            edge_type=EdgeType(
                src_node_type=NodeType(self._node_type),
                relation=Relation(self._relation_type),
                dst_node_type=NodeType(self._node_type),
            ),
        )

        feature_spec_fn = build_ingestion_feature_spec_fn(
            fixed_int_fields=[self._src_node, self._dst_node]
        )

        preprocessing_fn = build_passthrough_transform_preprocessing_fn()

        edge_output_id = EdgeOutputIdentifier(
            src_node=NodeOutputIdentifier(self._src_node),
            dst_node=NodeOutputIdentifier(self._dst_node),
        )

        return {
            edge_data_ref: EdgeDataPreprocessingSpec(
                identifier_output=edge_output_id,
                feature_spec_fn=feature_spec_fn,
                preprocessing_fn=preprocessing_fn,
            )
        }

    # TODO: Test custom transformation
    def custom_nodes_preprocessing_fn(
        self, inputs: Dict[str, tf.Tensor], config: Dict[str, Any]
    ) -> Dict[str, tf.Tensor]:
        outputs = {}

        # Pass through the node ID field
        outputs["node_id"] = inputs["node_id"]

        # Scale the float features to z-scores
        float_features = ["f0", "f1"]
        for feature in float_features:
            outputs[feature] = tft.scale_to_z_score(inputs[feature])

        return outputs
