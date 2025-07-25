from __future__ import annotations

from gigl.common.logger import Logger
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType, Relation
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
    """
    Any data preprocessor config needs to inherit from DataPreprocessorConfig and implement the necessary methods:
    - prepare_for_pipeline: This method is called at the very start of the pipeline. Can be used to prepare any data,
    such as running BigQuery queries, or kicking of dataflow pipelines etc. to generate node/edge feature tables.
    - get_nodes_preprocessing_spec: This method returns a dictionary of NodeDataReference to NodeDataPreprocessingSpec
        This is used to specify how to preprocess the node data using a TFT preprocessing function.
        See TFT documentation for more details: https://www.tensorflow.org/tfx/transform/get_started
    - get_edges_preprocessing_spec: This method returns a dictionary of EdgeDataReference to EdgeDataPreprocessingSpec
        This is used to specify how to preprocess the edge data using a TFT preprocessing function
    """

    def __init__(self):
        super().__init__()

        self._user_table = "external-snap-ci-github-gigl.public_gigl.toy_graph_heterogeneous_node_anchor_lp_user_nodes_2025-07-24--20-45-44-UTC"
        self._story_table = "external-snap-ci-github-gigl.public_gigl.toy_graph_heterogeneous_node_anchor_lp_story_nodes_2025-07-24--20-45-44-UTC"

        self._user_to_story_table = "external-snap-ci-github-gigl.public_gigl.toy_graph_heterogeneous_node_anchor_lp_user-to-story_edges_main_2025-07-24--20-45-44-UTC"
        self._story_to_user_table = "external-snap-ci-github-gigl.public_gigl.toy_graph_heterogeneous_node_anchor_lp_story-to-user_edges_main_2025-07-24--20-45-44-UTC"

        # We specify the node types and edge types for the heterogeneous graph;
        # Note: These types should match what is specified in task_config.yaml

        self._user_node_type = "user"
        self._story_node_type = "story"

        self._user_to_story_edge_type =  EdgeType(
            NodeType(self._user_node_type),
            Relation("to"),
            NodeType(self._story_node_type),
        )

        self._story_to_user_edge_type =  EdgeType(
            NodeType(self._story_node_type),
            Relation("to"),
            NodeType(self._user_node_type),
        )

        self._float_feature_list = ["f0", "f1"]

        # We store a mapping of each node type to their respective table URI.
        self._node_tables: dict[str, str] = {self._user_node_type: self._user_table, self._story_node_type: self._story_table}

        # We store a mapping of each edge type to their respective table URI. We have all three edge tables from the `fetch_data.ipynb` notebook.
        self._edge_tables: dict[EdgeType, str] = {
            self._user_to_story_edge_type: self._user_to_story_table,
            self._story_to_user_edge_type: self._story_to_user_table
        }

    def prepare_for_pipeline(
        self, applied_task_identifier: AppliedTaskIdentifier
    ) -> None:
        """
        This function is called at the very start of the pipeline before enumerator and datapreprocessor.
        This function does not return anything. It can be overwritten to perform any operation needed
        before running the pipeline, such as gathering data for node and edge sources

        :param applied_task_identifier: A unique identifier for the task being run. This is usually the job name if orchestrating
            through GiGL's orchestration logic.
        :return: None
        """
        pass

    def get_nodes_preprocessing_spec(
        self,
    ) -> dict[NodeDataReference, NodeDataPreprocessingSpec]:
        # We specify where the input data is located using NodeDataReference
        # In this case, we are reading from BigQuery, thus make use off BigqueryNodeDataReference

        output_dict: dict[NodeDataReference, NodeDataPreprocessingSpec] = {}
        node_data_reference: NodeDataReference

        for node_type, table in self._node_tables.items():
            node_data_reference = BigqueryNodeDataReference(
                reference_uri=table,
                node_type=NodeType(node_type),
            )

            node_output_id = NodeOutputIdentifier("node_id")

            # The ingestion feature spec function is used to specify the input columns and their types
            # that will be read from the NodeDataReference - which in this case is BQ.
            feature_spec_fn = build_ingestion_feature_spec_fn(
                fixed_int_fields=["node_id"],
                fixed_float_fields=self._float_feature_list,
            )

            # We don't need any special preprocessing for the node features.
            # Thus, we can make use of a "passthrough" transform preprocessing function that simply passes the input
            # features through to the output features.
            preprocessing_fn = build_passthrough_transform_preprocessing_fn()

            output_dict[node_data_reference] = NodeDataPreprocessingSpec(
                feature_spec_fn=feature_spec_fn,
                preprocessing_fn=preprocessing_fn,
                identifier_output=node_output_id,
                features_outputs=self._float_feature_list,
            )
        return output_dict

    def get_edges_preprocessing_spec(
        self,
    ) -> dict[EdgeDataReference, EdgeDataPreprocessingSpec]:
        output_dict: dict[EdgeDataReference, EdgeDataPreprocessingSpec] = {}

        for edge_type, table in self._edge_tables.items():
            src_node_type = "src"
            dst_node_type = "dst"

            edge_ref = BigqueryEdgeDataReference(
                reference_uri=table,
                edge_type=edge_type,
                edge_usage_type=EdgeUsageType.MAIN,
            )

            feature_spec_fn = build_ingestion_feature_spec_fn(
                fixed_int_fields=[
                    src_node_type,
                    dst_node_type,
                ]
            )

            # We don't need any special preprocessing for the edges as there are no edge features to begin with.
            # Thus, we can make use of a "passthrough" transform preprocessing function that simply passes the input
            # features through to the output features.
            preprocessing_fn = build_passthrough_transform_preprocessing_fn()
            edge_output_id = EdgeOutputIdentifier(
                src_node=NodeOutputIdentifier(src_node_type),
                dst_node=NodeOutputIdentifier(dst_node_type),
            )

            output_dict[edge_ref] = EdgeDataPreprocessingSpec(
                identifier_output=edge_output_id,
                feature_spec_fn=feature_spec_fn,
                preprocessing_fn=preprocessing_fn,
            )

        return output_dict
