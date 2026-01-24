from __future__ import annotations

from google.cloud.bigquery.job import WriteDisposition

from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.examples.MAG240M.common import NUM_PAPER_FEATURES
from gigl.examples.MAG240M.queries import query_template_compute_average_features
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.utils.bq import BqUtils
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


class Mag240DataPreprocessorConfig(DataPreprocessorConfig):
    """
    Any data preprocessor config needs to inherit from DataPreprocessorConfig and implement the necessary methods:
    - prepare_for_pipeline: This method is called at the very start of the pipeline. Can be used to prepare any data,
    such as running BigQuery queries, or kicking of dataflow pipelines etc. to generate node/edge feature tables.
        We will use this to prepare relevant node and edge BQ tables from raw MAG240M tables that we have created using
        fetch_data.ipynb
    - get_nodes_preprocessing_spec: This method returns a dictionary of NodeDataReference to NodeDataPreprocessingSpec
        This is used to specify how to preprocess the node data using a TFT preprocessing function.
        See TFT documentation for more details: https://www.tensorflow.org/tfx/transform/get_started
    - get_edges_preprocessing_spec: This method returns a dictionary of EdgeDataReference to EdgeDataPreprocessingSpec
        This is used to specify how to preprocess the edge data using a TFT preprocessing function
    """

    def __init__(self):
        super().__init__()

        self._resource_config: GiglResourceConfigWrapper = get_resource_config()

        """
        For this experiment we will use the heterogeneous MAG240M dataset with the `paper`, `author`, and `institution` node types, and the `paper_cite_paper`,
        `author_writes_paper`, and `institution_affiliated_author` edge types.

        |  Node Type  | # raw nodes | # raw features |                         Features                         |
        |-------------| ----------- | -------------- | ---------------------------------------------------------|
        | Paper       | 121_751_666 | 768            |  Provided MAG240M features from RoBERTa sentence decoder |
        | Author      | 122_383_112 | 768            |  Average of neighboring paper features                   |
        | Institution | 25_721      | 768            |  Average of neighboring author features                  |
        """
        self._institution_affiliated_author_table = "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_author_affiliated_with_institution"
        self._author_writes_paper_table = "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_author_writes_paper"
        self._paper_cites_paper_table = "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_paper_cites_paper"

        self._paper_table = (
            "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_paper"
        )

        # We specify the node types and edge types for the heterogeneous graph;
        # Note: These types should match what is specified in task_config.yaml
        self._paper_node_type = "paper"
        self._author_node_type = "author"
        self._institution_node_type = "institution"

        self._paper_cites_paper_edge_type = EdgeType(
            NodeType(self._paper_node_type),
            Relation("cites"),
            NodeType(self._paper_node_type),
        )

        self._author_writes_paper_edge_type = EdgeType(
            NodeType(self._author_node_type),
            Relation("writes"),
            NodeType(self._paper_node_type),
        )

        self._institution_affiliated_author_edge_type = EdgeType(
            NodeType(self._institution_node_type),
            Relation("affiliated"),
            NodeType(self._author_node_type),
        )

        self._feature_list = [f"feat_{i}" for i in range(NUM_PAPER_FEATURES)]

        # Query used to compute the average of neighboring features when computing author and institution features
        self._average_feature_query = ",\n".join(
            [f"AVG({col}) AS {col}" for col in self._feature_list]
        )

        # We store a mapping of each node type to their respective table URI. We initially only have the paper feature table from the `fetch_data.ipynb` notebook, the
        # author and institution features will be computed in the `prepare_for_pipeline` step.
        self._node_tables: dict[str, str] = {self._paper_node_type: self._paper_table}

        # We store a mapping of each edge type to their respective table URI. We have all three edge tables from the `fetch_data.ipynb` notebook.
        self._edge_tables: dict[EdgeType, str] = {
            self._paper_cites_paper_edge_type: self._paper_cites_paper_table,
            self._author_writes_paper_edge_type: self._author_writes_paper_table,
            self._institution_affiliated_author_edge_type: self._institution_affiliated_author_table,
        }

    def prepare_for_pipeline(
        self, applied_task_identifier: AppliedTaskIdentifier
    ) -> None:
        """
        This function is called at the very start of the pipeline before enumerator and datapreprocessor.
        This function does not return anything. It can be overwritten to perform any operation needed
        before running the pipeline, such as gathering data for node and edge sources

        Specifically, for the MAG240M dataset, we use this function to take the raw MAG240M tables generated from fetch_data.ipynb and
        prepare the following tables:
        - author node table: Computed author features taken as the average of the features of the 1-hop paper nodes
        - institution node table: Computed institution features taken as the average of the features of the 1-hop author nodes.

        Note that the author feature table is required to be generated first prior to generating the features for institution feature table.

        :param applied_task_identifier: A unique identifier for the task being run. This is usually the job name if orchestrating
            through GiGL's orchestration logic.
        :return: None
        """

        logger.info(
            f"Preparing for pipeline with applied task identifier: {applied_task_identifier}",
        )
        bq_utils = BqUtils(project=self._resource_config.project)

        author_table = (
            f"{self._resource_config.project}.{self._resource_config.temp_assets_bq_dataset_name}."
            + f"{applied_task_identifier}_author_feature_table"
        )
        institution_table = (
            f"{self._resource_config.project}.{self._resource_config.temp_assets_bq_dataset_name}."
            + f"{applied_task_identifier}_institution_feature_table"
        )

        self._node_tables[self._author_node_type] = author_table
        self._node_tables[self._institution_node_type] = institution_table

        average_author_query = query_template_compute_average_features.format(
            feature_table=self._paper_table,
            edge_table=self._author_writes_paper_table,
            join_identifier=self._paper_node_type,
            group_by_identifier=self._author_node_type,
            average_feature_query=self._average_feature_query,
        )
        bq_utils.run_query(
            query=average_author_query,
            labels={},
            destination=author_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )

        average_institution_query = query_template_compute_average_features.format(
            feature_table=author_table,
            edge_table=self._institution_affiliated_author_table,
            join_identifier=self._author_node_type,
            group_by_identifier=self._institution_node_type,
            average_feature_query=self._average_feature_query,
        )

        bq_utils.run_query(
            query=average_institution_query,
            labels={},
            destination=institution_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )

        logger.info(
            f"Preparation for pipeline with applied task identifier: {applied_task_identifier} is complete"
            + f"Generated the following tables: {author_table}, {institution_table}",
        )

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

            node_output_id = NodeOutputIdentifier(node_type)

            # The ingestion feature spec function is used to specify the input columns and their types
            # that will be read from the NodeDataReference - which in this case is BQ.
            feature_spec_fn = build_ingestion_feature_spec_fn(
                fixed_int_fields=[node_type],
                fixed_float_fields=self._feature_list,
            )

            # We don't need any special preprocessing for the node features.
            # Thus, we can make use of a "passthrough" transform preprocessing function that simply passes the input
            # features through to the output features.
            preprocessing_fn = build_passthrough_transform_preprocessing_fn()

            output_dict[node_data_reference] = NodeDataPreprocessingSpec(
                feature_spec_fn=feature_spec_fn,
                preprocessing_fn=preprocessing_fn,
                identifier_output=node_output_id,
                features_outputs=self._feature_list,
            )
        return output_dict

    def get_edges_preprocessing_spec(
        self,
    ) -> dict[EdgeDataReference, EdgeDataPreprocessingSpec]:
        output_dict: dict[EdgeDataReference, EdgeDataPreprocessingSpec] = {}

        for edge_type, table in self._edge_tables.items():
            if edge_type.src_node_type == edge_type.dst_node_type:
                src_node_type = "src"
                dst_node_type = "dst"
            else:
                src_node_type = edge_type.src_node_type
                dst_node_type = edge_type.dst_node_type

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
