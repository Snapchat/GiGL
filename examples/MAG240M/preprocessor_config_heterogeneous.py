from __future__ import annotations

from typing import Dict

from examples.MAG240M.common import NUM_PAPER_FEATURES
from examples.MAG240M.queries import query_template_compute_average_features
from google.cloud.bigquery.job import WriteDisposition

from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
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
    - prepare_for_pipeline: This method is called at the very start of the pipeline. Can be used to prepare any data
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

        self.resource_config: GiglResourceConfigWrapper = get_resource_config()

        """
        For this experiment we will use the heterogeneous MAG240M dataset with the `paper`, `author`, and `institution` node types and the `paper_cite_paper`,
        `author_writes_paper`, and `author_affiliated_institution_table` edge types.

        |  Node Type  | # raw nodes | # raw features |                           Notes                          |
        |-------------| ----------- | -------------- | ---------------------------------------------------------|
        | Paper       | 121_751_666 | 768            |  Provided MAG240M features from RoBERTa sentence decoder |
        | Author      | 122_383_112 | 768            |  Average of neighboring paper features                   |
        | Institution | 25_721      | 768            |  Average of neighboring author features                  |
        """
        self.author_affiliated_institution_table = "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_author_affiliated_with_institution"
        self.author_writes_paper_table = "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_author_writes_paper"
        self.paper_cites_paper_table = "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_paper_cites_paper"

        self.paper_table = (
            "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_paper"
        )

        # We specify the node types and edge types for the heterogeneous graph;
        # Note: These types should match what is specified in task_config.yaml
        self.paper_node_type = "paper"
        self.author_node_type = "author"
        self.institution_node_type = "institution"

        self.paper_cites_paper_edge_type = EdgeType(
            NodeType(self.paper_node_type),
            Relation("cites"),
            NodeType(self.paper_node_type),
        )

        self.author_writes_paper_edge_type = EdgeType(
            NodeType(self.author_node_type),
            Relation("writes"),
            NodeType(self.paper_node_type),
        )

        self.author_affiliated_insitution_edge_type = EdgeType(
            NodeType(self.author_node_type),
            Relation("affiliated"),
            NodeType(self.institution_node_type),
        )

        # We specify the column names for the input node/edge tables
        self.node_id_column_name = "node_id"
        self.src_id_column_name = "src"
        self.dst_id_column_name = "dst"
        self.feature_list = [f"feat_{i}" for i in range(NUM_PAPER_FEATURES)]
        self.average_feature_query = ",\n".join(
            [f"AVG({col}) AS {col}" for col in self.feature_list]
        )

    def prepare_for_pipeline(
        self, applied_task_identifier: AppliedTaskIdentifier
    ) -> None:
        """
        This function is called at the very start of the pipeline before enumerator and datapreprocessor.
        This function does not return anything. It can be overwritten to perform any operation needed
        before running the pipeline, such as gathering data for node and edge sources

        Specifically, we use this function to take the raw MAG240M tables generated from fetch_data.ipynb and
        prepare the following tables:
        - dst_casted_homogeneous_edge_table: edge table where both author writes paper and paper cites paper tables are combined
            into a single edge type. See info in __init__ for more details on the the node ids.
        - dst_casted_homogeneous_node_table: node table where both author and paper nodes are combined into a single node type.
            See info in __init__ for more details on the the node ids and the features in this table.

        Note this is where we also use BQ to concat the raw node degree as an input feature for all the nodes. We also
        zero pad a 768 dim input feature for the author nodes - as discussed in __init__.


        :param applied_task_identifier: A unique identifier for the task being run. This is usually the job name if orchestrating
            through GiGL's orchestration logic.
        :return: None
        """

        logger.info(
            f"Preparing for pipeline with applied task identifier: {applied_task_identifier}",
        )
        bq_utils = BqUtils(project=self.resource_config.project)

        self.author_table = (
            f"{self.resource_config.project}.{self.resource_config.temp_assets_bq_dataset_name}."
            + f"{applied_task_identifier}_author_feature_table"
        )
        self.institution_table = (
            f"{self.resource_config.project}.{self.resource_config.temp_assets_bq_dataset_name}."
            + f"{applied_task_identifier}_institution_feature_table"
        )

        average_author_query = query_template_compute_average_features.format(
            feature_table=self.paper_table,
            edge_table=self.author_writes_paper_table,
            join_identifier=self.paper_node_type,
            group_by_identifier=self.author_node_type,
            average_feature_query=self.average_feature_query,
        )
        bq_utils.run_query(
            query=average_author_query,
            labels={},
            destination=self.author_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )

        average_institution_query = query_template_compute_average_features.format(
            feature_table=self.author_table,
            edge_table=self.author_affiliated_institution_table,
            join_identifier=self.author_node_type,
            group_by_identifier=self.institution_node_type,
            average_feature_query=self.average_feature_query,
        )

        bq_utils.run_query(
            query=average_institution_query,
            labels={},
            destination=self.institution_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )

        logger.info(
            f"Preparation for pipeline with applied task identifier: {applied_task_identifier} is complete"
            + f"Generated the following tables: {self.author_table}, {self.institution_table}",
        )

    def get_nodes_preprocessing_spec(
        self,
    ) -> Dict[NodeDataReference, NodeDataPreprocessingSpec]:
        # We specify where the input data is located using NodeDataReference
        # In this case, we are reading from BigQuery, thus make use off BigqueryNodeDataReference

        paper_node_data_reference: NodeDataReference = BigqueryNodeDataReference(
            reference_uri=self.paper_table,
            node_type=NodeType(self.paper_node_type),
        )

        author_node_data_reference: NodeDataReference = BigqueryNodeDataReference(
            reference_uri=self.author_table,
            node_type=NodeType(self.author_node_type),
        )

        institution_node_data_reference: NodeDataReference = BigqueryNodeDataReference(
            reference_uri=self.institution_node_type,
            node_type=NodeType(self.institution_node_type),
        )

        # The ingestion feature spec function is used to specify the input columns and their types
        # that will be read from the NodeDataReference - which in this case is BQ.
        feature_spec_fn = build_ingestion_feature_spec_fn(
            fixed_int_fields=[self.node_id_column_name],
            fixed_float_fields=self.feature_list,
        )

        # We don't need any special preprocessing for the node features.
        # Thus, we can make use of a "passthrough" transform preprocessing function that simply passes the input
        # features through to the output features.
        preprocessing_fn = build_passthrough_transform_preprocessing_fn()

        node_output_id = NodeOutputIdentifier(self.node_id_column_name)
        node_features_outputs = self.feature_list

        return {
            paper_node_data_reference: NodeDataPreprocessingSpec(
                feature_spec_fn=feature_spec_fn,
                preprocessing_fn=preprocessing_fn,
                identifier_output=node_output_id,
                features_outputs=node_features_outputs,
            ),
            author_node_data_reference: NodeDataPreprocessingSpec(
                feature_spec_fn=feature_spec_fn,
                preprocessing_fn=preprocessing_fn,
                identifier_output=node_output_id,
                features_outputs=node_features_outputs,
            ),
            institution_node_data_reference: NodeDataPreprocessingSpec(
                feature_spec_fn=feature_spec_fn,
                preprocessing_fn=preprocessing_fn,
                identifier_output=node_output_id,
                features_outputs=node_features_outputs,
            ),
        }

    def get_edges_preprocessing_spec(
        self,
    ) -> Dict[EdgeDataReference, EdgeDataPreprocessingSpec]:
        paper_cites_paper_edge_ref = BigqueryEdgeDataReference(
            reference_uri=self.paper_cites_paper_table,
            edge_type=self.paper_cites_paper_edge_type,
            edge_usage_type=EdgeUsageType.MAIN,
        )

        author_writes_paper_edge_ref = BigqueryEdgeDataReference(
            reference_uri=self.author_writes_paper_table,
            edge_type=self.author_writes_paper_edge_type,
            edge_usage_type=EdgeUsageType.MAIN,
        )

        author_affiliated_insititution_edge_ref = BigqueryEdgeDataReference(
            reference_uri=self.author_affiliated_institution_table,
            edge_type=self.author_affiliated_insitution_edge_type,
            edge_usage_type=EdgeUsageType.MAIN,
        )

        # Our training task is link prediction on paper -> cites -> paper edges, thus we specify this as the only positive edge
        positive_edge_data_ref = BigqueryEdgeDataReference(
            reference_uri=self.paper_cite_paper_table,
            edge_type=main_edge_type,
            edge_usage_type=EdgeUsageType.POSITIVE,
        )

        feature_spec_fn = build_ingestion_feature_spec_fn(
            fixed_int_fields=[
                self.src_id_column_name,
                self.dst_id_column_name,
            ]
        )

        # We don't need any special preprocessing for the edges as there are no edge features to begin with.
        # Thus, we can make use of a "passthrough" transform preprocessing function that simply passes the input
        # features through to the output features.
        preprocessing_fn = build_passthrough_transform_preprocessing_fn()
        edge_output_id = EdgeOutputIdentifier(
            src_node=NodeOutputIdentifier(self.src_id_column_name),
            dst_node=NodeOutputIdentifier(self.dst_id_column_name),
        )

        edge_data_preprocessing_spec = EdgeDataPreprocessingSpec(
            identifier_output=edge_output_id,
            feature_spec_fn=feature_spec_fn,
            preprocessing_fn=preprocessing_fn,
        )

        return {
            main_edge_data_ref: edge_data_preprocessing_spec,
            positive_edge_data_ref: edge_data_preprocessing_spec,
        }
