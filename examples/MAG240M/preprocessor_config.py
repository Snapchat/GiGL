from __future__ import annotations

from typing import Callable, Dict

import tensorflow as tf
import tensorflow_transform as tft
from examples.MAG240M.common import NUM_PAPER_FEATURES, TOTAL_NUM_PAPERS
from examples.MAG240M.queries import (
    query_template_cast_to_homogeneous_edge_table,
    query_template_cast_to_intermediary_homogeneous_node_table,
    query_template_computed_node_degree_table,
    query_template_generate_homogeneous_node_table,
    query_template_reindex_author_writes_paper_table,
)
from google.cloud.bigquery.job import WriteDisposition

from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.utils.bq import BqUtils
from gigl.src.data_preprocessor.data_preprocessor import DataPreprocessorConfig
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
    TFTensorDict,
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
        For this experiment we will take the MAG240M dataset and cast it to a homogeneous graph where both authors and papers
        are considered the same node. We will not account for institutions in this experiment.

        |  Node Type  | # raw nodes | # raw features |    Casted Node ID Range   |                              Notes                             |
        |-------------| ----------- | -------------- | ------------------------- | -------------------------------------------------------------- |
        | Paper       | 121_751_666 | 768            | 0 - 121_751_665           |  Modify feature vector: [degree] + [original 768 features]     |
        | Author      | 122_383_112 | 0              | 121_751_666 - 244_134_777 |  Modify feature vector: [degree] + [0]*768 to align with Paper |
        | Institution | 25_721      | 0              | NA                        |  Not used for experiment                                       |
        """
        # self.author_affil_instit_table = "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_author_affiliated_with_institution" # Not used
        self.author_write_paper_table = "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_author_writes_paper"
        self.paper_cite_paper_table = "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_paper_cites_paper"

        self.paper_table = (
            "external-snap-ci-github-gigl.public_gigl.datasets_mag240m_paper"
        )

        # We specify the node types and edge types for the homogeneous graph;
        # Note: These types should match what is specified in task_config.yaml
        self.node_type_paper_or_author = "paper_or_author"
        self.relation_references = "references"

        # We specify the column names for the input node/edge tables
        self.node_id_column_name = "node_id"
        self.src_id_column_name = "src"
        self.dst_id_column_name = "dst"
        self.fixed_int_node_feature_list = [
            "degree"
        ]  # We specify which input feature columns are an int
        self.fixed_float_node_feature_list = (
            [  # And which input feature columns are a float
                f"feat_{i}" for i in range(NUM_PAPER_FEATURES)
            ]
        )

        self.node_feature_list = (
            self.fixed_int_node_feature_list + self.fixed_float_node_feature_list
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

        # We will write the reindexed author writes paper table to this table
        # Re-indexing meaning, we will re-index the author node ids to the same node space as the paper node ids
        # i.e. author node 0 --> paper node 0 + TOTAL_NUM_PAPERS; author node 1 --> paper node 1 + TOTAL_NUM_PAPERS; ...
        self.dst_casted_homogeneous_edge_table = (
            f"{self.resource_config.project}.{self.resource_config.temp_assets_bq_dataset_name}."
            + f"{applied_task_identifier}_edge_table"
        )
        self.dst_casted_homogeneous_node_table = (
            f"{self.resource_config.project}.{self.resource_config.temp_assets_bq_dataset_name}."
            + f"{applied_task_identifier}_node_table"
        )

        dst_reindex_author_writes_paper_table = (
            f"{self.resource_config.project}.{self.resource_config.temp_assets_bq_dataset_name}."
            + f"{applied_task_identifier}_reindexed_author_writes_paper_table"
        )

        dst_interim_node_degree_table = (
            f"{self.resource_config.project}.{self.resource_config.temp_assets_bq_dataset_name}."
            + f"{applied_task_identifier}_node_degree_table"
        )
        dst_interim_casted_homogeneous_node_table = (
            f"{self.resource_config.project}.{self.resource_config.temp_assets_bq_dataset_name}."
            + f"{applied_task_identifier}_interim_node_table"
        )

        query_reindex_author_writes_paper_table = (
            query_template_reindex_author_writes_paper_table.format(
                TOTAL_NUM_PAPERS=TOTAL_NUM_PAPERS,
                author_writes_paper_table=self.author_write_paper_table,
            )
        )
        bq_utils.run_query(
            query=query_reindex_author_writes_paper_table,
            labels={},
            destination=dst_reindex_author_writes_paper_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )

        query_cast_to_homogeneous_edge_table = query_template_cast_to_homogeneous_edge_table.format(
            reindexed_author_writes_paper_table=dst_reindex_author_writes_paper_table,
            paper_cites_paper_table=self.paper_cite_paper_table,
        )
        bq_utils.run_query(
            query=query_cast_to_homogeneous_edge_table,
            labels={},
            destination=self.dst_casted_homogeneous_edge_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )

        query_computed_node_degree_table = (
            query_template_computed_node_degree_table.format(
                homogeneous_edge_table=self.dst_casted_homogeneous_edge_table,
            )
        )
        bq_utils.run_query(
            query=query_computed_node_degree_table,
            labels={},
            destination=dst_interim_node_degree_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )

        query_cast_to_intermediary_homogeneous_node_table = query_template_cast_to_intermediary_homogeneous_node_table.format(
            reindexed_author_writes_paper_table=dst_reindex_author_writes_paper_table,
            paper_table=self.paper_table,
        )
        bq_utils.run_query(
            query=query_cast_to_intermediary_homogeneous_node_table,
            labels={},
            destination=dst_interim_casted_homogeneous_node_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )

        query_generate_homogeneous_node_table = (
            query_template_generate_homogeneous_node_table.format(
                interim_node_table=dst_interim_casted_homogeneous_node_table,
                node_degree_table=dst_interim_node_degree_table,
            )
        )
        bq_utils.run_query(
            query=query_generate_homogeneous_node_table,
            labels={},
            destination=self.dst_casted_homogeneous_node_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )

        logger.info(
            f"Preparation for pipeline with applied task identifier: {applied_task_identifier} is complete"
            + f"Generated the following tables: {self.dst_casted_homogeneous_edge_table}, {self.dst_casted_homogeneous_node_table}",
        )

    def _build_node_feat_transform_preprocessing_fn(
        self,
    ) -> Callable[[TFTensorDict], TFTensorDict]:
        """
        This function builds a TFTransform preprocessing function for the node features.
        See https://www.tensorflow.org/tfx/tutorials/transform/census#create_a_tftransform_preprocessing_fn/ for details.

        In this example, we will normalize the degree feature by taking the log of the degree and then scaling it to z-score.
        Since every other feature is a "768-dimensional roBERTa sentence encoding vector" we leave it as is i.e. we "pass it through"
        See the dataset graph description for more details on the features: https://ogb.stanford.edu/docs/lsc/mag240m/

        :return: A TFTransform preprocessing function
        """

        def preprocessing_fn(inputs: TFTensorDict) -> TFTensorDict:
            degree = inputs["degree"]  # Replace 'feature' with your actual feature name

            outputs = {}
            for key, value in inputs.items():
                if key == "degree":
                    # Cast to float32 to ensure tf.maximum below works; it cant take tensors off two diff types
                    degree = tf.cast(value, tf.float32)
                    log_degree = tf.math.log(
                        tf.maximum(degree, tf.constant(1e-6))  # Prevent log(0) error
                    )
                    normalized_degree = tft.scale_to_z_score(log_degree)
                    outputs["degree"] = normalized_degree
                else:
                    outputs[key] = value  # Pass through everything else

            return outputs

        return preprocessing_fn

    def get_nodes_preprocessing_spec(
        self,
    ) -> Dict[NodeDataReference, NodeDataPreprocessingSpec]:
        # We specify where the input data is located using NodeDataReference
        # In this case, we are reading from BigQuery, thus make use off BigqueryNodeDataReference
        paper_node_data_reference: NodeDataReference = BigqueryNodeDataReference(
            reference_uri=self.dst_casted_homogeneous_node_table,
            node_type=NodeType(self.node_type_paper_or_author),
        )

        # The ingestion feature spec function is used to specify the input columns and their types
        # that will be read from the NodeDataReference - which in this case is BQ.
        feature_spec_fn = build_ingestion_feature_spec_fn(
            fixed_int_fields=[self.node_id_column_name]
            + self.fixed_int_node_feature_list,
            fixed_float_fields=self.fixed_float_node_feature_list,
        )

        preprocessing_fn = self._build_node_feat_transform_preprocessing_fn()
        node_output_id = NodeOutputIdentifier(self.node_id_column_name)
        node_features_outputs = self.node_feature_list

        return {
            paper_node_data_reference: NodeDataPreprocessingSpec(
                feature_spec_fn=feature_spec_fn,
                preprocessing_fn=preprocessing_fn,
                identifier_output=node_output_id,
                features_outputs=node_features_outputs,
            ),
        }

    def get_edges_preprocessing_spec(
        self,
    ) -> Dict[EdgeDataReference, EdgeDataPreprocessingSpec]:
        main_edge_type = EdgeType(
            src_node_type=NodeType(self.node_type_paper_or_author),
            relation=Relation(self.relation_references),
            dst_node_type=NodeType(self.node_type_paper_or_author),
        )
        # We specify the message passing paths as culmination of edges: author -> writes -> paper, and paper -> cites -> paper
        main_edge_data_ref = BigqueryEdgeDataReference(
            reference_uri=self.dst_casted_homogeneous_edge_table,
            edge_type=main_edge_type,
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
