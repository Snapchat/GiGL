import unittest

from gigl.common.logger import Logger
from gigl.src.common.types.exception import (
    SubgraphSamplingValidationError,
    SubgraphSamplingValidationErrorType,
)
from gigl.src.common.types.pb_wrappers.subgraph_sampling_strategy import (
    SubgraphSamplingStrategyPbWrapper,
)
from snapchat.research.gbml.gbml_config_pb2 import GbmlConfig
from snapchat.research.gbml.graph_schema_pb2 import GraphMetadata
from snapchat.research.gbml.subgraph_sampling_strategy_pb2 import (
    MessagePassingPath,
    MessagePassingPathStrategy,
    RandomUniform,
    SamplingDirection,
    SamplingOp,
    SubgraphSamplingStrategy,
)
from tests.test_assets.graph_metadata_constants import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB,
    DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB,
    DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
    DEFAULT_NABLP_HOMOGENEOUS_TASK_METADATA_PB,
    EXAMPLE_HETEROGENEOUS_EDGE_TYPES,
    EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB,
    EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR,
    EXAMPLE_NABLP_HETEROGENEOUS_TASK_METADATA_PB,
)

logger = Logger()


class SubgraphSamplingStrategyValidationUnitTest(unittest.TestCase):
    """
    This tests checks the Subgraph Sampling Strategy Pb Wrapper validation check for logic affecting an entire DAG. Specifically, we test for
    correctness of validation checks from both building the SubgraphSamplingStrategyPbWrapper DAG and through calling the 'validate_dags' function.
    We test the success of several example homogeneous and heterogeneous dags and edge cases as well as failure through several exceptions from SubgraphSamplingValidationErrorType.
    These exceptions are:
        - REPEATED_OP_NAME: Sampling op name is repeated locally in a sampling op dag
        - BAD_INPUT_OP_NAME: Sampling op is given an input_op_name which doesn't exist in a dag
        - DAG_CONTAINS_CYCLE: DAG contains a cycle, resulting in an infinite traversal
        - REPEATED_ROOT_NODE_TYPE: Multiple DAGs are provided for the same Root Node Type
        - ROOT_NODE_TYPE_NOT_IN_GRAPH_METADATA: Root node type does not exist in provided graph metadata
        - ROOT_NODE_TYPE_NOT_IN_TASK_METADATA: Root node type does not exist in provided task metadata
        - MISSING_ROOT_SAMPLING_OP: A Sampling op dag does not contain any root sampling ops from the root node type
        - SAMPLING_OP_EDGE_TYPE_NOT_IN_GRAPH_METADATA: An edge type of a sampling op does not exist in the graph metadata
        - MISSING_EXPECTED_ROOT_NODE_TYPE: An expected root node type from task metadata does not exist in any provided dag

    The final exception, CONTAINS_INVALID_EDGE_IN_DAG, for testing edge type validity between parent and children within a dag, is located in
    a separate test for SamplingOpPbWRapper.
    """

    def setUp(self) -> None:
        self.heterogeneous_node_type_zero = EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[
            0
        ]  # Node Type '0'
        self.heterogeneous_node_type_one = EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[
            1
        ]  # Node Type '1'
        self.heterogeneous_node_type_two = EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[
            2
        ]  # Node Type '2'

        self.heterogeneous_edge_zero_to_one = EXAMPLE_HETEROGENEOUS_EDGE_TYPES[
            0
        ]  # Edge Type mapping source node type '0' to destination node type '1'
        self.heterogeneous_edge_zero_to_two = EXAMPLE_HETEROGENEOUS_EDGE_TYPES[
            1
        ]  # Edge Type mapping source node type '0' to destination node type '2'
        self.heterogeneous_edge_one_to_two = EXAMPLE_HETEROGENEOUS_EDGE_TYPES[
            2
        ]  # Edge Type mapping source node type '1' to destination node type '2'

        # Example Homogeneous Sampling Op Template33333
        self.example_homogeneous_sampling_op = SamplingOp(
            op_name="example_homogeneous_sampling_op",
            edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB,
            input_op_names=[],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.INCOMING,
        )

        # Example Homogeneous Sampling Op DAG
        self.example_homogeneous_dag = SubgraphSamplingStrategyPbWrapper(
            SubgraphSamplingStrategy(
                message_passing_paths=MessagePassingPathStrategy(
                    paths=[
                        MessagePassingPath(
                            root_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
                            sampling_ops=[
                                self.example_homogeneous_sampling_op,
                            ],
                        )
                    ]
                )
            )
        )

        # Example Heterogeneous Sampling Op Template
        self.example_heterogeneous_sampling_op_root_0 = SamplingOp(
            op_name="example_heterogeneous_sampling_op_0",
            edge_type=self.heterogeneous_edge_zero_to_one,
            input_op_names=[],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.OUTGOING,
        )

        self.example_heterogeneous_sampling_op_root_1 = SamplingOp(
            op_name="example_heterogeneous_sampling_op_1",
            edge_type=self.heterogeneous_edge_zero_to_one,
            input_op_names=[],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.INCOMING,
        )

        self.example_heterogeneous_sampling_op_root_2 = SamplingOp(
            op_name="example_heterogeneous_sampling_op_2",
            edge_type=self.heterogeneous_edge_zero_to_two,
            input_op_names=[],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.INCOMING,
        )

        # Example Heterogeneous Sampling Op DAG
        self.example_heterogeneous_dag = SubgraphSamplingStrategyPbWrapper(
            SubgraphSamplingStrategy(
                message_passing_paths=MessagePassingPathStrategy(
                    paths=[
                        MessagePassingPath(
                            root_node_type=EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[0],
                            sampling_ops=[
                                self.example_heterogeneous_sampling_op_root_0,
                            ],
                        ),
                        MessagePassingPath(
                            root_node_type=EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[1],
                            sampling_ops=[
                                self.example_heterogeneous_sampling_op_root_1,
                            ],
                        ),
                    ]
                )
            )
        )

    def tearDown(self) -> None:
        pass

    def __assert_subgraph_sampling_validate_dags_exception(
        self,
        subgraph_sampling_strategy_pb_wrapper: SubgraphSamplingStrategyPbWrapper,
        task_metadata_pb: GbmlConfig.TaskMetadata,
        graph_metadata_pb: GraphMetadata,
        error_type: SubgraphSamplingValidationErrorType,
    ):
        """
        Helper function to receive exception if the subgraph sampling strategy pb wrapper validate dags call is invalid
        with respect to specified error type
        Args:
            subgraph_sampling_strategy_pb_wrapper (SubgraphSamplingStrategyPbWrapper): Specified subgraph sampling strategy pb warapper to validate
            task_metadata_pb (GbmlConfig.TaskMetadata): Task metadata for current validation check
            graph_metadata_pb (GraphMetadata): Graph metadata for current validation check
            error_type (SubgraphSamplingValidationErrorType): Error type expected to be raised for this assertion
        """
        with self.assertRaises(
            SubgraphSamplingValidationError
        ) as subgraph_sampling_exception:
            subgraph_sampling_strategy_pb_wrapper.validate_dags(
                graph_metadata_pb=graph_metadata_pb,
                task_metadata_pb=task_metadata_pb,
            )
        self.assertEqual(subgraph_sampling_exception.exception.error_type, error_type)

    def test_successful_validation_for_example_sampling_op_dags(self) -> None:
        """
        Tests the validation checks pass for example homogeneous and heterogeneous sampling op dags
        """
        self.example_homogeneous_dag.validate_dags(
            graph_metadata_pb=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB,
            task_metadata_pb=DEFAULT_NABLP_HOMOGENEOUS_TASK_METADATA_PB,
        )

        self.example_heterogeneous_dag.validate_dags(
            graph_metadata_pb=EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB,
            task_metadata_pb=EXAMPLE_NABLP_HETEROGENEOUS_TASK_METADATA_PB,
        )

    def test_repeat_root_node_type_exception(self) -> None:
        """
        Testing whether exception 'REPEATED_ROOT_NODE_TYPE' is correctly raised on DAG with at least one repeated root node type
        """
        with self.assertRaises(
            SubgraphSamplingValidationError
        ) as subgraph_sampling_exception:
            repeated_root_node_type_dag = SubgraphSamplingStrategyPbWrapper(
                SubgraphSamplingStrategy(
                    message_passing_paths=MessagePassingPathStrategy(
                        paths=[
                            MessagePassingPath(
                                root_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
                                sampling_ops=[
                                    self.example_homogeneous_sampling_op,
                                ],
                            ),
                            MessagePassingPath(
                                root_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
                                sampling_ops=[
                                    self.example_homogeneous_sampling_op,
                                ],
                            ),
                        ]
                    )
                )
            )
        self.assertEqual(
            subgraph_sampling_exception.exception.error_type,
            SubgraphSamplingValidationErrorType.REPEATED_ROOT_NODE_TYPE,
        )

    def test_repeated_sampling_op_name_exception(self):
        """
        Testing whether exception 'REPEATED_OP_NAME' is correctly raised on sampling op names are not
        locally unique (same sampling op name within the same DAG) and that 'REPEATED_OP_NAME' is not
        raised on sampling op names that are globally unique (same sampline op name across different DAGs).
        """

        # Assets for DAG with locally repeated sampling op names
        locally_repeated_sampling_op_1 = SamplingOp(
            op_name="locally_repeated_sampling_op",
            edge_type=self.heterogeneous_edge_zero_to_two,
            input_op_names=[],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.INCOMING,
        )

        locally_repeated_sampling_op_2 = SamplingOp(
            op_name="locally_repeated_sampling_op",
            edge_type=self.heterogeneous_edge_one_to_two,
            input_op_names=[],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.INCOMING,
        )

        # Assets for DAG with globally repeated sampling op names
        globally_repeated_sampling_op_1 = SamplingOp(
            op_name="globally_repeated_sampling_op",
            edge_type=self.heterogeneous_edge_zero_to_one,
            input_op_names=[],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.INCOMING,
        )

        globally_repeated_sampling_op_2 = SamplingOp(
            op_name="globally_repeated_sampling_op",
            edge_type=self.heterogeneous_edge_zero_to_two,
            input_op_names=[],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.INCOMING,
        )

        # Ensure sampling op names are locally unique
        with self.assertRaises(
            SubgraphSamplingValidationError
        ) as subgraph_sampling_exception:
            locally_repeated_sampling_op_dag = SubgraphSamplingStrategyPbWrapper(
                SubgraphSamplingStrategy(
                    message_passing_paths=MessagePassingPathStrategy(
                        paths=[
                            MessagePassingPath(
                                root_node_type=self.heterogeneous_node_type_two,
                                sampling_ops=[
                                    locally_repeated_sampling_op_1,
                                    locally_repeated_sampling_op_2,
                                ],
                            )
                        ]
                    )
                )
            )
        self.assertEqual(
            subgraph_sampling_exception.exception.error_type,
            SubgraphSamplingValidationErrorType.REPEATED_OP_NAME,
        )

        # Globally shared sampling op names do not error
        globally_repeated_sampling_op_dag = SubgraphSamplingStrategyPbWrapper(
            SubgraphSamplingStrategy(
                message_passing_paths=MessagePassingPathStrategy(
                    paths=[
                        MessagePassingPath(
                            root_node_type=self.heterogeneous_node_type_one,
                            sampling_ops=[
                                globally_repeated_sampling_op_1,
                            ],
                        ),
                        MessagePassingPath(
                            root_node_type=self.heterogeneous_node_type_two,
                            sampling_ops=[
                                globally_repeated_sampling_op_2,
                            ],
                        ),
                    ]
                )
            )
        )

    def test_bad_input_op_name_exception(self) -> None:
        """
        Test whether exception 'BAD_INPUT_OP_NAME' is correctly raised on sampling op with an input op name which does not exist
        """
        nonexistent_input_name_sampling_op = SamplingOp(
            op_name="nonexistent_input_name_sampling_op",
            edge_type=self.heterogeneous_edge_zero_to_one,
            input_op_names=["nonexistent_input_name"],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.INCOMING,
        )
        with self.assertRaises(
            SubgraphSamplingValidationError
        ) as subgraph_sampling_exception:
            nonexistent_input_name_sampling_op_dag = SubgraphSamplingStrategyPbWrapper(
                SubgraphSamplingStrategy(
                    message_passing_paths=MessagePassingPathStrategy(
                        paths=[
                            MessagePassingPath(
                                root_node_type=self.heterogeneous_node_type_one,
                                sampling_ops=[
                                    nonexistent_input_name_sampling_op,
                                ],
                            )
                        ]
                    )
                )
            )

        self.assertEqual(
            subgraph_sampling_exception.exception.error_type,
            SubgraphSamplingValidationErrorType.BAD_INPUT_OP_NAME,
        )

    def test_sampling_op_edge_not_in_graph_metadata_exception(self):
        """
        Test whether exception 'SAMPLING_OP_EDGE_TYPE_NOT_IN_GRAPH_METADATA' is correctly raised
        when dag contains a sampling op edge type not in graph metadata
        """
        nonexistent_edge_type_sampling_op = SamplingOp(
            op_name="nonexistent_edge_type_sampling_op",
            edge_type=self.heterogeneous_edge_zero_to_one,
            input_op_names=[],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.OUTGOING,
        )

        nonexistent_edge_type_dag = SubgraphSamplingStrategyPbWrapper(
            SubgraphSamplingStrategy(
                message_passing_paths=MessagePassingPathStrategy(
                    paths=[
                        MessagePassingPath(
                            root_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
                            sampling_ops=[nonexistent_edge_type_sampling_op],
                        ),
                    ]
                )
            )
        )

        self.__assert_subgraph_sampling_validate_dags_exception(
            subgraph_sampling_strategy_pb_wrapper=nonexistent_edge_type_dag,
            graph_metadata_pb=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB,
            task_metadata_pb=DEFAULT_NABLP_HOMOGENEOUS_TASK_METADATA_PB,
            error_type=SubgraphSamplingValidationErrorType.SAMPLING_OP_EDGE_TYPE_NOT_IN_GRAPH_METADATA,
        )

    def test_dag_cycle_detection(self):
        """
        Tests whether exception'DAG_CONTAINS_CYCLE' is correctly raised when dag contains a cyle and that
        'DAG_CONTAINS_CYCLE' is not raised when it does not contain a cycle.
        """

        # DAG with a cycle with path (cycle_op_2 -> cycle_op_4 -> cycle_op_3 -> cycle_op_2 -> cycle_op_1)
        cycle_op_1_pb = SamplingOp(
            op_name="cycle_op_1",
            edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB,
            input_op_names=[],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.OUTGOING,
        )

        cycle_op_2_pb = SamplingOp(
            op_name="cycle_op_2",
            edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB,
            input_op_names=["cycle_op_1", "cycle_op_4"],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.OUTGOING,
        )

        cycle_op_3_pb = SamplingOp(
            op_name="cycle_op_3",
            edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB,
            input_op_names=["cycle_op_2"],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.OUTGOING,
        )

        cycle_op_4_pb = SamplingOp(
            op_name="cycle_op_4",
            edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB,
            input_op_names=["cycle_op_3"],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.OUTGOING,
        )

        cycle_dag = SubgraphSamplingStrategyPbWrapper(
            SubgraphSamplingStrategy(
                message_passing_paths=MessagePassingPathStrategy(
                    paths=[
                        MessagePassingPath(
                            root_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
                            sampling_ops=[
                                cycle_op_1_pb,
                                cycle_op_2_pb,
                                cycle_op_3_pb,
                                cycle_op_4_pb,
                            ],
                        )
                    ]
                )
            )
        )

        # DAG with no cycle
        no_cycle_op_1_pb = SamplingOp(
            op_name="no_cycle_op_1",
            edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB,
            input_op_names=[],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.OUTGOING,
        )

        no_cycle_op_2_pb = SamplingOp(
            op_name="no_cycle_op_2",
            edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB,
            input_op_names=["no_cycle_op_1"],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.OUTGOING,
        )

        no_cycle_op_3_pb = SamplingOp(
            op_name="no_cycle_op_3",
            edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB,
            input_op_names=["no_cycle_op_2", "no_cycle_op_1"],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.OUTGOING,
        )

        no_cycle_dag = SubgraphSamplingStrategyPbWrapper(
            SubgraphSamplingStrategy(
                message_passing_paths=MessagePassingPathStrategy(
                    paths=[
                        MessagePassingPath(
                            root_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
                            sampling_ops=[
                                no_cycle_op_1_pb,
                                no_cycle_op_2_pb,
                                no_cycle_op_3_pb,
                            ],
                        )
                    ]
                )
            )
        )

        self.__assert_subgraph_sampling_validate_dags_exception(
            subgraph_sampling_strategy_pb_wrapper=cycle_dag,
            graph_metadata_pb=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB,
            task_metadata_pb=DEFAULT_NABLP_HOMOGENEOUS_TASK_METADATA_PB,
            error_type=SubgraphSamplingValidationErrorType.DAG_CONTAINS_CYCLE,
        )

        no_cycle_dag.validate_dags(
            graph_metadata_pb=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB,
            task_metadata_pb=DEFAULT_NABLP_HOMOGENEOUS_TASK_METADATA_PB,
        )

    def test_dag_root_node_type_not_in_graph_metadata_exception(self) -> None:
        """
        Test whether exception 'ROOT_NODE_TYPE_NOT_IN_GRAPH_METADATA' is correctly raised
        when dag contains root node type not in graph metadata
        """
        nonexistent_root_node_type_in_graph_metadata_dag = (
            SubgraphSamplingStrategyPbWrapper(
                SubgraphSamplingStrategy(
                    message_passing_paths=MessagePassingPathStrategy(
                        paths=[
                            MessagePassingPath(
                                root_node_type=self.heterogeneous_node_type_two,
                                sampling_ops=[
                                    self.example_homogeneous_sampling_op,
                                ],
                            ),
                        ]
                    )
                )
            )
        )

        self.__assert_subgraph_sampling_validate_dags_exception(
            subgraph_sampling_strategy_pb_wrapper=nonexistent_root_node_type_in_graph_metadata_dag,
            graph_metadata_pb=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB,
            task_metadata_pb=DEFAULT_NABLP_HOMOGENEOUS_TASK_METADATA_PB,
            error_type=SubgraphSamplingValidationErrorType.ROOT_NODE_TYPE_NOT_IN_GRAPH_METADATA,
        )

    def test_dag_root_node_type_not_in_task_metadata_exception(self) -> None:
        """
        Test whether exception 'ROOT_NODE_TYPE_NOT_IN_TASK_METADATA' is correctly raised
        when dag contains root node type not in task metadata
        """
        nonexistent_root_node_type_in_task_metadata_dag = (
            SubgraphSamplingStrategyPbWrapper(
                SubgraphSamplingStrategy(
                    message_passing_paths=MessagePassingPathStrategy(
                        paths=[
                            MessagePassingPath(
                                root_node_type=self.heterogeneous_node_type_zero,
                                sampling_ops=[
                                    self.example_heterogeneous_sampling_op_root_0,
                                ],
                            ),
                            MessagePassingPath(
                                root_node_type=self.heterogeneous_node_type_one,
                                sampling_ops=[
                                    self.example_heterogeneous_sampling_op_root_1,
                                ],
                            ),
                            MessagePassingPath(
                                root_node_type=self.heterogeneous_node_type_two,
                                sampling_ops=[
                                    self.example_heterogeneous_sampling_op_root_2,
                                ],
                            ),
                        ]
                    )
                )
            )
        )
        self.__assert_subgraph_sampling_validate_dags_exception(
            subgraph_sampling_strategy_pb_wrapper=nonexistent_root_node_type_in_task_metadata_dag,
            graph_metadata_pb=EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB,
            task_metadata_pb=EXAMPLE_NABLP_HETEROGENEOUS_TASK_METADATA_PB,
            error_type=SubgraphSamplingValidationErrorType.ROOT_NODE_TYPE_NOT_IN_TASK_METADATA,
        )

    def test_dag_expected_root_node_type_missing_from_task_metadata_exception(
        self,
    ) -> None:
        """
        Test whether exception 'MISSING_EXPECTED_ROOT_NODE_TYPE' is correctly raised
        when expected root node type is missing from dag but specified by task metadata
        """
        missing_root_node_type_in_task_metadata_dag = SubgraphSamplingStrategyPbWrapper(
            SubgraphSamplingStrategy(
                message_passing_paths=MessagePassingPathStrategy(
                    paths=[
                        MessagePassingPath(
                            root_node_type=self.heterogeneous_node_type_zero,
                            sampling_ops=[
                                self.example_heterogeneous_sampling_op_root_0,
                            ],
                        ),
                    ]
                )
            )
        )
        self.__assert_subgraph_sampling_validate_dags_exception(
            subgraph_sampling_strategy_pb_wrapper=missing_root_node_type_in_task_metadata_dag,
            graph_metadata_pb=EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB,
            task_metadata_pb=EXAMPLE_NABLP_HETEROGENEOUS_TASK_METADATA_PB,
            error_type=SubgraphSamplingValidationErrorType.MISSING_EXPECTED_ROOT_NODE_TYPE,
        )

    def test_dag_contains_no_root_sampling_op_exception(self) -> None:
        """
        Test whether exception 'MISSING_ROOT_SAMPLING_OP' is correctly raised when dag contains no root sampling op
        """

        no_root_sampling_op_1 = SamplingOp(
            op_name="no_root_sampling_op_1",
            edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB,
            input_op_names=["no_root_sampling_op_2"],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.OUTGOING,
        )

        no_root_sampling_op_2 = SamplingOp(
            op_name="no_root_sampling_op_2",
            edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB,
            input_op_names=["no_root_sampling_op_1"],
            random_uniform=RandomUniform(num_nodes_to_sample=10),
            sampling_direction=SamplingDirection.OUTGOING,
        )

        no_root_sampling_op_dag = SubgraphSamplingStrategyPbWrapper(
            SubgraphSamplingStrategy(
                message_passing_paths=MessagePassingPathStrategy(
                    paths=[
                        MessagePassingPath(
                            root_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
                            sampling_ops=[
                                no_root_sampling_op_1,
                                no_root_sampling_op_2,
                            ],
                        )
                    ]
                )
            )
        )
        self.__assert_subgraph_sampling_validate_dags_exception(
            subgraph_sampling_strategy_pb_wrapper=no_root_sampling_op_dag,
            graph_metadata_pb=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB,
            task_metadata_pb=DEFAULT_NABLP_HOMOGENEOUS_TASK_METADATA_PB,
            error_type=SubgraphSamplingValidationErrorType.MISSING_ROOT_SAMPLING_OP,
        )

    def test_zero_hop_dag_success(self):
        """
        Test the validation checks pass when dag is a 0-hop, containing no sampling ops
        """

        zero_hop_dag = SubgraphSamplingStrategyPbWrapper(
            SubgraphSamplingStrategy(
                message_passing_paths=MessagePassingPathStrategy(
                    paths=[
                        MessagePassingPath(
                            root_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
                            sampling_ops=[],
                        )
                    ]
                )
            )
        )

        zero_hop_dag.validate_dags(
            graph_metadata_pb=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB,
            task_metadata_pb=DEFAULT_NABLP_HOMOGENEOUS_TASK_METADATA_PB,
        )
