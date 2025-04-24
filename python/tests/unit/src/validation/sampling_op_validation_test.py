import unittest
from typing import Optional

from gigl.common.logger import Logger
from gigl.src.common.types.exception import (
    SubgraphSamplingValidationError,
    SubgraphSamplingValidationErrorType,
)
from gigl.src.common.types.pb_wrappers.graph_metadata import NodeType
from gigl.src.common.types.pb_wrappers.sampling_op import SamplingOpPbWrapper
from snapchat.research.gbml.subgraph_sampling_strategy_pb2 import (
    RandomUniform,
    SamplingDirection,
    SamplingOp,
)
from tests.test_assets.graph_metadata_constants import (
    EXAMPLE_HETEROGENEOUS_EDGE_TYPES,
    EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR,
)

logger = Logger()


class SamplingOpValidationUnitTest(unittest.TestCase):
    """
    These tests check the Sampling Op Pb Wrapper validation check for logic specific to a Sampling Op and its immediate Parent/Children Sampling Ops.

    For more information on Sampling Ops and the tests being evaluated for, see the SamplingOpPbWrapper 'check_sampling_op_edge_type_validity' function.
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

    def tearDown(self) -> None:
        pass

    def __assert_sampling_op_edge_type_validity_exception(
        self,
        sampling_op: SamplingOpPbWrapper,
        root_node_type: Optional[NodeType] = None,
    ):
        """
        Helper function to receive exception if specified sampling op is not valid for containing invalid edge
        Args:
            sampling_op (SamplingOpPbWrapper): Current sampling op to validate
            root_node_type (Optional[NodeType]): Root node type if sampling op specified is a root sampling op
        """
        with self.assertRaises(
            SubgraphSamplingValidationError
        ) as subgraph_sampling_exception:
            sampling_op.check_sampling_op_edge_type_validity(
                root_node_type=root_node_type
            )
        self.assertEqual(
            subgraph_sampling_exception.exception.error_type,
            SubgraphSamplingValidationErrorType.CONTAINS_INVALID_EDGE_IN_DAG,
        )

    def test_incoming_root_sampling_op_validity(self):
        """
        If Sampling Direction = INCOMING and we are at a root sampling op (connected to root node),
        we require that sampling_op.edge_type.dst_node_type = root_node_type. In the below example, the incoming root_sampling_op
        has edge type '0' -> '1', so we require that root node type be '1'.
        """
        incoming_root_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="incoming_root_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_one,
                input_op_names=[],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.INCOMING,
            )
        )

        self.__assert_sampling_op_edge_type_validity_exception(
            sampling_op=incoming_root_sampling_op,
            root_node_type=NodeType(self.heterogeneous_node_type_zero),
        )
        incoming_root_sampling_op.check_sampling_op_edge_type_validity(
            root_node_type=NodeType(self.heterogeneous_node_type_one)
        )

    def test_outgoing_root_sampling_op_validity(self):
        """
        If Sampling Direction = OUTGOING and we are at a root sampling op (connected to root node),
        we require that sampling_op.edge_type.src_node_type = root_node_type. In the below example, the incoming root_sampling_op
        has edge type '0' -> '1', so we require that root node type be '0'.
        """
        outgoing_root_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="outgoing_root_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_one,
                input_op_names=[],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.OUTGOING,
            )
        )
        self.__assert_sampling_op_edge_type_validity_exception(
            sampling_op=outgoing_root_sampling_op,
            root_node_type=NodeType(self.heterogeneous_node_type_one),
        )

        outgoing_root_sampling_op.check_sampling_op_edge_type_validity(
            root_node_type=NodeType(self.heterogeneous_node_type_zero)
        )

    def test_incoming_child_incoming_parent_validity(self):
        """
        If Child Sampling Direction = INCOMING and Parent Sampling Direction = INCOMING, we require that
        child_sampling_op.edge_type.dst_node_type = parent_sampling_op.edge_type.src_node_type. In the below example,
        in the correct case, we have child sampling op with edge '0' -> '1' and parent sampling op with edge '1' -> '2', which
        is correct since '1' == '1'. In the incorrect case, we have child sampling op with edge '0' -> '1' and parent sampling op
        with edge '0' -> '2', indicating failure since '1' != '0'
        """
        # Correct Case
        correct_ic_ip_parent_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="correct_ic_ip_parent_sampling_op",
                edge_type=self.heterogeneous_edge_one_to_two,  # 1 -> 2
                input_op_names=[],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.INCOMING,
            )
        )

        correct_ic_ip_base_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="correct_ic_ip_base_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_one,  # 0 -> 1
                input_op_names=["correct_ic_ip_parent_sampling_op"],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.INCOMING,
            )
        )

        correct_ic_ip_base_sampling_op.add_parent_sampling_op_pb_wrapper(
            correct_ic_ip_parent_sampling_op
        )
        correct_ic_ip_parent_sampling_op.add_child_sampling_op_pb_wrapper(
            correct_ic_ip_base_sampling_op
        )

        # Incorrect Case

        incorrect_ic_ip_parent_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="incorrect_ic_ip_parent_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_two,
                input_op_names=[],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.INCOMING,
            )
        )

        incorrect_ic_ip_base_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="incorrect_ic_ip_base_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_one,
                input_op_names=["incorrect_ic_ip_parent_sampling_op"],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.INCOMING,
            )
        )

        incorrect_ic_ip_base_sampling_op.add_parent_sampling_op_pb_wrapper(
            incorrect_ic_ip_parent_sampling_op
        )
        incorrect_ic_ip_parent_sampling_op.add_child_sampling_op_pb_wrapper(
            incorrect_ic_ip_base_sampling_op
        )

        self.__assert_sampling_op_edge_type_validity_exception(
            sampling_op=incorrect_ic_ip_base_sampling_op,
        )

        correct_ic_ip_base_sampling_op.check_sampling_op_edge_type_validity()

    def test_incoming_child_outgoing_parent_validity(self):
        """
        If Child Sampling Direction = INCOMING and Parent Sampling Direction = OUTGOING, we require that
        child_sampling_op.edge_type.dst_node_type = parent_sampling_op.edge_type.dst_node_type. In the below example,
        in the correct case, we have child sampling op with edge '0' -> '2' and parent sampling op with edge '1' -> '2', which
        is correct since '2' == '2'. In the incorrect case, we have child sampling op with edge '0' -> '1' and parent sampling op
        with edge '0' -> '2', indicating failure since '1' != '2'
        """
        # Correct Case
        correct_ic_op_parent_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="correct_ic_op_parent_sampling_op",
                edge_type=self.heterogeneous_edge_one_to_two,
                input_op_names=[],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.OUTGOING,
            )
        )

        correct_ic_op_base_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="correct_ic_op_base_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_two,
                input_op_names=["correct_ic_op_parent_sampling_op"],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.INCOMING,
            )
        )

        correct_ic_op_base_sampling_op.add_parent_sampling_op_pb_wrapper(
            correct_ic_op_parent_sampling_op
        )
        correct_ic_op_parent_sampling_op.add_child_sampling_op_pb_wrapper(
            correct_ic_op_base_sampling_op
        )

        # Incorrect Case

        incorrect_ic_op_parent_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="incorrect_ic_op_parent_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_two,
                input_op_names=[],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.OUTGOING,
            )
        )

        incorrect_ic_op_base_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="incorrect_ic_op_base_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_one,
                input_op_names=["incorrect_ic_op_parent_sampling_op"],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.INCOMING,
            )
        )

        incorrect_ic_op_base_sampling_op.add_parent_sampling_op_pb_wrapper(
            incorrect_ic_op_parent_sampling_op
        )
        incorrect_ic_op_parent_sampling_op.add_child_sampling_op_pb_wrapper(
            incorrect_ic_op_base_sampling_op
        )

        self.__assert_sampling_op_edge_type_validity_exception(
            sampling_op=incorrect_ic_op_base_sampling_op,
        )

        correct_ic_op_base_sampling_op.check_sampling_op_edge_type_validity()

    def test_outgoing_child_incoming_parent_validity(self):
        """
        If Child Sampling Direction = OUTGOING and Parent Sampling Direction = INCOMING, we require that
        child_sampling_op.edge_type.src_node_type = parent_sampling_op.edge_type.src_node_type. In the below example,
        in the correct case, we have child sampling op with edge '0' -> '1' and parent sampling op with edge '0' -> '2', which
        is correct since '0' == '0'. In the incorrect case, we have child sampling op with edge '0' -> '2' and parent sampling op
        with edge '1' -> '2', indicating failure since '0' != '1'
        """
        # Correct Case
        correct_oc_ip_parent_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="correct_oc_ip_parent_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_two,
                input_op_names=[],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.INCOMING,
            )
        )

        correct_oc_ip_base_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="correct_oc_ip_base_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_one,
                input_op_names=["correct_oc_ip_parent_sampling_op"],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.OUTGOING,
            )
        )

        correct_oc_ip_base_sampling_op.add_parent_sampling_op_pb_wrapper(
            correct_oc_ip_parent_sampling_op
        )
        correct_oc_ip_parent_sampling_op.add_child_sampling_op_pb_wrapper(
            correct_oc_ip_base_sampling_op
        )

        # Incorrect Case

        incorrect_oc_ip_parent_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="incorrect_oc_ip_parent_sampling_op",
                edge_type=self.heterogeneous_edge_one_to_two,
                input_op_names=[],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.INCOMING,
            )
        )

        incorrect_oc_ip_base_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="incorrect_oc_ip_base_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_two,
                input_op_names=["incorrect_oc_ip_parent_sampling_op"],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.OUTGOING,
            )
        )

        incorrect_oc_ip_base_sampling_op.add_parent_sampling_op_pb_wrapper(
            incorrect_oc_ip_parent_sampling_op
        )
        incorrect_oc_ip_parent_sampling_op.add_child_sampling_op_pb_wrapper(
            incorrect_oc_ip_base_sampling_op
        )

        self.__assert_sampling_op_edge_type_validity_exception(
            sampling_op=incorrect_oc_ip_base_sampling_op,
        )

        correct_oc_ip_base_sampling_op.check_sampling_op_edge_type_validity()

    def test_outgoing_child_outgoing_parent_validity(self):
        """
        If Child Sampling Direction = OUTGOING and Parent Sampling Direction = OUTGOING, we require that
        child_sampling_op.edge_type.src_node_type = parent_sampling_op.edge_type.dst_node_type. In the below example,
        in the correct case, we have child sampling op with edge '1' -> '2' and parent sampling op with edge '0' -> '1', which
        is correct since '1' == '1'. In the incorrect case, we have child sampling op with edge '0' -> '2' and parent sampling op
        with edge '0' -> '1', indicating failure since '0' != '1'
        """
        # Correct Case
        correct_oc_op_parent_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="correct_oc_op_parent_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_one,
                input_op_names=[],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.OUTGOING,
            )
        )

        correct_oc_op_base_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="correct_oc_op_base_sampling_op",
                edge_type=self.heterogeneous_edge_one_to_two,
                input_op_names=["correct_oc_op_parent_sampling_op"],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.OUTGOING,
            )
        )

        correct_oc_op_base_sampling_op.add_parent_sampling_op_pb_wrapper(
            correct_oc_op_parent_sampling_op
        )
        correct_oc_op_parent_sampling_op.add_child_sampling_op_pb_wrapper(
            correct_oc_op_base_sampling_op
        )

        # Incorrect Case

        incorrect_oc_op_parent_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="incorrect_oc_op_parent_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_one,
                input_op_names=[],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.OUTGOING,
            )
        )

        incorrect_oc_op_base_sampling_op = SamplingOpPbWrapper(
            SamplingOp(
                op_name="incorrect_oc_op_base_sampling_op",
                edge_type=self.heterogeneous_edge_zero_to_two,
                input_op_names=["incorrect_oc_op_parent_sampling_op"],
                random_uniform=RandomUniform(num_nodes_to_sample=10),
                sampling_direction=SamplingDirection.OUTGOING,
            )
        )

        incorrect_oc_op_base_sampling_op.add_parent_sampling_op_pb_wrapper(
            incorrect_oc_op_parent_sampling_op
        )
        incorrect_oc_op_parent_sampling_op.add_child_sampling_op_pb_wrapper(
            incorrect_oc_op_base_sampling_op
        )
        self.__assert_sampling_op_edge_type_validity_exception(
            sampling_op=incorrect_oc_op_base_sampling_op,
        )
        correct_oc_op_base_sampling_op.check_sampling_op_edge_type_validity()
