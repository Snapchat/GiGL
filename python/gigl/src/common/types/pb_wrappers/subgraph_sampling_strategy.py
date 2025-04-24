from dataclasses import dataclass
from typing import Dict, List, Set, cast

from gigl.src.common.types.exception import (
    SubgraphSamplingValidationError,
    SubgraphSamplingValidationErrorType,
)
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.sampling_op import SamplingOpPbWrapper
from gigl.src.common.types.pb_wrappers.task_metadata import TaskMetadataPbWrapper
from snapchat.research.gbml.gbml_config_pb2 import GbmlConfig
from snapchat.research.gbml.graph_schema_pb2 import GraphMetadata
from snapchat.research.gbml.subgraph_sampling_strategy_pb2 import (
    GlobalRandomUniformStrategy,
    MessagePassingPath,
    MessagePassingPathStrategy,
    SubgraphSamplingStrategy,
)


@dataclass
class MessagePassingPathPbWrapper:
    message_passing_path_pb: MessagePassingPath

    def __post_init__(self):
        """
        Builds dags using the provided message passing path sampling strategy for graph traversal
        """
        op_name_to_sampling_op_pb_wrapper: Dict[str, SamplingOpPbWrapper] = {}
        root_sampling_op_names: Set[str] = set()
        # Firstly create the raw SamplingOpNodes indexed by the op name and identify root sampling op nodes
        for sampling_op_pb in self.message_passing_path_pb.sampling_ops:
            cur_sampling_op_pb_wrapper = SamplingOpPbWrapper(
                sampling_op_pb=sampling_op_pb
            )

            # Check that each op name is unique within the sampling op dag
            if cur_sampling_op_pb_wrapper.op_name in op_name_to_sampling_op_pb_wrapper:
                raise SubgraphSamplingValidationError(
                    message=f"Found repeated op name {cur_sampling_op_pb_wrapper.op_name} when constructing sampling op, please ensure each op name is unique.",
                    error_type=SubgraphSamplingValidationErrorType.REPEATED_OP_NAME,
                )

            op_name_to_sampling_op_pb_wrapper[
                sampling_op_pb.op_name
            ] = cur_sampling_op_pb_wrapper

            is_root_sampling_op_node = (
                len(cur_sampling_op_pb_wrapper.input_op_names) == 0
            )

            if is_root_sampling_op_node:
                root_sampling_op_names.add(cur_sampling_op_pb_wrapper.op_name)

        # Now create the actual DAG
        for sampling_op_pb in self.message_passing_path_pb.sampling_ops:
            cur_sampling_op_pb_wrapper = op_name_to_sampling_op_pb_wrapper[
                sampling_op_pb.op_name
            ]
            parent_input_op_names = cur_sampling_op_pb_wrapper.input_op_names
            parent_sampling_op_pb_wrappers: List[SamplingOpPbWrapper] = []

            for parent_input_op_name in parent_input_op_names:
                # Check that each input op name maps to a valid sampling op
                if parent_input_op_name not in op_name_to_sampling_op_pb_wrapper:
                    raise SubgraphSamplingValidationError(
                        message=f"Found input op name {parent_input_op_name} that is not the op name of any sampling op.",
                        error_type=SubgraphSamplingValidationErrorType.BAD_INPUT_OP_NAME,
                    )

                parent_sampling_op_pb_wrappers.append(
                    op_name_to_sampling_op_pb_wrapper[parent_input_op_name]
                )

            for parent_sampling_op_pb_wrapper in parent_sampling_op_pb_wrappers:
                parent_sampling_op_pb_wrapper.add_child_sampling_op_pb_wrapper(
                    child_sampling_op_pb_wrapper=cur_sampling_op_pb_wrapper
                )
                cur_sampling_op_pb_wrapper.add_parent_sampling_op_pb_wrapper(
                    parent_sampling_op_pb_wrapper=parent_sampling_op_pb_wrapper
                )

        self.__root_sampling_op_names = list(root_sampling_op_names)
        self.__op_name_to_sampling_op_pb_wrapper = op_name_to_sampling_op_pb_wrapper

    @property
    def root_node_type(self) -> NodeType:
        return NodeType(self.message_passing_path_pb.root_node_type)

    @property
    def root_sampling_op_names(self) -> List[str]:
        return self.__root_sampling_op_names

    @property
    def op_name_to_sampling_op_pb_wrapper(self) -> Dict[str, SamplingOpPbWrapper]:
        return self.__op_name_to_sampling_op_pb_wrapper


@dataclass
class MessagePassingPathStrategyPbWrapper:
    message_passing_path_strategy_pb: MessagePassingPathStrategy

    def __post_init__(self):
        root_node_type_to_message_passing_path_pb_wrapper: Dict[
            NodeType, MessagePassingPathPbWrapper
        ] = {}
        for message_passing_path_pb in self.message_passing_path_strategy_pb.paths:
            message_passing_path_pb_wrapper = MessagePassingPathPbWrapper(
                message_passing_path_pb=message_passing_path_pb
            )
            root_node_type = message_passing_path_pb_wrapper.root_node_type
            # Check that each root node type only has one associated dag
            if root_node_type in root_node_type_to_message_passing_path_pb_wrapper:
                raise SubgraphSamplingValidationError(
                    message=f"Found repeated root node type {root_node_type} when constructing message passing paths, please ensure each MessagePassingPath root node type is unique.",
                    error_type=SubgraphSamplingValidationErrorType.REPEATED_ROOT_NODE_TYPE,
                )
            root_node_type_to_message_passing_path_pb_wrapper[
                root_node_type
            ] = message_passing_path_pb_wrapper

        self.__root_node_type_to_message_passing_path_pb_wrapper = (
            root_node_type_to_message_passing_path_pb_wrapper
        )

    @property
    def root_node_type_to_message_passing_path_pb_wrapper(
        self,
    ) -> Dict[NodeType, MessagePassingPathPbWrapper]:
        return self.__root_node_type_to_message_passing_path_pb_wrapper


@dataclass
class GlobalRandomUniformStrategyPbWrapper:
    global_random_uniform_strategy_pb: GlobalRandomUniformStrategy

    def __post_init__(self):
        # TODO (mkolodner): Implement support for this, need to figure out best way to provide information from graph and task metadata here
        self.__root_node_type_to_message_passing_path_pb_wrapper = dict()

    @property
    def root_node_type_to_message_passing_path_pb_wrapper(
        self,
    ) -> Dict[NodeType, MessagePassingPathPbWrapper]:
        return self.__root_node_type_to_message_passing_path_pb_wrapper


@dataclass
class SubgraphSamplingStrategyPbWrapper:
    subgraph_sampling_strategy_pb: SubgraphSamplingStrategy

    def __post_init__(self) -> None:
        self.__root_node_type_to_message_passing_path_pb_wrapper: Dict[
            NodeType, MessagePassingPathPbWrapper
        ]
        sampling_strategy_field = cast(
            str, self.subgraph_sampling_strategy_pb.WhichOneof("strategy")
        )
        if sampling_strategy_field == "global_random_uniform":
            self.__root_node_type_to_message_passing_path_pb_wrapper = GlobalRandomUniformStrategyPbWrapper(
                global_random_uniform_strategy_pb=self.subgraph_sampling_strategy_pb.global_random_uniform
            ).root_node_type_to_message_passing_path_pb_wrapper
        elif sampling_strategy_field == "message_passing_paths":
            self.__root_node_type_to_message_passing_path_pb_wrapper = MessagePassingPathStrategyPbWrapper(
                message_passing_path_strategy_pb=self.subgraph_sampling_strategy_pb.message_passing_paths
            ).root_node_type_to_message_passing_path_pb_wrapper
        else:
            raise ValueError(
                "Invalid SubgraphSamplingStrategy. Must provide one of GlobalRandomUniform or MessagePassingPaths."
            )

    def validate_dags(
        self,
        graph_metadata_pb: GraphMetadata,
        task_metadata_pb: GbmlConfig.TaskMetadata,
    ) -> None:
        """
        Given the provided gbml_config_pb, validates the correctness for all provided dag, checking for
            - Root node type being present in graph metadata
            - Root node type being a node type that is in a supervision edge or a supervision node type
            - Whether there is at least one message passing traversal path into the root node if it is not a 0-hop traversal
            - Sampling op edge type present in graph metadata
            - Edge types are properly aligned for parent and children sampling ops (i.e. the src node type of parent must be the dst node type of child)
            - Whether any of the provided dags contain cycles
        Args:
            graph_metadata_pb (GraphMetadata): Graph metadata pb for validating correctness
            task_metadata_pb (TaskMetadata): Task metadata pb for validating correctness
        Returns:
            None
        """
        graph_metadata_pb_wrapper = GraphMetadataPbWrapper(
            graph_metadata_pb=graph_metadata_pb
        )
        task_metadata_pb_wrapper = TaskMetadataPbWrapper(
            task_metadata_pb=task_metadata_pb
        )
        expected_root_node_types: Set[
            NodeType
        ] = task_metadata_pb_wrapper.get_task_root_node_types()
        graph_edge_types = graph_metadata_pb_wrapper.edge_types
        graph_node_types = graph_metadata_pb_wrapper.node_types

        for (
            root_node_type,
            message_passing_path_pb_wrapper,
        ) in self.root_node_type_to_message_passing_path_pb_wrapper.items():
            # Check that the root node type of current dag is in the graph metadata
            if root_node_type not in graph_node_types:
                raise SubgraphSamplingValidationError(
                    message=f"Found root node type {root_node_type} not defined in graph metadata: {graph_node_types}.",
                    error_type=SubgraphSamplingValidationErrorType.ROOT_NODE_TYPE_NOT_IN_GRAPH_METADATA,
                )
            # Check that root node type of current dag is specified in task metadata as supervision edge node type or supervision node type
            if root_node_type not in expected_root_node_types:
                raise SubgraphSamplingValidationError(
                    message=f"Found root node type {root_node_type} that is not in a supervision edge or is not a supervision node type.",
                    error_type=SubgraphSamplingValidationErrorType.ROOT_NODE_TYPE_NOT_IN_TASK_METADATA,
                )

            expected_root_node_types.remove(root_node_type)

            # Check that there is at least one sampling op connected to the root node type if it is not a 0-hop DAG
            has_root_sampling_op = (
                len(message_passing_path_pb_wrapper.root_sampling_op_names) > 0
            )
            is_zero_hop = (
                len(message_passing_path_pb_wrapper.op_name_to_sampling_op_pb_wrapper)
                == 0
            )

            if not (is_zero_hop or has_root_sampling_op):
                raise SubgraphSamplingValidationError(
                    message=f"Sampling Op DAG with root node type {root_node_type} has no sampling ops from root node.",
                    error_type=SubgraphSamplingValidationErrorType.MISSING_ROOT_SAMPLING_OP,
                )

            for (
                op_name,
                sampling_op_pb_wrapper,
            ) in (
                message_passing_path_pb_wrapper.op_name_to_sampling_op_pb_wrapper.items()
            ):
                # Check sampling op edge type is defined in graph metadata
                if sampling_op_pb_wrapper.edge_type not in graph_edge_types:
                    raise SubgraphSamplingValidationError(
                        message=f"Found sampling op edge type {sampling_op_pb_wrapper.edge_type} not defined in graph metadata: {graph_edge_types}.",
                        error_type=SubgraphSamplingValidationErrorType.SAMPLING_OP_EDGE_TYPE_NOT_IN_GRAPH_METADATA,
                    )
                # Check sampling op edge type validity in the DAG
                sampling_op_pb_wrapper.check_sampling_op_edge_type_validity(
                    root_node_type=root_node_type,
                )
            # Check if we have any cycles in the current dag
            visited_sampling_op_names: Set[str] = set()
            cycle_sampling_op_names: List[str] = []
            for (
                root_sampling_op_name
            ) in message_passing_path_pb_wrapper.root_sampling_op_names:
                current_sampling_op_pb_wrapper = (
                    message_passing_path_pb_wrapper.op_name_to_sampling_op_pb_wrapper[
                        root_sampling_op_name
                    ]
                )
                if current_sampling_op_pb_wrapper.check_if_dag_contains_cycles(
                    visited_sampling_op_names=visited_sampling_op_names,
                    recursing_sampling_op_names=set(),
                    cycle_sampling_op_names=cycle_sampling_op_names,
                ):
                    formatted_cycle_path = (
                        f'[LEAF] {"->".join(cycle_sampling_op_names)} [ROOT]'
                    )
                    raise SubgraphSamplingValidationError(
                        message=f"Detected a cycle in provided dag in message passing traversal path {formatted_cycle_path}",
                        error_type=SubgraphSamplingValidationErrorType.DAG_CONTAINS_CYCLE,
                    )

        # Check that all root node types specified by task metadata have associated dags
        if len(expected_root_node_types) > 0:
            raise SubgraphSamplingValidationError(
                message=f"Found root node types {expected_root_node_types} with no Sampling Op DAG.",
                error_type=SubgraphSamplingValidationErrorType.MISSING_EXPECTED_ROOT_NODE_TYPE,
            )

    @property
    def root_node_type_to_message_passing_path_pb_wrapper(
        self,
    ) -> Dict[NodeType, MessagePassingPathPbWrapper]:
        """
        Returns a mapping of each root node type to their respective sampling op dag
        """

        return self.__root_node_type_to_message_passing_path_pb_wrapper
