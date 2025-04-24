from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union, cast

from gigl.src.common.types.exception import (
    SubgraphSamplingValidationError,
    SubgraphSamplingValidationErrorType,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from snapchat.research.gbml.subgraph_sampling_strategy_pb2 import (
    RandomUniform,
    RandomWeighted,
    SamplingDirection,
    SamplingOp,
    TopK,
    UserDefined,
)


class SamplingDirectionEnumType(Enum):
    INCOMING = 0
    OUTGOING = 1

    @classmethod
    def get_all_directions(cls) -> List[str]:
        return [m.name for m in cls]


@dataclass
class RandomUniformPbWrapper:
    random_uniform_pb: RandomUniform

    def __post_init__(self):
        assert (
            self.random_uniform_pb.num_nodes_to_sample > 0
        ), f"Found invalid {RandomUniformPbWrapper.__name__} num_nodes_to_sample: {self.num_nodes_to_sample}. Must be > 0."

    @property
    def num_nodes_to_sample(self) -> int:
        return self.random_uniform_pb.num_nodes_to_sample


@dataclass
class RandomWeightedPbWrapper:
    random_weighted_pb: RandomWeighted

    def __post_init__(self):
        assert (
            self.random_weighted_pb.num_nodes_to_sample > 0
        ), f"Found invalid {RandomWeightedPbWrapper.__name__} num_nodes_to_sample: {self.num_nodes_to_sample}. Must be > 0."
        assert (
            self.random_weighted_pb.edge_feat_name
        ), f"Found invalid {RandomWeightedPbWrapper.__name__} edge_feat_name: {self.edge_feat_name}. Must be non-empty."

    @property
    def num_nodes_to_sample(self) -> int:
        return self.random_weighted_pb.num_nodes_to_sample

    @property
    def edge_feat_name(self) -> str:
        return self.random_weighted_pb.edge_feat_name


@dataclass
class TopKPbWrapper:
    top_k_pb: TopK

    def __post_init__(self):
        assert (
            self.top_k_pb.num_nodes_to_sample > 0
        ), f"Found invalid {TopKPbWrapper.__name__} num_nodes_to_sample: {self.num_nodes_to_sample}. Must be > 0."
        assert (
            self.top_k_pb.edge_feat_name
        ), f"Found invalid {TopKPbWrapper.__name__} edge_feat_name: {self.edge_feat_name}. Must be non-empty."

    @property
    def num_nodes_to_sample(self) -> int:
        return self.top_k_pb.num_nodes_to_sample

    @property
    def edge_feat_name(self) -> str:
        return self.top_k_pb.edge_feat_name


@dataclass
class UserDefinedPbWrapper:
    user_defined_pb: UserDefined

    def __post_init__(self):
        # TODO (mkolodner): Update check to validate import once this interface is defined
        assert (
            self.path_to_udf
        ), f"Found invalid {UserDefinedPbWrapper.__name__} path_to_udf: {self.path_to_udf}. Must be non-empty."

    @property
    def path_to_udf(self) -> str:
        return self.user_defined_pb.path_to_udf

    @property
    def params(self) -> Dict[str, str]:
        return dict(self.user_defined_pb.params)


@dataclass
class SamplingOpPbWrapper:
    sampling_op_pb: SamplingOp = field(repr=False)

    _hash: int = field(init=False)

    def __post_init__(self):
        sampling_method_field = cast(
            str, self.sampling_op_pb.WhichOneof("sampling_method")
        )
        sampling_method_pb = getattr(self.sampling_op_pb, sampling_method_field)
        self.__sampling_method: Union[
            RandomUniformPbWrapper,
            RandomWeightedPbWrapper,
            TopKPbWrapper,
            UserDefinedPbWrapper,
        ]
        if sampling_method_field == "random_uniform":
            self.__sampling_method = RandomUniformPbWrapper(
                random_uniform_pb=sampling_method_pb
            )
        elif sampling_method_field == "random_weighted":
            self.__sampling_method = RandomWeightedPbWrapper(
                random_weighted_pb=sampling_method_pb
            )
        elif sampling_method_field == "top_k":
            self.__sampling_method = TopKPbWrapper(top_k_pb=sampling_method_pb)
        elif sampling_method_field == "user_defined":
            self.__sampling_method = UserDefinedPbWrapper(
                user_defined_pb=sampling_method_pb
            )
        else:
            raise ValueError(
                f"Invalid sampling method found. Must specify one of {RandomUniform.__name__, RandomWeighted.__name__, TopK.__name__, UserDefined.__name__}"
            )

        self.__child_sampling_op_pb_wrappers = set()
        self.__parent_sampling_op_pb_wrappers = set()

        # Convert the sampling op pb wrapper to a hashable type and use its hash
        self._hash = hash(self.sampling_op_pb.op_name)

    def __hash__(self) -> int:
        """
        Use the pre-computed hash for this sampling op pb wrapper
        """
        return self._hash

    def __reduce__(self):
        proto_serialized = self.sampling_op_pb.SerializeToString()

        return (self.from_serialized, (proto_serialized,))

    @classmethod
    def from_serialized(cls, proto_serialized):
        # Deserialize the Protobuf message from a string
        sampling_op_pb = SamplingOp()
        sampling_op_pb.ParseFromString(proto_serialized)
        return cls(sampling_op_pb=sampling_op_pb)

    @property
    def op_name(self) -> str:
        return self.sampling_op_pb.op_name

    @property
    def edge_type(self) -> EdgeType:
        return EdgeType(
            src_node_type=NodeType(self.sampling_op_pb.edge_type.src_node_type),
            relation=Relation(self.sampling_op_pb.edge_type.relation),
            dst_node_type=NodeType(self.sampling_op_pb.edge_type.dst_node_type),
        )

    @property
    def sampling_direction(self) -> SamplingDirectionEnumType:
        return SamplingDirectionEnumType[
            SamplingDirection.Name(self.sampling_op_pb.sampling_direction)
        ]

    @property
    def sampling_method(
        self,
    ) -> Union[
        RandomUniformPbWrapper,
        RandomWeightedPbWrapper,
        TopKPbWrapper,
        UserDefinedPbWrapper,
    ]:
        return self.__sampling_method

    @property
    def input_op_names(self) -> List[str]:
        return list(self.sampling_op_pb.input_op_names)

    @property
    def child_sampling_op_pb_wrappers(self) -> Set[SamplingOpPbWrapper]:
        """
        Uses the input_op_names field to infer children samplings ops from the current, populated manually with
        the add_child_sampling_op_pb_wrapper class method
        """
        return self.__child_sampling_op_pb_wrappers

    @property
    def parent_sampling_op_pb_wrappers(self) -> Set[SamplingOpPbWrapper]:
        """
        Uses the input_op_names field to infer parent samplings ops from the current, populated manually with
        the add_parent_sampling_op_pb_wrapper class method
        """
        return self.__parent_sampling_op_pb_wrappers

    def add_child_sampling_op_pb_wrapper(
        self, child_sampling_op_pb_wrapper: SamplingOpPbWrapper
    ) -> None:
        """
        Adds a child sampling op pb wrapper to the set of children sampling ops
        """
        self.__child_sampling_op_pb_wrappers.add(child_sampling_op_pb_wrapper)

    def add_parent_sampling_op_pb_wrapper(
        self, parent_sampling_op_pb_wrapper: SamplingOpPbWrapper
    ) -> None:
        """
        Adds a parent sampling op pb wrapper to the set of parent sampling ops
        """
        self.__parent_sampling_op_pb_wrappers.add(parent_sampling_op_pb_wrapper)

    def __get_sampling_error_msg(
        self,
        child_node_type: NodeType,
        parent_node_type: NodeType,
        child_sampling_op_direction: str,
        parent_sampling_op_direction: Optional[str] = None,
    ):
        # Case where parent is the root node
        if child_sampling_op_direction == SamplingDirectionEnumType.INCOMING.name:
            child_node_location = "dst"
        else:
            child_node_location = "src"
        if parent_sampling_op_direction is None:
            return (
                f"Found root sampling op node {self.op_name} with edge {child_node_location} node type {child_node_type} and root node type {parent_node_type}. "
                f"These must be equal for root sampling op sampling direction {child_sampling_op_direction}"
            )
        else:
            if parent_sampling_op_direction == SamplingDirectionEnumType.INCOMING.name:
                parent_node_location = "src"
            else:
                parent_node_location = "dst"

            return (
                f"Found child sampling op node {self.op_name} with edge {child_node_location} node type {child_node_type} "
                f"and parent sampling op with edge {parent_node_location} node type {parent_node_type}. "
                f"These two must be equal for child sampling direction {child_sampling_op_direction} "
                f"and parent sampling direction {parent_sampling_op_direction}."
            )

    def check_sampling_op_edge_type_validity(
        self, root_node_type: Optional[NodeType] = None
    ) -> None:
        """
        All Sampling Op DAGs are centered around some root node, performing some k-hop traversal around this node. A parent sampling op is a sampling op
        which is closer to the root node than the current sampling op, being earlier in the traversal. Parents are defined by the input_op_names field, and
        a sampling op with no parent sampling op is moving toward a 1-hop distance from the root node. Likewise, a child sampling op is a sampling op which comes
        logically later the in the traversal, being farther away from the root node than the current sampling op. Sampling Ops also have a specified direction,
        listed as INCOMING or OUTGOING. This indicates what direction the traversal is going. For example, an INCOMING sampling direction means we are
        moving towards the root node, while an OUTGOING sampling direction means we are moving away from the root node. Finally, edge types are defined
        by some source node, relation, and destination node. We always move from source node to destination node.

        As an example, an edge type with source node type '0' and destination node type '1' will always go from '0' to '1'. If it is INCOMING,
        this means the current sampling op must expect to receive a '0' from any children sampling ops and will provide a '1' to any parent sampling
        ops. If it is OUTGOING, this means the current sampling will provide a '1' to any children sampling ops and must receive a '0' from any parents.

        Given this context, this function validates that
        - If the sampling op is a root sampling op, the sampling op direction and edge type align with the root node type. Specifically:
            - Sampling Direction = INCOMING -> sampling_op.edge_type.dst_node_type = root_node_type
            - Sampling Direction = OUTGOING -> sampling_op.edge_type.src_node_type = root_node_type
        - Otherwise, the child's sampling op direction and edge type align with each parent's sampling op direction and edge type. Specifically:
            - Child Sampling Direction = INCOMING, Parent Sampling Direction = INCOMING ->
                child_sampling_op.edge_type.dst_node_type = parent_sampling_op.edge_type.src_node_type

            - Child Sampling Direction = INCOMING, Parent Sampling Direction = OUTGOING ->
                child_sampling_op.edge_type.dst_node_type = parent_sampling_op.edge_type.dst_node_type

            - Child Sampling Direction = OUTGOING, Parent Sampling Direction = INCOMING ->
                child_sampling_op.edge_type.src_node_type = parent_sampling_op.edge_type.src_node_type

            - Child Sampling Direction = OUTGOING, Parent Sampling Direction = OUTGOING ->
                child_sampling_op.edge_type.src_node_type = parent_sampling_op.edge_type.dst_node_type

        Args:
            root_node_type (Optional[NodeType]): The root node type to check validity of root sampling ops, not required or used for non-root sampling ops
        """

        is_root_sampling_op_node = len(self.input_op_names) == 0
        sampling_op_direction = self.sampling_direction.name
        sampling_op_edge_type = self.edge_type

        if is_root_sampling_op_node:
            if root_node_type is None:
                raise ValueError(
                    "Root node type must be specified for checking root sampling op validity"
                )
            if sampling_op_direction == SamplingDirectionEnumType.INCOMING.name:
                if sampling_op_edge_type.dst_node_type != root_node_type:
                    raise SubgraphSamplingValidationError(
                        message=self.__get_sampling_error_msg(
                            child_node_type=sampling_op_edge_type.dst_node_type,
                            parent_node_type=root_node_type,
                            child_sampling_op_direction=sampling_op_direction,
                        ),
                        error_type=SubgraphSamplingValidationErrorType.CONTAINS_INVALID_EDGE_IN_DAG,
                    )
            elif sampling_op_direction == SamplingDirectionEnumType.OUTGOING.name:
                if sampling_op_edge_type.src_node_type != root_node_type:
                    raise SubgraphSamplingValidationError(
                        self.__get_sampling_error_msg(
                            child_node_type=sampling_op_edge_type.src_node_type,
                            parent_node_type=root_node_type,
                            child_sampling_op_direction=sampling_op_direction,
                        ),
                        error_type=SubgraphSamplingValidationErrorType.CONTAINS_INVALID_EDGE_IN_DAG,
                    )
            else:
                raise ValueError(
                    f"Required root sampling op direction to be in {SamplingDirectionEnumType.get_all_directions()}, got {sampling_op_direction}."
                )
        else:
            for parent_sampling_op_pb_wrapper in self.parent_sampling_op_pb_wrappers:
                parent_sampling_op_direction = (
                    parent_sampling_op_pb_wrapper.sampling_direction.name
                )
                parent_sampling_op_edge_type = parent_sampling_op_pb_wrapper.edge_type

                if (
                    sampling_op_direction == SamplingDirectionEnumType.INCOMING.name
                    and parent_sampling_op_direction
                    == SamplingDirectionEnumType.INCOMING.name
                ):
                    if (
                        sampling_op_edge_type.dst_node_type
                        != parent_sampling_op_edge_type.src_node_type
                    ):
                        raise SubgraphSamplingValidationError(
                            self.__get_sampling_error_msg(
                                child_node_type=sampling_op_edge_type.dst_node_type,
                                parent_node_type=parent_sampling_op_edge_type.src_node_type,
                                child_sampling_op_direction=sampling_op_direction,
                                parent_sampling_op_direction=parent_sampling_op_direction,
                            ),
                            error_type=SubgraphSamplingValidationErrorType.CONTAINS_INVALID_EDGE_IN_DAG,
                        )
                elif (
                    sampling_op_direction == SamplingDirectionEnumType.INCOMING.name
                    and parent_sampling_op_direction
                    == SamplingDirectionEnumType.OUTGOING.name
                ):
                    if (
                        sampling_op_edge_type.dst_node_type
                        != parent_sampling_op_edge_type.dst_node_type
                    ):
                        raise SubgraphSamplingValidationError(
                            message=self.__get_sampling_error_msg(
                                child_node_type=sampling_op_edge_type.dst_node_type,
                                parent_node_type=parent_sampling_op_edge_type.dst_node_type,
                                child_sampling_op_direction=sampling_op_direction,
                                parent_sampling_op_direction=parent_sampling_op_direction,
                            ),
                            error_type=SubgraphSamplingValidationErrorType.CONTAINS_INVALID_EDGE_IN_DAG,
                        )
                elif (
                    sampling_op_direction == SamplingDirectionEnumType.OUTGOING.name
                    and parent_sampling_op_direction
                    == SamplingDirectionEnumType.INCOMING.name
                ):
                    if (
                        sampling_op_edge_type.src_node_type
                        != parent_sampling_op_edge_type.src_node_type
                    ):
                        raise SubgraphSamplingValidationError(
                            message=self.__get_sampling_error_msg(
                                child_node_type=sampling_op_edge_type.src_node_type,
                                parent_node_type=parent_sampling_op_edge_type.src_node_type,
                                child_sampling_op_direction=sampling_op_direction,
                                parent_sampling_op_direction=parent_sampling_op_direction,
                            ),
                            error_type=SubgraphSamplingValidationErrorType.CONTAINS_INVALID_EDGE_IN_DAG,
                        )
                elif (
                    sampling_op_direction == SamplingDirectionEnumType.OUTGOING.name
                    and parent_sampling_op_direction
                    == SamplingDirectionEnumType.OUTGOING.name
                ):
                    if (
                        sampling_op_edge_type.src_node_type
                        != parent_sampling_op_edge_type.dst_node_type
                    ):
                        raise SubgraphSamplingValidationError(
                            self.__get_sampling_error_msg(
                                child_node_type=sampling_op_edge_type.src_node_type,
                                parent_node_type=parent_sampling_op_edge_type.dst_node_type,
                                child_sampling_op_direction=sampling_op_direction,
                                parent_sampling_op_direction=parent_sampling_op_direction,
                            ),
                            error_type=SubgraphSamplingValidationErrorType.CONTAINS_INVALID_EDGE_IN_DAG,
                        )
                else:
                    raise ValueError(
                        (
                            f"Required child and parent sampling op directions to be in {SamplingDirectionEnumType.get_all_directions()}, "
                            f"got {sampling_op_direction} and {parent_sampling_op_direction}."
                        )
                    )

    def check_if_dag_contains_cycles(
        self,
        visited_sampling_op_names: Set[str],
        recursing_sampling_op_names: Set[str],
        cycle_sampling_op_names: List[str],
    ) -> bool:
        """
        Recursively checks if provided dag contains cycles
        Args:
            visited_sampling_op_names (Set[str]): set of sampling op names that have been fully visited and verified to not have a cycle
            recursing_sampling_op_names (Set[str]): set of sampling op names that are currently being explored recursively in DFS
            cycle_sampling_op_names (List[str]): Populated with list of sampling op node names in message passing traversal path where a cycle is found, empty if there is no cycle.
        Returns:
            bool: Whether there is a cycle from the sampling op
        """
        if self.op_name in recursing_sampling_op_names:
            cycle_sampling_op_names.append(self.op_name)
            return True
        elif self.op_name in visited_sampling_op_names:
            return False

        visited_sampling_op_names.add(self.op_name)
        recursing_sampling_op_names.add(self.op_name)

        for child_sampling_op_pb_wrapper in self.child_sampling_op_pb_wrappers:
            if child_sampling_op_pb_wrapper.check_if_dag_contains_cycles(
                visited_sampling_op_names=visited_sampling_op_names,
                recursing_sampling_op_names=recursing_sampling_op_names,
                cycle_sampling_op_names=cycle_sampling_op_names,
            ):
                cycle_sampling_op_names.append(self.op_name)
                return True

        recursing_sampling_op_names.remove(self.op_name)
        return False
