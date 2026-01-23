from enum import Enum


class SubgraphSamplingValidationErrorType(Enum):
    REPEATED_OP_NAME = "REPEATED_OP_NAME"
    BAD_INPUT_OP_NAME = "BAD_INPUT_OP_NAME"
    DAG_CONTAINS_CYCLE = "DAG_CONTAINS_CYCLE"
    REPEATED_ROOT_NODE_TYPE = "REPEATED_ROOT_NODE_TYPE"
    ROOT_NODE_TYPE_NOT_IN_GRAPH_METADATA = "ROOT_NODE_TYPE_NOT_IN_GRAPH_METADATA"
    ROOT_NODE_TYPE_NOT_IN_TASK_METADATA = "ROOT_NODE_TYPE_NOT_IN_TASK_METADATA"
    MISSING_ROOT_SAMPLING_OP = "MISSING_ROOT_SAMPLING_OP"
    SAMPLING_OP_EDGE_TYPE_NOT_IN_GRAPH_METADATA = (
        "SAMPLING_OP_EDGE_TYPE_NOT_IN_GRAPH_METADATA"
    )
    MISSING_EXPECTED_ROOT_NODE_TYPE = "MISSING_EXPECTED_ROOT_NODE_TYPE"
    CONTAINS_INVALID_EDGE_IN_DAG = "CONTAINS_INVALID_EDGE_IN_DAG"


class SubgraphSamplingValidationError(Exception):
    def __init__(self, message: str, error_type: SubgraphSamplingValidationErrorType):
        super().__init__(message)
        self.message = message
        self.error_type = error_type

    def __str__(self):
        return f"{self.message} (Error Type: {self.error_type.value})"
