from gigl.src.common.constants.graph_metadata import (
    DEFAULT_CONDENSED_EDGE_TYPE,
    DEFAULT_CONDENSED_NODE_TYPE,
)
from gigl.src.common.types.graph_data import CondensedEdgeType, CondensedNodeType
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.common.types.pb_wrappers.task_metadata import TaskMetadataPbWrapper
from snapchat.research.gbml import (
    gbml_config_pb2,
    graph_schema_pb2,
    preprocessed_metadata_pb2,
)

DEFAULT_HOMOGENEOUS_NODE_TYPE_STR = str(DEFAULT_CONDENSED_NODE_TYPE)

DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB = graph_schema_pb2.EdgeType(
    src_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
    relation=str(DEFAULT_CONDENSED_EDGE_TYPE),
    dst_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE_STR,
)

DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB = graph_schema_pb2.GraphMetadata(
    node_types=[DEFAULT_HOMOGENEOUS_NODE_TYPE_STR],
    edge_types=[DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB],
    condensed_node_type_map={
        DEFAULT_CONDENSED_NODE_TYPE: DEFAULT_HOMOGENEOUS_NODE_TYPE_STR
    },
    condensed_edge_type_map={
        DEFAULT_CONDENSED_EDGE_TYPE: DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB
    },
)


DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB = preprocessed_metadata_pb2.PreprocessedMetadata(
    condensed_node_type_to_preprocessed_metadata={
        DEFAULT_CONDENSED_NODE_TYPE: preprocessed_metadata_pb2.PreprocessedMetadata.NodeMetadataOutput()
    },
    condensed_edge_type_to_preprocessed_metadata={
        DEFAULT_CONDENSED_EDGE_TYPE: preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataOutput(
            main_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
            positive_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
            negative_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
        )
    },
)

DEFAULT_NABLP_HOMOGENEOUS_TASK_METADATA_PB = gbml_config_pb2.GbmlConfig.TaskMetadata(
    node_anchor_based_link_prediction_task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata(
        supervision_edge_types=[DEFAULT_HOMOGENEOUS_EDGE_TYPE_PB]
    )
)

DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER = GraphMetadataPbWrapper(
    graph_metadata_pb=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB
)


DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER = PreprocessedMetadataPbWrapper(
    preprocessed_metadata_pb=DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB
)

DEFAULT_NABLP_HOMOGENEOUS_TASK_METADATA_PB_WRAPPER = TaskMetadataPbWrapper(
    task_metadata_pb=DEFAULT_NABLP_HOMOGENEOUS_TASK_METADATA_PB
)

EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES = [
    CondensedNodeType(0),
    CondensedNodeType(1),
    CondensedNodeType(2),
]
EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR = [
    str(condensed_node_type)
    for condensed_node_type in EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES
]

EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES = [
    CondensedEdgeType(0),
    CondensedEdgeType(1),
    CondensedEdgeType(2),
]
EXAMPLE_HETEROGENEOUS_EDGE_TYPES = [
    graph_schema_pb2.EdgeType(
        src_node_type=EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[0],
        relation=str(EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[0]),
        dst_node_type=EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[1],
    ),
    graph_schema_pb2.EdgeType(
        src_node_type=EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[0],
        relation=str(EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[1]),
        dst_node_type=EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[2],
    ),
    graph_schema_pb2.EdgeType(
        src_node_type=EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[1],
        relation=str(EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[2]),
        dst_node_type=EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[2],
    ),
]


EXAMPLE_HETEROGENEOUS_PREPROCESSED_METADATA_PB = preprocessed_metadata_pb2.PreprocessedMetadata(
    condensed_node_type_to_preprocessed_metadata={
        EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES[
            0
        ]: preprocessed_metadata_pb2.PreprocessedMetadata.NodeMetadataOutput(),
        EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES[
            1
        ]: preprocessed_metadata_pb2.PreprocessedMetadata.NodeMetadataOutput(),
        EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES[
            2
        ]: preprocessed_metadata_pb2.PreprocessedMetadata.NodeMetadataOutput(),
    },
    condensed_edge_type_to_preprocessed_metadata={
        EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[
            0
        ]: preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataOutput(
            main_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
            positive_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
            negative_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
        ),
        EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[
            1
        ]: preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataOutput(
            main_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
            positive_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
            negative_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
        ),
        EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[
            2
        ]: preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataOutput(
            main_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
            positive_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
            negative_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(),
        ),
    },
)

EXAMPLE_NABLP_HETEROGENEOUS_TASK_METADATA_PB = gbml_config_pb2.GbmlConfig.TaskMetadata(
    node_anchor_based_link_prediction_task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata(
        supervision_edge_types=[EXAMPLE_HETEROGENEOUS_EDGE_TYPES[0]]
    )
)

EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB = graph_schema_pb2.GraphMetadata(
    node_types=EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR,
    edge_types=EXAMPLE_HETEROGENEOUS_EDGE_TYPES,
    condensed_node_type_map={
        EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES[
            0
        ]: EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[0],
        EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES[
            1
        ]: EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[1],
        EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES[
            2
        ]: EXAMPLE_HETEROGENEOUS_NODE_TYPES_STR[2],
    },
    condensed_edge_type_map={
        EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[0]: EXAMPLE_HETEROGENEOUS_EDGE_TYPES[
            0
        ],
        EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[1]: EXAMPLE_HETEROGENEOUS_EDGE_TYPES[
            1
        ],
        EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[2]: EXAMPLE_HETEROGENEOUS_EDGE_TYPES[
            2
        ],
    },
)

EXAMPLE_HETEROGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER = PreprocessedMetadataPbWrapper(
    preprocessed_metadata_pb=EXAMPLE_HETEROGENEOUS_PREPROCESSED_METADATA_PB
)

EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER = GraphMetadataPbWrapper(
    graph_metadata_pb=EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB
)

EXAMPLE_NABLP_HETEROGENEOUS_TASK_METADATA_PB_WRAPPER = TaskMetadataPbWrapper(
    task_metadata_pb=EXAMPLE_NABLP_HETEROGENEOUS_TASK_METADATA_PB
)
