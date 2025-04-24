from enum import Enum

from snapchat.research.gbml import gbml_config_pb2


class TaskMetadataType(str, Enum):
    NODE_BASED_TASK = (
        gbml_config_pb2.GbmlConfig.TaskMetadata.NodeBasedTaskMetadata.__name__
    )
    NODE_ANCHOR_BASED_LINK_PREDICTION_TASK = (
        gbml_config_pb2.GbmlConfig.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata.__name__
    )
    LINK_BASED_TASK = (
        gbml_config_pb2.GbmlConfig.TaskMetadata.LinkBasedTaskMetadata.__name__
    )
