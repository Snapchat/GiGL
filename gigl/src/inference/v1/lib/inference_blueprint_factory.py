from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.inference.v1.lib.base_inference_blueprint import BaseInferenceBlueprint
from gigl.src.inference.v1.lib.base_inferencer import (
    BaseInferencer,
    SupervisedNodeClassificationBaseInferencer,
)
from gigl.src.inference.v1.lib.node_anchor_based_link_prediction_inferencer import (
    NodeAnchorBasedLinkPredictionInferenceBlueprint,
)
from gigl.src.inference.v1.lib.node_classification_inferencer import (
    NodeClassificationInferenceBlueprint,
)


class InferenceBlueprintFactory:
    @classmethod
    def get_inference_blueprint(
        cls,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        inferencer_instance: BaseInferencer,
        graph_builder: GraphBuilder,
    ) -> BaseInferenceBlueprint:
        blueprint: BaseInferenceBlueprint

        task_metadata_type = (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_type
        )

        if task_metadata_type == TaskMetadataType.NODE_BASED_TASK:
            assert isinstance(
                inferencer_instance, SupervisedNodeClassificationBaseInferencer
            )
            blueprint = NodeClassificationInferenceBlueprint(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                inferencer=inferencer_instance,
                graph_builder=graph_builder,
            )
        elif (
            task_metadata_type
            == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
        ):
            blueprint = NodeAnchorBasedLinkPredictionInferenceBlueprint(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                inferencer=inferencer_instance,
                graph_builder=graph_builder,
            )
        else:
            raise TypeError(f"GBML task type not supported: {task_metadata_type}")

        return blueprint
