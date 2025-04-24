import argparse
from typing import Optional

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.validation_check.libs.frozen_config_path_checks import (
    assert_preprocessed_metadata_exists,
    assert_split_generator_output_exists,
    assert_subgraph_sampler_output_exists,
    assert_trained_model_exists,
)
from gigl.src.validation_check.libs.name_checks import (
    check_if_kfp_pipeline_job_name_valid,
)
from gigl.src.validation_check.libs.resource_config_checks import (
    check_if_inferencer_resource_config_valid,
    check_if_preprocessor_resource_config_valid,
    check_if_shared_resource_config_valid,
    check_if_split_generator_resource_config_valid,
    check_if_subgraph_sampler_resource_config_valid,
    check_if_trainer_resource_config_valid,
)
from gigl.src.validation_check.libs.template_config_checks import (
    check_if_data_preprocessor_config_cls_valid,
    check_if_graph_metadata_valid,
    check_if_inferencer_cls_valid,
    check_if_post_processor_cls_valid,
    check_if_preprocessed_metadata_valid,
    check_if_split_generator_config_valid,
    check_if_subgraph_sampler_config_valid,
    check_if_task_metadata_valid,
    check_if_trainer_cls_valid,
    check_pipeline_has_valid_start_and_stop_flags,
)
from snapchat.research.gbml import gbml_config_pb2
from snapchat.research.gbml.gigl_resource_config_pb2 import GiglResourceConfig

START_STOP_COMPONENT_TO_CLS_CHECKS_MAP = {
    # TODO: (svij-sc) Add checks as needed, otherwise we default to below anyways
    (GiGLComponents.SubgraphSampler.value, GiGLComponents.SubgraphSampler.value): [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_subgraph_sampler_config_valid,
    ],
}

START_COMPONENT_TO_CLS_CHECKS_MAP = {
    GiGLComponents.ConfigPopulator.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_data_preprocessor_config_cls_valid,
        check_if_subgraph_sampler_config_valid,
        check_if_split_generator_config_valid,
        check_if_trainer_cls_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.DataPreprocessor.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_data_preprocessor_config_cls_valid,
        check_if_subgraph_sampler_config_valid,
        check_if_split_generator_config_valid,
        check_if_trainer_cls_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.SubgraphSampler.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_preprocessed_metadata_valid,
        check_if_subgraph_sampler_config_valid,
        check_if_split_generator_config_valid,
        check_if_trainer_cls_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.SplitGenerator.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_preprocessed_metadata_valid,
        check_if_split_generator_config_valid,
        check_if_trainer_cls_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.Trainer.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_preprocessed_metadata_valid,
        check_if_trainer_cls_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.Inferencer.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_preprocessed_metadata_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.PostProcessor.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_post_processor_cls_valid,
    ],
}

START_COMPONENT_TO_ASSET_CHECKS_MAP = {
    GiGLComponents.SubgraphSampler.value: [
        assert_preprocessed_metadata_exists,
    ],
    GiGLComponents.SplitGenerator.value: [
        assert_preprocessed_metadata_exists,
        assert_subgraph_sampler_output_exists,
    ],
    GiGLComponents.Trainer.value: [
        assert_preprocessed_metadata_exists,
        assert_subgraph_sampler_output_exists,
        assert_split_generator_output_exists,
    ],
    GiGLComponents.Inferencer.value: [
        assert_preprocessed_metadata_exists,
        assert_subgraph_sampler_output_exists,
        assert_trained_model_exists,
    ],
}

START_STOP_COMPONENT_TO_RESOURCE_CONFIG_CHECKS_MAP = {
    (GiGLComponents.SubgraphSampler.value, GiGLComponents.SubgraphSampler.value): [
        check_if_shared_resource_config_valid,
        check_if_subgraph_sampler_resource_config_valid,
    ],
}

START_COMPONENT_TO_RESOURCE_CONFIG_CHECKS_MAP = {
    GiGLComponents.ConfigPopulator.value: [
        check_if_shared_resource_config_valid,
    ],
    GiGLComponents.DataPreprocessor.value: [
        check_if_shared_resource_config_valid,
        check_if_preprocessor_resource_config_valid,
        check_if_subgraph_sampler_resource_config_valid,
        check_if_split_generator_resource_config_valid,
        check_if_trainer_resource_config_valid,
        check_if_inferencer_resource_config_valid,
    ],
    GiGLComponents.SubgraphSampler.value: [
        check_if_shared_resource_config_valid,
        check_if_subgraph_sampler_resource_config_valid,
        check_if_split_generator_resource_config_valid,
        check_if_trainer_resource_config_valid,
        check_if_inferencer_resource_config_valid,
    ],
    GiGLComponents.SplitGenerator.value: [
        check_if_shared_resource_config_valid,
        check_if_split_generator_resource_config_valid,
        check_if_trainer_resource_config_valid,
        check_if_inferencer_resource_config_valid,
    ],
    GiGLComponents.Trainer.value: [
        check_if_shared_resource_config_valid,
        check_if_trainer_resource_config_valid,
        check_if_inferencer_resource_config_valid,
    ],
    GiGLComponents.Inferencer.value: [
        check_if_shared_resource_config_valid,
        check_if_inferencer_resource_config_valid,
    ],
    GiGLComponents.PostProcessor.value: [
        check_if_shared_resource_config_valid,
    ],
}

logger = Logger()


def kfp_validation_checks(
    job_name: str,
    task_config_uri: Uri,
    start_at: str,
    resource_config_uri: Uri,
    stop_after: Optional[str] = None,
) -> None:
    # check if job_name is valid
    check_if_kfp_pipeline_job_name_valid(job_name=job_name)
    # check if start_at and stop_after aligns with glt backend use
    check_pipeline_has_valid_start_and_stop_flags(
        start_at=start_at, stop_after=stop_after, task_config_uri=task_config_uri.uri
    )
    proto_utils = ProtoUtils()
    gbml_config_pb: gbml_config_pb2.GbmlConfig = proto_utils.read_proto_from_yaml(
        uri=task_config_uri, proto_cls=gbml_config_pb2.GbmlConfig
    )

    resource_config_wrapper: GiglResourceConfigWrapper = get_resource_config(
        resource_config_uri=resource_config_uri
    )
    resource_config_pb: GiglResourceConfig = resource_config_wrapper.resource_config
    # check user defined classes and their runtime args
    if (
        stop_after is not None
        and (start_at, stop_after) in START_STOP_COMPONENT_TO_CLS_CHECKS_MAP
    ):
        for cls_check in START_STOP_COMPONENT_TO_CLS_CHECKS_MAP[(start_at, stop_after)]:
            cls_check(gbml_config_pb=gbml_config_pb)
    else:
        for cls_check in START_COMPONENT_TO_CLS_CHECKS_MAP.get(start_at, []):
            cls_check(gbml_config_pb=gbml_config_pb)
    # check the existence of needed assets
    for asset_check in START_COMPONENT_TO_ASSET_CHECKS_MAP.get(start_at, []):
        asset_check(gbml_config_pb=gbml_config_pb)
    # check if user-provided resource config is valid
    if (
        stop_after is not None
        and (start_at, stop_after) in START_STOP_COMPONENT_TO_RESOURCE_CONFIG_CHECKS_MAP
    ):
        for resource_config_check in START_STOP_COMPONENT_TO_RESOURCE_CONFIG_CHECKS_MAP[
            (start_at, stop_after)
        ]:
            resource_config_check(resource_config_pb=resource_config_pb)
    else:
        for resource_config_check in START_COMPONENT_TO_RESOURCE_CONFIG_CHECKS_MAP.get(
            start_at, []
        ):
            resource_config_check(resource_config_pb=resource_config_pb)
    # check if trained model file exist when skipping training
    if gbml_config_pb.shared_config.should_skip_training == True:
        assert_trained_model_exists(gbml_config_pb=gbml_config_pb)

    logger.info("[SUCCESS] All checks passed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Checks if config files and assets are valid for a GiGL pipeline run."
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="Unique identifier for the job name",
    )
    parser.add_argument(
        "--task_config_uri",
        type=str,
        help="GCS URI to template_or_frozen_config_uri",
    )
    parser.add_argument(
        "--start_at",
        type=str,
        help="Specify the component where to start the pipeline",
    )
    parser.add_argument(
        "--stop_after",
        type=str,
        help="Specify the component where to stop the pipeline",
    )
    parser.add_argument(
        "--resource_config_uri",
        type=str,
        help="Runtime argument for resource and env specifications of each component",
    )
    args = parser.parse_args()

    kfp_validation_checks(
        job_name=args.job_name,
        task_config_uri=UriFactory.create_uri(args.task_config_uri),
        start_at=args.start_at,
        resource_config_uri=UriFactory.create_uri(args.resource_config_uri),
        stop_after=args.stop_after,
    )
