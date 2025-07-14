from typing import Any, Callable, NamedTuple, TypeVar

import kfp

_T = TypeVar("_T")


def gigl_component(
    image: str, component_fn: Callable[[Any], _T]
) -> Callable[[Any], _T]:
    return kfp.dsl.component(
        component_fn,
        base_image=image,
    )


def decompose(
    image: str,
    task_config_uri: str,
) -> NamedTuple(
    "DecomposedTaskConfig",
    task_config_uri=str,
    graph_metadata_json=str,
    data_preprocessr_config_json=str,
):
    def decompose_fn(
        task_config_uri: str,
    ) -> NamedTuple(
        "DecomposedTaskConfig",
        task_config_uri=str,
        graph_metadata_json=str,
        data_preprocessr_config_json=str,
    ):
        from typing import NamedTuple

        from google.protobuf.json_format import MessageToJson

        from gigl.common import UriFactory
        from gigl.common.logger import Logger
        from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper

        logger = Logger()
        logger.info(f"Task config uri: {task_config_uri}")
        gbml_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
            gbml_config_uri=UriFactory.create_uri(task_config_uri)
        )
        DecomposedTaskConfig = NamedTuple(
            "DecomposedTaskConfig",
            task_config_uri=str,
            graph_metadata_json=str,
            data_preprocessr_config_json=str,
        )

        return DecomposedTaskConfig(
            task_config_uri=task_config_uri,
            graph_metadata_json=MessageToJson(gbml_pb_wrapper.graph_metadata),
            data_preprocessr_config_json=MessageToJson(
                gbml_pb_wrapper.gbml_config_pb.dataset_config.data_preprocessor_config
            ),
        )

    return gigl_component(
        image=image,
        component_fn=decompose_fn,
    )(task_config_uri=task_config_uri)


def preprocessor(
    image: str,
) -> NamedTuple("PreprocessorOutputs", preprocessed_metadata_uri=str):
    @gigl_component(image=image)
    def preprocessor_fn(
        task_config_uri: str,
    ) -> NamedTuple("PreprocessorOutputs", preprocessed_metadata_uri=str):
        pass

    return preprocessor_fn
