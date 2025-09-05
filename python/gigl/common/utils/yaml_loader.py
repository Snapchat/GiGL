from typing import Type, TypeVar, cast

from omegaconf import OmegaConf

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.common.omegaconf_resolvers import register_resolvers
from gigl.src.common.utils.file_loader import FileLoader

logger = Logger()

T = TypeVar("T")

register_resolvers()


def load_resolved_yaml(uri: Uri, type_of_object: Type[T]) -> T:
    with FileLoader().load_to_temp_file(uri) as tf:
        test_spec_data = OmegaConf.load(tf.name)

    # Merge OmegaConf structured config with loaded data for validation
    merged_config = OmegaConf.merge(
        OmegaConf.structured(type_of_object), test_spec_data
    )

    # Convert to strongly typed E2ETestsSpec object
    return cast(T, OmegaConf.to_object(merged_config))
