import os

from gigl.env.distributed import GIGL_COMPONENT_ENV_KEY
from gigl.src.common.constants.components import GiGLComponents


def get_component() -> GiGLComponents:
    """Get the component of the current job.

    Returns:
        GiGLComponents: The component of the current job.
    Raises:
        ValueError: If the component is not valid.
    """
    if GIGL_COMPONENT_ENV_KEY not in os.environ:
        raise KeyError(
            f"Environment variable {GIGL_COMPONENT_ENV_KEY} is not set. Cannot determine the component of the current job. Please set the environment variable like `export GIGL_COMPONENT=trainer`."
        )
    return GiGLComponents(os.environ[GIGL_COMPONENT_ENV_KEY])
