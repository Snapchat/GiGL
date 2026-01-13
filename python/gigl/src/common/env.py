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
    return GiGLComponents(os.environ[GIGL_COMPONENT_ENV_KEY])
