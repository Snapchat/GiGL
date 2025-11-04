from gigl.common.logger import Logger
from gigl.nn.models import LightGCN, LinkPredictionGNN

__all__ = ["LinkPredictionGNN", "LightGCN"]

logger = Logger()

logger.warning(
    "gigl.module.models is deprecated and will be removed in a future release. "
    "Please use the `gigl.nn.models` module instead."
)
