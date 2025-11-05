from gigl.common.logger import Logger
from gigl.nn.loss import RetrievalLoss

__all__ = ["RetrievalLoss"]

logger = Logger()

logger.warning(
    "gigl.module.loss is deprecated and will be removed in a future release. "
    "Please use the `gigl.nn.loss` module instead."
)
