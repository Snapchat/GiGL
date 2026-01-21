from multiprocessing.reduction import ForkingPickler

from gigl.common.logger import Logger
from gigl.distributed.dist_dataset import DistDataset, _reduce_distributed_dataset

logger = Logger()


# TODO (mkolodner-sc): Deprecate this class in favor of DistDataset
class DistLinkPredictionDataset(DistDataset):
    def __init__(self, *args, **kwargs):
        logger.warning(
            "DistLinkPredictionDataset is deprecated. Please use DistDataset instead."
        )
        super().__init__(*args, **kwargs)


# Register custom serialization for DistLinkPredictionDataset with multiprocessing's ForkingPickler.
# This enables DistLinkPredictionDataset objects to be safely passed between processes by using
# IPC handles instead of trying to pickle the underlying shared memory directly,
# which would fail or cause data corruption.
ForkingPickler.register(DistLinkPredictionDataset, _reduce_distributed_dataset)
