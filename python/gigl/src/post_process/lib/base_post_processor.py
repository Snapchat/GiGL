from abc import ABC, abstractmethod
from typing import Optional

from gigl.common.logger import Logger
from gigl.src.common.types.model_eval_metrics import EvalMetricsCollection
from snapchat.research.gbml import gbml_config_pb2

logger = Logger()


class BasePostProcessor(ABC):
    """
    Post processor does all operations required after inferencer.
    Ex. persist inferencer output assets to text files, or run checks on output metrics etc.
    """

    @abstractmethod
    def run_post_process(
        self, gbml_config_pb: gbml_config_pb2.GbmlConfig
    ) -> Optional[EvalMetricsCollection]:
        raise NotImplementedError
