from __future__ import annotations

from enum import Enum
from typing import TypeVar

from gigl.src.common.types.pb_wrappers.types import TrainingSamplePb

T = TypeVar("T", bound=TrainingSamplePb)


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
