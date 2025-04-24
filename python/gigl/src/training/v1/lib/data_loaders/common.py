from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import NodeType

logger = Logger()
_DEFAULT_DATA_LOADER_BATCH_SIZE = 32
_DEFAULT_DATA_LOADER_NUM_WORKERS = 0
_DEFAULT_DATA_LOADER_SEED = 42
# TODO(nshah-sc): refactor out sample-wise preprocessing methods into pb wrappers.


class DataloaderTypes(Enum):
    train_main = "train_main"
    val_main = "val_main"
    test_main = "test_main"
    train_random_negative = "train_random_negative"
    val_random_negative = "val_random_negative"
    test_random_negative = "test_random_negative"


@dataclass
class DataloaderConfig:
    uris: Union[List[Uri], Dict[NodeType, List[Uri]]]
    batch_size: int = _DEFAULT_DATA_LOADER_BATCH_SIZE
    num_workers: int = _DEFAULT_DATA_LOADER_NUM_WORKERS
    should_loop: bool = False
    pin_memory: bool = False
    seed: int = _DEFAULT_DATA_LOADER_SEED
