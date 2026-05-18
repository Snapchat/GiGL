"""
Matches the ``set_seed(seed, deterministic=False)`` shape used by
Hugging Face Transformers, MMEngine, and Accelerate; follows the recipe
at https://pytorch.org/docs/stable/notes/randomness.html.
"""

import os
import random
from typing import Final

import numpy as np
import torch

from gigl.common.logger import Logger

logger = Logger()

_DEFAULT_SEED: Final[int] = 42  # Answer to the Ultimate Question.
# Required on CUDA >= 10.2 when use_deterministic_algorithms(True) is set,
# otherwise cuBLAS matmuls raise RuntimeError. ":4096:8" trades ~24 MiB of
# extra cuBLAS workspace for keeping perf reasonable vs ":16:8".
_CUBLAS_WORKSPACE_CONFIG: Final[str] = ":4096:8"


def seed_everything(
    seed: int = _DEFAULT_SEED,
    should_enable_expensive_deterministic_compute: bool = False,
) -> None:
    """Seed Python / NumPy / PyTorch RNGs, optionally enforce deterministic torch ops.

    What gets seeded:

    - ``random.seed(seed)`` — Python stdlib.
    - ``np.random.seed(seed)`` — NumPy global RNG.
    - ``torch.manual_seed(seed)`` — CPU **and all CUDA devices**
      (``torch.manual_seed`` calls ``torch.cuda.manual_seed_all`` internally.
      Also covers PyTorch Geometric.

    When ``should_enable_expensive_deterministic_compute=True`` (opt-in; default False because it costs
    throughput and should not be enabled for training or for production inference - can be used for debugging purposes.

    - Important: Graph Sampling currently do not follow determism outlined here.
    Example:
        >>> seed_everything(42)
        42

    Args:
        seed: RNG seed.
        deterministic: If True, also enforces bitwise-deterministic torch
            ops (cudnn flags, ``use_deterministic_algorithms``,
            ``CUBLAS_WORKSPACE_CONFIG``). Default False — most training
            pipelines want seeded RNGs without paying the throughput cost.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if should_enable_expensive_deterministic_compute:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        logger.info(
            f"seed_everything: seeded python/numpy/torch with seed={seed}; "
            f"expensive deterministic algorithms ON; "
            f"throughput will degrade"
        )
    else:
        logger.info(f"seed_everything: seeded python/numpy/torch with seed={seed}")
