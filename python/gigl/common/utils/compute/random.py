import os
import random

from gigl.common.logger import Logger

logger = Logger()


def make_compute_deterministic_and_set_seed(
    seed: int = 42,  # Answer to the Ultimate Question of Life, The Universe, and Everything
    should_consider_numpy=True,
    should_consider_torch=False,
    should_consider_tensorflow=False,
):
    logger.info(
        """
        Ensure data loading is also deterministic and you are using deterministic algorithms
        for relevant frameworks, otherwise nondeterminism will persist
        """
    )

    # Setting PYTHONHASHSEED doesn't seem like it actually does anything
    # See: https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    # os.environ["PYTHONHASHSEED"] = "0"

    random.seed(seed)

    if should_consider_numpy:
        import numpy as np

        np.random.seed(seed)

    if should_consider_torch:
        import torch
        import torch.backends.cudnn

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    if should_consider_tensorflow:
        import tensorflow as tf

        tf.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
