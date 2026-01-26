from dataclasses import dataclass


@dataclass
class RunConfig:
    """
    Configuration for runtime execution environment settings.

    Controls the basic execution environment for knowledge graph embedding training,
    particularly hardware acceleration preferences.

    Attributes:
        should_use_cuda (bool): Whether to use CUDA (GPU) acceleration for training.
            If True, training will use available GPUs for faster computation.
            If False, training will run on CPU only. Automatically adjusted based
            on GPU availability during initialization. Defaults to True.
    """

    should_use_cuda: bool = True
