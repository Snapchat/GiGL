from dataclasses import dataclass


@dataclass
class DataloaderConfig:
    """
    Configuration for PyTorch DataLoader used in knowledge graph embedding training.

    Controls data loading efficiency and memory usage during training and evaluation.

    Attributes:
        num_workers (int): Number of worker processes for data loading. Higher values can
            improve data loading speed but use more memory and CPU cores. Setting to 0
            uses the main process for data loading. Defaults to 1.
        pin_memory (bool): Whether to pin loaded data tensors to GPU memory for faster
            host-to-device transfer. Should be True when using CUDA for training.
            May use additional GPU memory. Defaults to True.
    """

    num_workers: int = 1
    pin_memory: bool = True
