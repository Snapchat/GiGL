from dataclasses import dataclass


@dataclass
class DataloaderConfig:
    num_workers: int = 1
    pin_memory: bool = True
