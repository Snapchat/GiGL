from dataclasses import dataclass


@dataclass
class RunConfig:
    should_use_cuda: bool = True
