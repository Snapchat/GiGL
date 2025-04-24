import numpy as np


class CosineDecayScheduler:
    """A cosine decay scheduler that is used instead of torch.optim.lr_scheduler.CosineAnnealingLR since we need a scheduler specifically for
    EMA weight averaging (momentum).
    This class is from https://github.com/nerdslab/bgrl/blob/dec99f8c605e3c4ae2ece57f3fa1d41f350d11a9/bgrl/scheduler.py
    """

    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return (
                self.max_val
                * (
                    1
                    + np.cos(
                        (step - self.warmup_steps)
                        * np.pi
                        / (self.total_steps - self.warmup_steps)
                    )
                )
                / 2
            )
        else:
            raise ValueError(
                "Step ({}) > total number of steps ({}).".format(step, self.total_steps)
            )
