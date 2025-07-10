from typing import Callable, Final

from torch.nn import functional as F

ACT_MAP: Final[dict[str, Callable]] = {
    "relu": F.relu,
    "elu": F.elu,
    "leakyrelu": F.leaky_relu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "gelu": F.gelu,
    "silu": F.silu,
}

DEFAULT_NUM_GNN_HOPS: Final[int] = 2
