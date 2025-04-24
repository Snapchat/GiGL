from torch.nn import functional as F

ACT_MAP = {
    "relu": F.relu,
    "elu": F.elu,
    "leakyrelu": F.leaky_relu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "gelu": F.gelu,
    "silu": F.silu,
}
DEFAULT_NUM_GNN_HOPS = 2
