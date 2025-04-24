from typing import Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class GINEConv(MessagePassing):
    r"""
    Modified version of PyG's GINE conv implementation
    https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gin_conv.py

    PyG's implementation assumes edge_attr is always present, see the message function for more details
    """

    def __init__(
        self,
        nn: torch.nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, "in_features"):
                in_channels = nn.in_features
            elif hasattr(nn, "in_channels"):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)  # type: ignore

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        """
        PyG's implementation assumes edge_attr to be present, we allow None for edge_attr
        """
        if (
            isinstance(edge_attr, Tensor)
            and self.lin is None
            and x_j.size(-1) != edge_attr.size(-1)
        ):
            raise ValueError(
                "Node and edge feature dimensionalities do not "
                "match. Consider setting the 'edge_dim' "
                "attribute of 'GINEConv'"
            )

        if isinstance(edge_attr, Tensor) and self.lin is not None:
            edge_attr = self.lin(edge_attr)

        if not isinstance(edge_attr, Tensor):
            return x_j.relu()
        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"
