from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor  # noqa


class JumpingKnowledge(MessagePassing):
    """
    The Jumping Knowledge layer aggregation module from the
    "Representation Learning on Graphs with Jumping Knowledge Networks"
    <https://arxiv.org/abs/1806.03536> paper, which supports several
    aggregation schemes: concatenation ("cat"), max pooling ("max"), and
    weighted summation with attention scores from a bi-directional LSTM ("lstm").

    This implementation applies a final linear transformation to ensure the output
    of this layer matches the specified output dimension.

    :param mode: The aggregation scheme to use ("cat", "max", or "lstm").
    :type mode: str
    :param hid_dim: Dimension of the hidden layers in graph convolutions.
    :type hid_dim: int
    :param out_dim: Dimension of the output embeddings.
    :type out_dim: int
    :param num_layers: The number of layers to aggregate. Required for "cat" and "lstm" modes.
    :type num_layers: Optional[int], optional
    :param lstm_dim: The dimension of LSTM hidden layers. Used only in "lstm" mode. Defaults to `hid_dim` if not set.
    :type lstm_dim: Optional[int], optional

    Supported aggregation schemes:

    - **concatenation** ("cat"):

      .. math::
          \\mathbf{x}_v^{(1)} \\, \\Vert \\, \\ldots \\, \\Vert \\, \\mathbf{x}_v^{(T)}

    - **max pooling** ("max"):

      .. math::
          \\max \\left( \\mathbf{x}_v^{(1)}, \\ldots, \\mathbf{x}_v^{(T)} \\right)

    - **weighted summation**:

      With attention scores :math:`\\alpha_v^{(t)}` obtained from a bi-directional LSTM ("lstm").

      .. math::
          \\sum_{t=1}^T \\alpha_v^{(t)} \\mathbf{x}_v^{(t)}
    """

    def __init__(
        self,
        mode: str,
        hid_dim: int,
        out_dim: int,
        num_layers: Optional[int] = None,
        lstm_dim: Optional[int] = None,
    ):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ["cat", "max", "lstm"]
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        if self.mode == "lstm":
            assert num_layers is not None, "num_layers cannot be None for lstm mode"
            self.lstm_dim = lstm_dim if lstm_dim else hid_dim
            self.lstm = nn.LSTM(
                input_size=hid_dim,
                hidden_size=(num_layers * self.lstm_dim) // 2,
                bidirectional=True,
                batch_first=True,
            )
            self.att = nn.Linear(2 * ((num_layers * self.lstm_dim) // 2), 1)
            self.num_layers = num_layers
            self.output_linear = nn.Linear(hid_dim, out_dim)
        elif self.mode == "cat":
            assert num_layers is not None, "num_layers cannot be none for cat mode"
            self.lstm_dim = None  # type: ignore
            self.lstm = None  # type: ignore
            self.att = None  # type: ignore
            self.num_layers = num_layers
            self.output_linear = nn.Linear((num_layers * hid_dim), out_dim)
        else:  # self.mode == "max"
            self.lstm_dim = None  # type: ignore
            self.lstm = None  # type: ignore
            self.att = None  # type: ignore
            self.num_layers = None  # type: ignore
            self.output_linear = nn.Linear(hid_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.lstm is not None:
            self.lstm.reset_parameters()
        if self.att is not None:
            self.att.reset_parameters()
        self.output_linear.reset_parameters()

    def forward(self, xs: List[torch.Tensor]) -> Tensor:
        r"""
        Args:
            xs (List[torch.Tensor]): List containing the layer-wise
                representations.
        """
        if self.mode == "cat":
            return self.output_linear(torch.cat(xs, dim=-1))
        elif self.mode == "max":
            return self.output_linear(torch.stack(xs, dim=-1).max(dim=-1)[0])
        else:  # self.mode == 'lstm'
            assert self.lstm is not None and self.att is not None
            x = torch.stack(xs, dim=1)  # [num_nodes, num_layers, hid_dim]
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            return self.output_linear((x * alpha.unsqueeze(-1)).sum(dim=1))

    def __repr__(self) -> str:
        if self.mode == "lstm":
            return (
                f"{self.__class__.__name__}({self.mode}, "
                f"hid_dim={self.hid_dim}, out_dim={self.out_dim}, num_layers={self.num_layers}, lstm_dim={self.lstm_dim})"
            )
        elif self.mode == "cat":
            return (
                f"{self.__class__.__name__}({self.mode}, "
                f"hid_dim={self.hid_dim}, out_dim={self.out_dim}, num_layers={self.num_layers})"
            )
        return f"{self.__class__.__name__}({self.mode}, hid_dim={self.hid_dim}, out_dim={self.out_dim})"
