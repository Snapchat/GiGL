from typing import Union

from torch import Tensor
from torch_geometric.nn import GATConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor, set_diag


class EdgeAttrGATConv(GATConv):
    r"""
    Compared to GATConv, EdgeAttrGATConv combines node features with edge features for message passing.
    """

    def __init__(self, share_edge_att_message_weight: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.edge_dim is not None and not share_edge_att_message_weight:
            self.lin_edge_message = Linear(
                self.edge_dim,
                self.heads * self.out_channels,
                bias=False,
                weight_initializer="glorot",
            )
        else:
            self.lin_edge_message = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "lin_edge_message") and self.lin_edge_message is not None:
            self.lin_edge_message.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights=None,
    ):
        r"""
        Compared to GATConv, EdgeAttrGATConv adds edge_attr as extra input of propagate

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(
            edge_index, x=x, alpha=alpha, size=size, edge_attr=edge_attr
        )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(self, x_j, alpha, edge_attr):
        r"""
        Compared to GATConv, EdgeAttrGATConv has extra step of adding node features and edge features
        """
        if edge_attr is not None and (self.lin_edge_message or self.lin_edge):
            edge_message_layer = (
                self.lin_edge_message if self.lin_edge_message else self.lin_edge
            )
            edge_attr = edge_message_layer(edge_attr).view(
                -1, self.heads, self.out_channels
            )
            x_j += edge_attr

        return alpha.unsqueeze(-1) * x_j
