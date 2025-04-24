from typing import Optional

import torch
import torch.nn as nn


class DCNCross(nn.Module):
    """
    Derived from tensorflow_recommenders [implementation](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/layers/dcn/Cross)
    Cross Layer in Deep & Cross Network to learn explicit feature interactions.

    A layer that creates explicit and bounded-degree feature interactions efficiently. The `call` method accepts `inputs` as a tuple of size 2 tensors. The first input `x0` is the base layer that contains the original features (usually the embedding layer); the second input `xi` is the output of the previous `DCNCross` layer in the stack, i.e., the i-th `DCNCross` layer. For the first `DCNCross` layer in the stack, x0 = xi.

    The output is x_{i+1} = x0 .* (W * xi + bias + diag_scale * xi) + xi,
    where .* designates elementwise multiplication, W could be a full-rank matrix, or a low-rank matrix U*V to reduce the computational cost, and diag_scale increases the diagonal of W to improve training stability (especially for the low-rank case).

    References:
    - [R. Wang et al.](https://arxiv.org/pdf/2008.13535.pdf) See Eq. (1) for full-rank and Eq. (2) for low-rank version.
    - [R. Wang et al.](https://arxiv.org/pdf/1708.05123.pdf)

    Args:
        in_dim (int): The input feature dimension.
        projection_dim (Optional[int]): Projection dimension to reduce the computational cost. Default is `None` such that a full (`in_dim` by `in_dim`) matrix W is used. If enabled, a low-rank matrix W = U*V will be used, where U is of size `in_dim` by `projection_dim` and V is of size `projection_dim` by `in_dim`. `projection_dim` needs to be smaller than `in_dim`/2 to improve the model efficiency. In practice, we've observed that `projection_dim` = d/4 consistently preserved the accuracy of a full-rank version.
        diag_scale (float): A non-negative float used to increase the diagonal of the kernel W by `diag_scale`, that is, W + diag_scale * I, where I is an identity matrix.
        use_bias (bool): Whether to add a bias term for this layer. If set to False, no bias term will be used.

    Input shape:
        A tuple of 2 (batch_size, `in_dim`) dimensional inputs.
    Output shape:
        A single (batch_size, `in_dim`) dimensional output.
    """

    def __init__(
        self,
        in_dim: int,
        projection_dim: Optional[int] = None,
        diag_scale: float = 0.0,
        use_bias: bool = True,
        **kwargs,
    ):
        super(DCNCross, self).__init__(**kwargs)
        self._in_dim = in_dim
        self._projection_dim = projection_dim
        self._diag_scale = diag_scale
        self._use_bias = use_bias

        if self._diag_scale < 0.0:
            raise ValueError(
                f"`diag_scale` should be non-negative. Got `diag_scale` = {diag_scale}"
            )

        if self._projection_dim is None:
            self._lin = nn.Linear(self._in_dim, self._in_dim, bias=self._use_bias)
        else:
            self._lin_u = nn.Linear(
                self._in_dim, self._projection_dim, bias=self._use_bias
            )
            self._lin_v = nn.Linear(
                self._projection_dim, self._in_dim, bias=self._use_bias
            )

    def forward(
        self, x0: torch.Tensor, x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the feature cross.
        Args:
        x0: The input tensor
        x: Optional second input tensor. If provided, the layer will compute
            crosses between x0 and x; if not provided, the layer will compute
            crosses between x0 and itself.

        Returns:
        Tensor of crosses.
        """
        if x is None:
            x = x0

        if x0.shape[-1] != x.shape[-1]:
            raise ValueError(
                f"`x0` and `x` dimension mismatch! Got `x0` dimension {x0.shape[-1]}, and x "
                "dimension {x.shape[-1]}. This case is not supported yet."
            )
        if self._projection_dim is None:
            prod_output = self._lin(x)
        else:
            prod_output = self._lin_v(self._lin_u(x))

        if self._diag_scale:
            prod_output += self._diag_scale * x
        return x0 * prod_output + x

    def reset_parameters(self):
        if self._projection_dim is None:
            self._lin.reset_parameters()
        else:
            self._lin_u.reset_parameters()
            self._lin_v.reset_parameters()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_dim={self._in_dim}, projection_dim={self._projection_dim}, diag_scale={self._diag_scale}, use_bias={self._use_bias})"


class DCNv2(nn.Module):
    """
    Wraps around `DCNCross` for multi-layer feature crossing. See documentation for `DCNCross` for more details.

    Args:
        in_dim (int): The input feature dimension.
        num_layers (int): How many feature crossing layers to use. K layers will produce as high as (K+1)-order features.
        projection_dim (Optional[int]): Projection dimension to reduce the computational cost. Default is `None` such that a full (`in_dim` by `in_dim`) matrix W is used. If enabled, a low-rank matrix W = U*V will be used, where U is of size `in_dim` by `projection_dim` and V is of size `projection_dim` by `in_dim`. `projection_dim` needs to be smaller than `in_dim`/2 to improve the model efficiency. In practice, we've observed that `projection_dim` = d/4 consistently preserved the accuracy of a full-rank version.
        diag_scale (float): A non-negative float used to increase the diagonal of the kernel W by `diag_scale`, that is, W + diag_scale * I, where I is an identity matrix.
        use_bias (bool): Whether to add a bias term for this layer. If set to False, no bias term will be used.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int = 1,
        projection_dim: Optional[int] = None,
        diag_scale: float = 0.0,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._in_dim = in_dim
        self._num_layers = num_layers
        self._projection_dim = projection_dim
        self._diag_scale = diag_scale
        self._use_bias = use_bias

        self._layers = nn.ModuleList()
        for _ in range(num_layers):
            self._layers.append(
                DCNCross(
                    self._in_dim,
                    projection_dim=self._projection_dim,
                    diag_scale=self._diag_scale,
                    use_bias=self._use_bias,
                    **kwargs,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, xl = x, x
        for i in range(self._num_layers):
            xl = self._layers[i](x0, xl)
        return xl

    def reset_parameters(self):
        for layer in self._layers:
            layer.reset_parameters()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_dim={self._in_dim}, num_layers={self._num_layers}, projection_dim={self._projection_dim}, diag_scale={self._diag_scale}, use_bias={self._use_bias})"
