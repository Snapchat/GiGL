from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Data
from typing_extensions import Self

from gigl.src.common.models.pyg.homogeneous import GraphSAGE


class HomogeneousNodeClassificationGNN(nn.Module):
    """
    Example homogeneous node-classification model wrapping a GraphSAGE encoder with a
    Linear classifier head.

    The outer object stays a plain ``nn.Module``; DDP wraps the internal ``_encoder`` and
    ``_head`` in place via :py:meth:`to_ddp`. See
    :py:class:`gigl.nn.models.LinkPredictionGNN` for the analogous pattern.

    Args:
        encoder (nn.Module): GNN encoder producing per-node embeddings of shape ``[N, hid_dim]``.
        head (nn.Module): Linear classifier mapping embeddings to ``[N, num_classes]`` logits.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self._encoder = encoder
        self._head = head

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def head(self) -> nn.Module:
        return self._head

    def forward(self, data: Data, device: torch.device) -> torch.Tensor:
        """
        Runs the encoder then the classifier head on a sampled subgraph batch.

        Args:
            data (Data): Sampled subgraph batch.
            device (torch.device): Compute device for the forward pass.

        Returns:
            torch.Tensor: Logits of shape ``[num_sampled_nodes, num_classes]``.
        """
        node_embeddings = self._encoder(data=data, device=device)
        logits = self._head(node_embeddings)
        return logits

    def to_ddp(
        self,
        device: torch.device,
        find_unused_encoder_parameters: bool = False,
    ) -> Self:
        """
        Wraps the internal encoder and classifier head in ``DistributedDataParallel`` in place.

        Mirrors :py:meth:`gigl.nn.models.LinkPredictionGNN.to_ddp`: the outer module stays a plain
        ``nn.Module`` so saved state-dict keys remain unprefixed after a subsequent
        :py:meth:`unwrap_from_ddp`.

        Args:
            device (torch.device): The device DDP should bind to.
            find_unused_encoder_parameters (bool): Forwarded as ``find_unused_parameters`` to the
                encoder's ``DistributedDataParallel`` wrapper.

        Returns:
            Self: This instance with ``_encoder`` and ``_head`` replaced by DDP wrappers.
        """
        self._encoder = DistributedDataParallel(
            self._encoder.to(device),
            device_ids=[device] if device.type != "cpu" else None,
            find_unused_parameters=find_unused_encoder_parameters,
        )
        self._head = DistributedDataParallel(
            self._head.to(device),
            device_ids=[device] if device.type != "cpu" else None,
        )
        return self

    def unwrap_from_ddp(self) -> "HomogeneousNodeClassificationGNN":
        """
        Returns a fresh instance with unwrapped sub-modules suitable for saving.

        Returns:
            HomogeneousNodeClassificationGNN: New instance where ``_encoder`` and ``_head`` are
            the original plain modules.
        """
        encoder = (
            self._encoder.module
            if isinstance(self._encoder, DistributedDataParallel)
            else self._encoder
        )
        head = (
            self._head.module
            if isinstance(self._head, DistributedDataParallel)
            else self._head
        )
        return HomogeneousNodeClassificationGNN(encoder=encoder, head=head)


def init_example_gigl_homogeneous_node_classification_model(
    node_feature_dim: int,
    num_classes: int,
    hid_dim: int = 16,
    num_layers: int = 2,
    device: Optional[torch.device] = None,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
    wrap_with_ddp: bool = False,
    find_unused_encoder_parameters: bool = False,
) -> HomogeneousNodeClassificationGNN:
    """
    Initializes a homogeneous node-classification model: ``GraphSAGE`` encoder + ``Linear`` head.

    The factory order is deliberately:

    1. Construct the model with plain sub-modules.
    2. Move to ``device``.
    3. If ``state_dict`` is provided, load it against the unwrapped model so that the saved keys
       (e.g. ``_encoder.<weight>``, ``_head.<weight>``) match the live model's keys.
    4. If ``wrap_with_ddp`` is ``True``, wrap the sub-modules in ``DistributedDataParallel``.

    This deviates from :py:func:`examples.link_prediction.models.init_example_gigl_homogeneous_model`,
    which calls ``to_ddp`` *before* ``load_state_dict``. That ordering has a latent key-mismatch
    bug on the ``should_skip_training=True`` eval-only path because DDP-wrapped sub-modules expose
    state-dict keys with a ``module.`` prefix.

    Args:
        node_feature_dim (int): Input node feature dimension.
        num_classes (int): Number of output classes for the classifier head.
        hid_dim (int): Encoder hidden and output dimension (head input dim).
        num_layers (int): Number of GraphSAGE convolution layers.
        device (Optional[torch.device]): Target device; defaults to CPU.
        state_dict (Optional[dict[str, torch.Tensor]]): Optional pretrained weights.
        wrap_with_ddp (bool): If ``True``, internally wrap ``_encoder`` and ``_head`` in DDP.
        find_unused_encoder_parameters (bool): Forwarded to the encoder's DDP wrapper.

    Returns:
        HomogeneousNodeClassificationGNN: Ready-to-train (or ready-to-infer) model.
    """
    # `GraphSAGE.supports_edge_attr` is False (see `gigl/src/common/models/pyg/homogeneous.py:173`),
    # so any edge features present in the dataset are ignored by this encoder. We pass `edge_dim=None`
    # to make that explicit. An edge-aware variant (e.g. `GINE`, `EdgeAttrGAT`) is a natural follow-up.
    encoder = GraphSAGE(
        in_dim=node_feature_dim,
        hid_dim=hid_dim,
        out_dim=hid_dim,
        edge_dim=None,
        num_layers=num_layers,
        conv_kwargs={},
        should_l2_normalize_embedding_layer_output=False,
    )
    head = nn.Linear(hid_dim, num_classes)
    model = HomogeneousNodeClassificationGNN(encoder=encoder, head=head)

    if device is None:
        device = torch.device("cpu")
    model.to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    if wrap_with_ddp:
        model.to_ddp(
            device=device,
            find_unused_encoder_parameters=find_unused_encoder_parameters,
        )

    return model
