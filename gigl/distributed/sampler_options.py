"""Sampler option types for configuring which sampler class to use in distributed loading.

Provides ``KHopNeighborSamplerOptions`` for using GiGL's built-in ``DistNeighborSampler``.

Frozen dataclass so it is safe to pickle across RPC boundaries
(required for Graph Store mode).
"""

from dataclasses import dataclass
from typing import Optional, Union

from graphlearn_torch.typing import EdgeType


@dataclass(frozen=True)
class KHopNeighborSamplerOptions:
    """Default sampler options using GiGL's DistNeighborSampler.

    Attributes:
        num_neighbors: Fanout per hop, either a flat list (homogeneous) or a
            dict mapping edge types to per-hop fanout lists (heterogeneous).
    """

    num_neighbors: Union[list[int], dict[EdgeType, list[int]]]


SamplerOptions = KHopNeighborSamplerOptions


def resolve_sampler_options(
    num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
    sampler_options: Optional[SamplerOptions],
) -> SamplerOptions:
    """Resolve sampler_options from user-provided values.

    If ``sampler_options`` is ``None``, wraps ``num_neighbors`` in a
    ``KHopNeighborSamplerOptions``. If ``KHopNeighborSamplerOptions`` is
    provided, validates that its ``num_neighbors`` matches the explicit value.

    Args:
        num_neighbors: Fanout per hop (always required).
        sampler_options: Sampler configuration, or None.

    Returns:
        The resolved SamplerOptions.

    Raises:
        ValueError: If ``KHopNeighborSamplerOptions.num_neighbors`` conflicts
            with the explicit ``num_neighbors``.
    """
    if sampler_options is None:
        return KHopNeighborSamplerOptions(num_neighbors)

    if num_neighbors != sampler_options.num_neighbors:
        raise ValueError(
            f"num_neighbors ({num_neighbors}) does not match "
            f"sampler_options.num_neighbors ({sampler_options.num_neighbors})."
        )

    return sampler_options
