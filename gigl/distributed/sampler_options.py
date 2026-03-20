"""Sampler option types for configuring which sampler class to use in distributed loading.

Provides ``KHopNeighborSamplerOptions`` for using GiGL's built-in ``DistNeighborSampler``,
and ``PPRSamplerOptions`` for PPR-based sampling using ``DistPPRNeighborSampler``.

Frozen dataclasses so they are safe to pickle across RPC boundaries
(required for Graph Store mode).
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from graphlearn_torch.typing import EdgeType

from gigl.common.logger import Logger

logger = Logger()


@dataclass(frozen=True)
class KHopNeighborSamplerOptions:
    """Default sampler options using GiGL's DistNeighborSampler.

    Attributes:
        num_neighbors: Fanout per hop, either a flat list (homogeneous) or a
            dict mapping edge types to per-hop fanout lists (heterogeneous).
    """

    num_neighbors: Union[list[int], dict[EdgeType, list[int]]]


@dataclass(frozen=True)
class PPRSamplerOptions:
    """Sampler options for PPR-based neighbor sampling using DistPPRNeighborSampler.

    Attributes:
        alpha: Restart probability (teleport probability back to seed). Higher
            values keep samples closer to seeds. Typical values: 0.15-0.25.
        eps: Convergence threshold for the Forward Push algorithm. Smaller
            values give more accurate PPR scores but require more computation.
            Typical values: 1e-4 to 1e-6.
        max_ppr_nodes: Maximum number of nodes to return per seed based on PPR
            scores.
        num_neighbors_per_hop: Maximum number of neighbors fetched per node per edge
            type during PPR traversal. Set large to approximate fetching all
            neighbors.
        total_degree_dtype: Dtype for precomputed total-degree tensors. Defaults
            to ``torch.int32``, which supports total degrees up to ~2 billion.
            Use a larger dtype if nodes have exceptionally high aggregate degrees.
    """

    alpha: float = 0.5
    eps: float = 1e-4
    max_ppr_nodes: int = 50
    num_neighbors_per_hop: int = 100_000
    total_degree_dtype: torch.dtype = torch.int32


SamplerOptions = Union[KHopNeighborSamplerOptions, PPRSamplerOptions]


def resolve_sampler_options(
    num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
    sampler_options: Optional[SamplerOptions],
) -> SamplerOptions:
    """Resolve sampler_options from user-provided values.

    If ``sampler_options`` is a ``PPRSamplerOptions``, returns it directly (``num_neighbors`` is unused for PPR).
    If ``sampler_options`` is ``None``, wraps ``num_neighbors`` in a ``KHopNeighborSamplerOptions``.
    If ``KHopNeighborSamplerOptions`` is provided, validates that its ``num_neighbors`` matches the explicit value.

    Args:
        num_neighbors: Fanout per hop (required for KHop; ignored for PPR).
        sampler_options: Sampler configuration, or None.

    Returns:
        The resolved SamplerOptions.

    Raises:
        ValueError: If ``KHopNeighborSamplerOptions.num_neighbors`` conflicts
            with the explicit ``num_neighbors``.
    """
    if isinstance(sampler_options, PPRSamplerOptions):
        return sampler_options

    if sampler_options is None:
        return KHopNeighborSamplerOptions(num_neighbors)

    if num_neighbors != sampler_options.num_neighbors:
        raise ValueError(
            f"num_neighbors ({num_neighbors}) does not match "
            f"sampler_options.num_neighbors ({sampler_options.num_neighbors})."
        )
    logger.info(f"Using sampler options: {sampler_options}")

    return sampler_options
