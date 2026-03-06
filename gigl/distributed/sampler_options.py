"""Sampler option types for configuring which sampler class to use in distributed loading.

Provides two options:
- ``KHopNeighborSamplerOptions``: Uses GiGL's built-in ``DistNeighborSampler``.
- ``CustomSamplerOptions``: Dynamically imports and uses a user-provided sampler class.

Both are frozen dataclasses so they are safe to pickle across RPC boundaries
(required for Graph Store mode).
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from graphlearn_torch.typing import EdgeType


@dataclass(frozen=True)
class KHopNeighborSamplerOptions:
    """Default sampler options using GiGL's DistNeighborSampler.

    Attributes:
        num_neighbors: Fanout per hop, either a flat list (homogeneous) or a
            dict mapping edge types to per-hop fanout lists (heterogeneous).
    """

    num_neighbors: Union[list[int], dict[EdgeType, list[int]]]


@dataclass(frozen=True)
class CustomSamplerOptions:
    """Custom sampler options that dynamically import a user-provided sampler class.

    The class at ``class_path`` must conform to the same interface as
    ``DistNeighborSampler`` (extend ``GLTDistNeighborSampler`` or at minimum
    support ``start_loop``, ``sample_from_nodes``, etc.).

    Attributes:
        class_path: Fully qualified Python import path, e.g.
            ``"my.module.MySampler"``.
        class_args: Additional keyword arguments passed to the sampler
            constructor (on top of the standard GLT arguments).
    """

    class_path: str
    class_args: dict[str, Any] = field(default_factory=dict)


SamplerOptions = Union[KHopNeighborSamplerOptions, CustomSamplerOptions]


def resolve_sampler_options(
    num_neighbors: Optional[Union[list[int], dict[EdgeType, list[int]]]],
    sampler_options: Optional[SamplerOptions],
) -> tuple[Union[list[int], dict[EdgeType, list[int]]], SamplerOptions]:
    """Resolve num_neighbors and sampler_options from user-provided values.

    Handles backwards compatibility: callers can provide ``num_neighbors``
    alone (old API), ``sampler_options`` alone (new API), or both (validated
    for consistency).

    Args:
        num_neighbors: Fanout per hop, or None.
        sampler_options: Sampler configuration, or None.

    Returns:
        A tuple of (resolved_num_neighbors, resolved_sampler_options).

    Raises:
        ValueError: If neither is provided, or if both provide conflicting
            num_neighbors values.
    """
    if num_neighbors is None and sampler_options is None:
        raise ValueError("Either num_neighbors or sampler_options must be provided.")

    if sampler_options is None:
        assert num_neighbors is not None
        return num_neighbors, KHopNeighborSamplerOptions(num_neighbors)

    if isinstance(sampler_options, KHopNeighborSamplerOptions):
        if num_neighbors is None:
            return sampler_options.num_neighbors, sampler_options
        if num_neighbors != sampler_options.num_neighbors:
            raise ValueError(
                f"num_neighbors ({num_neighbors}) does not match "
                f"sampler_options.num_neighbors ({sampler_options.num_neighbors}). "
                f"Provide one or the other, not both with different values."
            )
        return num_neighbors, sampler_options

    # CustomSamplerOptions — num_neighbors is not meaningful, default to []
    assert isinstance(sampler_options, CustomSamplerOptions)
    if num_neighbors is None:
        return [], sampler_options
    return num_neighbors, sampler_options
