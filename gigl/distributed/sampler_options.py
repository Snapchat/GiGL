"""Sampler option types for configuring which sampler class to use in distributed loading.

Provides two options:
- ``KHopNeighborSamplerOptions``: Uses GiGL's built-in ``DistNeighborSampler``.
- ``CustomSamplerOptions``: Dynamically imports and uses a user-provided sampler class.

Both are frozen dataclasses so they are safe to pickle across RPC boundaries
(required for Graph Store mode).
"""

from dataclasses import dataclass, field
from typing import Any, Union

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
