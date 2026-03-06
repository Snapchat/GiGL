"""Sampler option types for configuring which sampler class to use in distributed loading.

Provides two options:
- ``NeighborSamplerOptions``: Uses GiGL's built-in ``DistNeighborSampler``.
- ``CustomSamplerOptions``: Dynamically imports and uses a user-provided sampler class.

Both are frozen dataclasses so they are safe to pickle across RPC boundaries
(required for Graph Store mode).
"""

import importlib
from dataclasses import dataclass, field
from typing import Any, Union

from graphlearn_torch.typing import EdgeType


@dataclass(frozen=True)
class NeighborSamplerOptions:
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


SamplerOptions = Union[NeighborSamplerOptions, CustomSamplerOptions]


def resolve_sampler_class(sampler_options: SamplerOptions) -> type:
    """Resolve a sampler class from the given options.

    Args:
        sampler_options: Either ``NeighborSamplerOptions`` (returns the built-in
            ``DistNeighborSampler``) or ``CustomSamplerOptions`` (dynamically
            imports the class at ``class_path``).

    Returns:
        The sampler class to instantiate.

    Raises:
        TypeError: If ``sampler_options`` is not a recognized type.
        ImportError: If the module in ``class_path`` cannot be imported.
        AttributeError: If the class name is not found in the module.
    """
    if isinstance(sampler_options, NeighborSamplerOptions):
        from gigl.distributed.dist_neighbor_sampler import DistNeighborSampler

        return DistNeighborSampler
    elif isinstance(sampler_options, CustomSamplerOptions):
        module_path, class_name = sampler_options.class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        raise TypeError(
            f"Unsupported sampler_options type: {type(sampler_options)}. "
            f"Expected NeighborSamplerOptions or CustomSamplerOptions."
        )
