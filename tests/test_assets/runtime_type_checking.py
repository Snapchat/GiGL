"""Enable runtime tensor contracts before test discovery imports GiGL modules."""

from collections.abc import Callable
from typing import Any, Final, Optional, get_args

from beartype import beartype
from jaxtyping import AbstractArray, install_import_hook

_SHAPE_CONTRACT_MODULES: Final[tuple[str, ...]] = (
    "gigl.distributed.base_sampler",
    "gigl.distributed.dist_ablp_neighborloader",
    "gigl.distributed.dist_ppr_sampler",
    "gigl.distributed.distributed_neighborloader",
    "gigl.distributed.sampler",
    "gigl.nn.graph_transformer",
    "gigl.nn.loss",
    "gigl.nn.models",
    "gigl.src.common.models.layers.decoder",
    "gigl.src.common.models.layers.feature_interaction",
    "gigl.src.common.models.layers.loss",
    "gigl.src.common.models.layers.task",
    "gigl.src.common.models.pyg.heterogeneous",
    "gigl.src.common.models.pyg.homogeneous",
    "gigl.src.common.models.pyg.link_prediction",
    "gigl.src.common.models.pyg.nn.models.feature_embedding",
    "gigl.src.common.models.pyg.nn.models.feature_interaction",
    "gigl.src.common.models.pyg.nn.models.jumping_knowledge",
    "gigl.src.common.types.task_inputs",
    "gigl.transforms.graph_transformer",
)

_import_hook: Optional[object] = None


def _contains_shape_contract(annotation: object) -> bool:
    if isinstance(annotation, type) and issubclass(annotation, AbstractArray):
        return True
    return any(_contains_shape_contract(arg) for arg in get_args(annotation))


def shape_contract_typechecker(
    function: Callable[..., Any],
) -> Callable[..., Any]:
    """Apply Beartype only to annotations containing Jaxtyping arrays.

    Args:
        function: Function imported from a Shape Contract module.

    Returns:
        Function wrapped to enforce only its Shape Contract annotations.
    """
    annotations = function.__annotations__
    shape_annotations = {
        name: annotation
        for name, annotation in annotations.items()
        if _contains_shape_contract(annotation)
    }
    function.__annotations__ = shape_annotations
    try:
        wrapped_function = beartype(function)
    finally:
        function.__annotations__ = annotations
    wrapped_function.__annotations__ = annotations
    return wrapped_function


def install_runtime_typechecking() -> None:
    """Enable runtime checks for modules declaring tensor Shape Contracts.

    Repeated calls are safe. Runtime checking remains scoped to the current
    process and modules imported after this function runs. A contract violation
    raises ``jaxtyping.TypeCheckError`` at the call site, so an uncaught
    violation fails the active test command.
    """
    global _import_hook
    if _import_hook is None:
        _import_hook = install_import_hook(
            modules=_SHAPE_CONTRACT_MODULES,
            typechecker=(
                "tests.test_assets.runtime_type_checking.shape_contract_typechecker"
            ),
        )
