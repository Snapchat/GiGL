"""Install test-only runtime checks for GiGL tensor Shape Contracts.

Test launchers call ``install_runtime_typechecking()`` before test discovery.
Jaxtyping then instruments GiGL and example modules as they are imported, and
Typeguard checks only annotations containing Jaxtyping array types when those
calls run. Production imports do not install this hook.
"""

import atexit
import os
from functools import reduce
from operator import or_
from types import FunctionType, UnionType
from typing import Any, Final, Optional, Union, cast, get_args, get_origin

from jaxtyping import AbstractArray, install_import_hook
from typeguard import typechecked

from gigl.common.logger import Logger

_SHAPE_CONTRACT_PACKAGES: Final[tuple[str, ...]] = ("gigl", "examples")

_import_hook: Optional[object] = None
_instrumented_functions: set[str] = set()
logger = Logger()


def _log_runtime_typechecking_summary() -> None:
    """Log how many Shape Contract functions this process instrumented."""
    logger.info(
        "Runtime Shape Contract checking summary: "
        f"pid={os.getpid()}, instrumented_functions={len(_instrumented_functions)}"
    )


def shape_contract_typechecker(
    function: FunctionType,
) -> FunctionType:
    """Apply Typeguard only to annotations containing Jaxtyping arrays.

    Args:
        function: Function imported from a Shape Contract module.

    Returns:
        Function wrapped to enforce only its Shape Contract annotations.
    """

    def contains_shape_contract(annotation: object) -> bool:
        if isinstance(annotation, type) and issubclass(annotation, AbstractArray):
            return True
        return any(contains_shape_contract(arg) for arg in get_args(annotation))

    def retain_shape_contract(annotation: object) -> object:
        if isinstance(annotation, type) and issubclass(annotation, AbstractArray):
            return annotation
        origin = get_origin(annotation)
        args = get_args(annotation)
        if not args or origin is None:
            return annotation
        if origin in (Union, UnionType):
            retained_args = tuple(
                retain_shape_contract(arg) if contains_shape_contract(arg) else arg
                for arg in args
            )
        else:
            # Existing non-shape members are outside this test-only contract and
            # may contain forward references Typeguard cannot resolve here.
            retained_args = tuple(
                retain_shape_contract(arg) if contains_shape_contract(arg) else Any
                for arg in args
            )
        if hasattr(annotation, "copy_with"):
            return cast(Any, annotation).copy_with(retained_args)
        if origin is UnionType:
            return reduce(or_, retained_args)
        return origin[retained_args]

    annotations = function.__annotations__
    shape_annotations = {
        name: retain_shape_contract(annotation)
        for name, annotation in annotations.items()
        if contains_shape_contract(annotation)
    }
    if not shape_annotations:
        return function
    # Typeguard reads annotations again at call time. A private function copy
    # keeps its shape-only view from replacing the public API annotations.
    checking_function = FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    checking_function.__annotations__ = shape_annotations
    checking_function.__kwdefaults__ = function.__kwdefaults__
    wrapped_function = typechecked(checking_function)
    wrapped_function.__annotations__ = annotations
    _instrumented_functions.add(f"{function.__module__}.{function.__qualname__}")
    return wrapped_function


def install_runtime_typechecking() -> None:
    """Enable test-only runtime checks for tensor Shape Contracts.

    Jaxtyping's simpler, general-purpose setup would be::

        install_import_hook(
            modules=("gigl", "examples"),
            typechecker="typeguard.typechecked",
        )

    That setup makes every annotation in those packages a runtime contract.
    GiGL has static-only annotations, such as TypeVars bounded by Protocols that
    cannot be used with ``isinstance``. The custom typechecker filters those out
    so Typeguard enforces Shape Contracts only.

    Repeated calls are safe. Runtime checking remains scoped to the current
    process and modules imported after this function runs. A contract violation
    raises ``jaxtyping.TypeCheckError`` at the call site, so an uncaught
    violation fails the active test command.
    """
    global _import_hook
    if _import_hook is None:
        _import_hook = install_import_hook(
            modules=_SHAPE_CONTRACT_PACKAGES,
            typechecker="tests.test_assets.runtime_type_checking.shape_contract_typechecker",
        )
        atexit.register(_log_runtime_typechecking_summary)
        logger.info(
            "Runtime Shape Contract checking enabled: "
            f"pid={os.getpid()}, packages={_SHAPE_CONTRACT_PACKAGES}"
        )
