"""Install test-only runtime checks for GiGL tensor Shape Contracts.

Test launchers call ``install_runtime_typechecking()`` before test discovery.
Jaxtyping then instruments GiGL and example modules as they are imported, and
Typeguard checks only annotations containing Jaxtyping array types when those
calls run. Production imports do not install this hook.
"""

import atexit
import os
import threading
from functools import reduce
from operator import or_
from types import FunctionType, UnionType
from typing import Any, Callable, Final, Optional, Union, cast, get_args, get_origin

import jaxtyping
from jaxtyping import AbstractArray, install_import_hook
from typeguard import typechecked

from gigl.common.logger import Logger

_SHAPE_CONTRACT_PACKAGES: Final[tuple[str, ...]] = ("gigl", "examples")

_import_hook: Optional[object] = None
_instrumented_functions: set[str] = set()
# Per thread: whether shape_contract_typechecker found a Shape Contract while
# _jaxtyped_shape_contracts_only decorates a function. Each decoration sets it
# to None first, so None afterwards means the typechecker did not run for that
# function. Jaxtyping also calls the typechecker outside any decoration, when a
# call-time check fails, so the value between decorations means nothing.
_decoration = threading.local()
logger = Logger()


def shape_contract_typechecker(function: FunctionType) -> FunctionType:
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
    _decoration.found_shape_contract = bool(
        getattr(_decoration, "found_shape_contract", None)
    ) or bool(shape_annotations)
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


_jaxtyped = jaxtyping.jaxtyped


def _jaxtyped_shape_contracts_only(*args: Any, **kwargs: Any) -> Any:
    """Keep Jaxtyping's wrapper only on functions with Shape Contracts.

    The import hook decorates every function with
    ``jaxtyping.jaxtyped(typechecker=...)``. For a function without Shape
    Contracts the wrapper checks nothing, since shape_contract_typechecker
    leaves it unchecked, but it still closes over a ``weakref``. Cloudpickle
    cannot pickle a weakref, and Beam pickles nested functions such as a TFT
    ``preprocessing_fn`` by value, so the wrapper would make them unpicklable
    only under test. Such functions keep their original, undecorated object.

    Calls that do not match the import hook's form, and calls where
    shape_contract_typechecker does not run, go to Jaxtyping unchanged.
    """
    if args:
        return _jaxtyped(*args, **kwargs)
    decorate = _jaxtyped(**kwargs)

    def decorator(function: Callable[..., Any]) -> Any:
        if not isinstance(function, FunctionType):
            return decorate(function)
        outer_found = getattr(_decoration, "found_shape_contract", None)
        _decoration.found_shape_contract = None
        try:
            wrapped_function = decorate(function)
            found_shape_contract = _decoration.found_shape_contract
        finally:
            _decoration.found_shape_contract = outer_found
        return function if found_shape_contract is False else wrapped_function

    return decorator


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
        # Hooked modules look up ``jaxtyping.jaxtyped`` when each decorator runs.
        setattr(jaxtyping, "jaxtyped", _jaxtyped_shape_contracts_only)
        _import_hook = install_import_hook(
            modules=_SHAPE_CONTRACT_PACKAGES,
            typechecker="tests.test_assets.runtime_type_checking.shape_contract_typechecker",
        )
        atexit.register(
            lambda: logger.info(
                f"Shape checks: pid={os.getpid()} count={len(_instrumented_functions)}"
            )
        )
        logger.info(f"Shape checks enabled: pid={os.getpid()}")
