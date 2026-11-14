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
import torch
from jaxtyping import AbstractArray, Float, install_import_hook
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
    Nested functions that do have Shape Contracts stay wrapped, so they cannot
    be pickled by value under test.

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


# The config class is private to Jaxtyping, so it is read from the config object.
_JaxtypingConfig: Any = cast(Any, type(jaxtyping.config))


class _SkipWhileCompilingConfig(_JaxtypingConfig):
    """Jaxtyping config whose ``jaxtyping_disable`` reads true while Dynamo traces.

    Jaxtyping's wrapper returns the undecorated function when ``jaxtyping_disable``
    is true, before it binds arguments or pushes a shape memo. Dynamo cannot trace
    that bookkeeping or Typeguard, and guards on it cause recompiles. Eager calls
    read the configured value, so they are still checked.
    """

    @property
    def jaxtyping_disable(self) -> bool:
        if torch.compiler.is_compiling():
            return True
        return self._eager_jaxtyping_disable

    @jaxtyping_disable.setter
    def jaxtyping_disable(self, value: bool) -> None:
        self._eager_jaxtyping_disable = value


def _skip_shape_checks_while_compiling() -> None:
    """Keeps Shape Contract wrappers out of ``torch.compile``.

    Inside a compiled frame, ``jaxtyping.config`` reads ``jaxtyping_disable`` as
    true, so the inlined wrapper calls the undecorated function.

    Every wrapped function shares the code object of Jaxtyping's wrapper. Dynamo
    caches compiled frames per code object, so a wrapper that starts a frame
    after a graph break would count each contracted function as a recompile of
    the same code. ``skip_code`` runs that frame eagerly, with its checks, and
    compiles the function it calls as its own frame.
    """
    config = jaxtyping.config
    if isinstance(config, _SkipWhileCompilingConfig):
        return
    eager_jaxtyping_disable = config.__dict__.pop("jaxtyping_disable")
    config.__class__ = _SkipWhileCompilingConfig
    config.jaxtyping_disable = eager_jaxtyping_disable

    def probe(x: Float[torch.Tensor, "_"]) -> Float[torch.Tensor, "_"]:
        return x

    wrapper = jaxtyping.jaxtyped(typechecker=shape_contract_typechecker)(probe)
    torch._dynamo.eval_frame.skip_code(wrapper.__code__)


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

    Checks run in eager calls and are skipped while ``torch.compile`` traces.

    Repeated calls are safe. Runtime checking remains scoped to the current
    process and modules imported after this function runs. A contract violation
    raises ``jaxtyping.TypeCheckError`` at the call site, so an uncaught
    violation fails the active test command.
    """
    global _import_hook
    if _import_hook is None:
        _skip_shape_checks_while_compiling()
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
