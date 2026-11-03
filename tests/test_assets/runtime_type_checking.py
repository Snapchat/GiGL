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

import jaxtyping
import torch
from jaxtyping import AbstractArray, Float, install_import_hook
from typeguard import typechecked

from gigl.common.logger import Logger

_SHAPE_CONTRACT_PACKAGES: Final[tuple[str, ...]] = ("gigl", "examples")

_import_hook: Optional[object] = None
_instrumented_functions: set[str] = set()
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
