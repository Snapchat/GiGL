import functools
from typing import Any, Callable, Optional, TypeVar, Union, no_type_check, overload

RT = TypeVar("RT", bound=Callable[..., Any])

# Just wrapper for functool functions to support type checking


@overload
def lru_cache(maxsize: Callable[..., RT], typed: bool = False) -> Callable[..., RT]:
    ...


@overload
def lru_cache(maxsize: Optional[int], typed: bool = False) -> Callable[[RT], RT]:
    ...


@overload
def lru_cache(
    maxsize: Union[Callable[..., RT], Optional[int]], typed: bool = False
) -> Union[Callable[..., RT], Callable[[RT], RT]]:
    ...


@no_type_check
def lru_cache(*args, **kwargs):
    return functools.lru_cache(*args, **kwargs)
