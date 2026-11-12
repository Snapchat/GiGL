import gc
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def freeze_gc() -> Iterator[None]:
    """Freezes the current heap so `gc.collect()` calls only scan newly created objects.

    After imports a process can hold ~1M tracked objects, so each full `gc.collect()` takes ~0.6s.
    `gc.freeze()` moves existing objects to the permanent generation, which later collections skip.
    Cyclic garbage created inside the block is still collected.

    If the heap is already frozen by an outer caller, this does nothing, so nested use
    does not unfreeze the outer freeze early.

    Can be used as a context manager or as a decorator.

    Example:
        >>> with freeze_gc():
        ...     gc.collect()
        >>> @freeze_gc()
        ... def build() -> None:
        ...     gc.collect()

    Yields:
        None
    """
    if gc.get_freeze_count() > 0:
        yield
        return
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()
