import gc
from collections.abc import Iterable, Iterator
from typing import TypeVar

from graphlearn_torch.distributed import DistLoader

from gigl.common.logger import Logger

logger = Logger()

_T = TypeVar("_T")


class InfiniteIterator(Iterator[_T]):
    """
    A wrapper around iterators (objects with __iter__ and __next__ methods) that loop indefinitely over the data.
    """

    def __init__(self, iterable: Iterable[_T]):
        self._iterable = iterable
        self._iter = iter(iterable)

    def __iter__(self) -> Iterator[_T]:
        return self

    def __next__(self) -> _T:
        try:
            return next(self._iter)
        except StopIteration:
            logger.info("InfiniteIterator restarting the internal iterator")
            self._iter = iter(self._iterable)
            return next(self._iter)

    def shutdown(self):
        if isinstance(self._iterable, DistLoader):
            self._iterable.shutdown()
        del self._iter
        del self._iterable
        gc.collect()
