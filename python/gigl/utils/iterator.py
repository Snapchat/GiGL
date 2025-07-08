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
        self._shutdowned = False

    def __iter__(self) -> Iterator[_T]:
        return self

    def __next__(self) -> _T:
        if self._shutdowned:
            raise ValueError(
                "InfiniteIterator has been shut down, but attempted to access the next item. Ensure that the InfiniteIterator is not used after being shut down."
            )
        try:
            return next(self._iter)
        except StopIteration:
            logger.info("InfiniteIterator restarting the internal iterator")
            self._iter = iter(self._iterable)
            return next(self._iter)

    def shutdown(self):
        if self._shutdowned:
            return
        self._shutdowned = True
        if isinstance(self._iterable, DistLoader):
            logger.info("Shutting down dataloader ...")
            self._iterable.shutdown()
