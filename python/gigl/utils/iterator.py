from collections.abc import Iterator

from gigl.common.logger import Logger

logger = Logger()

from typing import TypeVar

_T = TypeVar("_T")


class InfiniteIterator(Iterator[_T]):
    """
    A wrapper around iterators (objects with __iter__ and __next__ methods) that loop indefinitely over the data.
    """

    def __init__(self, iterable: Iterator):
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
