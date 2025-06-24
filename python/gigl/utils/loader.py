import collections

from gigl.common.logger import Logger

logger = Logger()


class InfiniteIterator(collections.abc.Iterator):
    """
    A wrapper around iterators (objects with __iter__ and __next__ methods) that loop indefinitely over the data.
    """

    def __init__(self, loader: collections.abc.Iterator):
        self.loader = loader
        # We don't expect the iter to be called outside of the class
        self._iter = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            logger.info("InfiniteIterator restarting the internal iterator")
            self._iter = iter(self.loader)
            return next(self._iter)
