import itertools
from typing import Iterator


def batched(it: Iterator, n: int):
    """
    Create batches of up to n elements from an iterator.

    Takes an input iterator and yields sub-iterators, each containing up to n elements.
    This is useful for processing data in chunks or creating batched operations for
    efficient data pipeline processing.

    Args:
        it (Iterator): The input iterator to batch.
        n (int): Maximum number of elements per batch. Must be >= 1.

    Yields:
        Iterator: Sub-iterators containing up to n elements from the input iterator.
                 The last batch may contain fewer than n elements if the input
                 iterator is exhausted.

    Raises:
        AssertionError: If n < 1.

    Example:
        >>> data = iter([1, 2, 3, 4, 5, 6, 7])
        >>> for batch in batched(data, 3):
        ...     print(list(batch))
        [1, 2, 3]
        [4, 5, 6]
        [7]
    """
    assert n >= 1
    for x in it:
        yield itertools.chain((x,), itertools.islice(it, n - 1))
