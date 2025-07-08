import unittest
from typing import Final

from gigl.utils.iterator import InfiniteIterator

_ITERABLE_LIST: Final[list[int]] = [3, 4, 5, 6]


class IteratorTest(unittest.TestCase):
    def test_infinite_iterator(self):
        expected_list: list[int] = [3, 4, 5, 6, 3, 4, 5, 6, 3, 4]
        actual_list: list[int] = []
        infinite_iterator = InfiniteIterator(iterable=_ITERABLE_LIST)
        count = 0
        for item in infinite_iterator:
            if count == len(expected_list):
                break
            actual_list.append(item)
            count += 1
        self.assertEqual(actual_list, expected_list)

    def test_infinite_iterator_shutdown(self):
        infinite_iterator = InfiniteIterator(iterable=_ITERABLE_LIST)
        for _ in infinite_iterator:
            break
        infinite_iterator.shutdown()
        with self.assertRaises(ValueError):
            for _ in infinite_iterator:
                break


if __name__ == "__main__":
    unittest.main()
