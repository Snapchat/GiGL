import unittest

from gigl.utils.iterator import InfiniteIterator


class IteratorTest(unittest.TestCase):
    def test_infinite_iterator(self):
        iterable_list: list[int] = [3, 4, 5, 6]
        expected_list: list[int] = [3, 4, 5, 6, 3, 4, 5, 6, 3, 4]
        actual_list: list[int] = []
        infinite_iterator = InfiniteIterator(iterable=iterable_list)
        count = 0
        for item in infinite_iterator:
            if count == len(expected_list):
                break
            actual_list.append(item)
            count += 1
        self.assertEqual(actual_list, expected_list)


if __name__ == "__main__":
    unittest.main()
