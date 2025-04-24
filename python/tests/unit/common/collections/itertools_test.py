import unittest

from gigl.common.collections.itertools import batch


class ItertoolsTest(unittest.TestCase):
    def test_batch(self):
        input_list = [1, 2, 3, 4, 5]
        output = batch(list_of_items=input_list, chunk_size=2)
        expected_output = [[1, 2], [3, 4], [5]]
        self.assertEquals(output, expected_output)
