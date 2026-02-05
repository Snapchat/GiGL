from gigl.common.collections.itertools import batch
from tests.test_assets.test_case import TestCase


class ItertoolsTest(TestCase):
    def test_batch(self):
        input_list = [1, 2, 3, 4, 5]
        output = batch(list_of_items=input_list, chunk_size=2)
        expected_output = [[1, 2], [3, 4], [5]]
        self.assertEquals(output, expected_output)
