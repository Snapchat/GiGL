import unittest

import torch

from gigl.common.logger import Logger
from gigl.src.common.models.layers.count_min_sketch import CountMinSketch

logger = Logger()


class CountMinSketchTest(unittest.TestCase):
    def test_count(self):
        # Initialize the CountMinSketch object
        cms = CountMinSketch(width=20, depth=5)
        candidate_ids = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=torch.long)
        cms.add_torch_long_tensor(candidate_ids)  # type: ignore
        # Check the total count
        self.assertEqual(cms.total(), 10)
        # Check the estimated count
        self.assertEqual(cms.estimate(1), 1)
        self.assertEqual(cms.estimate(2), 2)
        self.assertEqual(cms.estimate(3), 3)
        self.assertEqual(cms.estimate(4), 4)


if __name__ == "__main__":
    unittest.main()
