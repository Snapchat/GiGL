import torch

from gigl_core import add_one
from tests.test_assets.test_case import TestCase


class TestAddOne(TestCase):
    def test_add_one_returns_elementwise_increment(self) -> None:
        actual = add_one(torch.tensor([1, 2, 3]))
        expected = torch.tensor([2, 3, 4])
        self.assertTrue(torch.equal(actual, expected))

    def test_add_one_is_out_of_place(self) -> None:
        original = torch.tensor([1.0, 2.0])
        _ = add_one(original)
        self.assertTrue(torch.equal(original, torch.tensor([1.0, 2.0])))

    def test_add_one_rejects_cuda_tensor(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        with self.assertRaises(RuntimeError):
            add_one(torch.tensor([1, 2, 3], device="cuda"))
