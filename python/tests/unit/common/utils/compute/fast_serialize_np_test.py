import unittest

import numpy as np

from gigl.common.utils.compute.serialization.serialize_np import NumpyCoder


class FastSerializeNpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.numpy_coder = NumpyCoder()

    def test_encode_decode_np_array(self):
        a1 = np.array([[10, 20, 30], [11, 12, 13]])
        a2 = self.numpy_coder.decode(byte_str=self.numpy_coder.encode(obj=a1))
        self.assertTrue(np.array_equal(a1, a2))

        a1 = np.array([[10.1, 20.2, 30.3], [11.4, 12.5, 13.2222]])
        a2 = self.numpy_coder.decode(byte_str=self.numpy_coder.encode(obj=a1))
        self.assertTrue(np.array_equal(a1, a2))

        a1 = np.array(True)
        a2 = self.numpy_coder.decode(byte_str=self.numpy_coder.encode(obj=a1))
        self.assertTrue(np.array_equal(a1, a2))
