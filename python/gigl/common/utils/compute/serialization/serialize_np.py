from typing import Tuple, TypedDict

import msgpack
import numpy as np

from gigl.common.utils.compute.serialization.coder import CoderProtocol


class EncodedNdArray(TypedDict):
    dtype: str
    shape: Tuple
    data: bytes


class NumpyCoder(CoderProtocol[np.ndarray]):
    def encode(self, obj: np.ndarray) -> bytes:
        return msgpack.dumps(
            obj, default=self.__encode_nd_array_helper, use_bin_type=True
        )

    def decode(self, byte_str: bytes) -> np.ndarray:
        return msgpack.loads(
            byte_str, object_hook=self.__decode_nd_array_helper, raw=False
        )

    @staticmethod
    def __decode_nd_array_helper(obj: EncodedNdArray):
        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])

    @staticmethod
    def __encode_nd_array_helper(array: np.ndarray) -> EncodedNdArray:
        # Using array.data is a slight optimization given that we can use it
        serialized_array: bytes = (
            array.data if array.flags["C_CONTIGUOUS"] else array.tobytes()
        )

        if array.dtype == object:
            raise TypeError(f"can't convert np.ndarray of type {array.dtype}")

        return {
            "dtype": str(array.dtype),
            "shape": array.shape,
            "data": serialized_array,
        }
