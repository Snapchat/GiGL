from typing import Callable, TypeVar

import tensorflow as tf

_ReturnType = TypeVar("_ReturnType")  # Generic Return Type of function for decorator


def tf_on_cpu(func: Callable[..., _ReturnType]) -> Callable[..., _ReturnType]:
    """
    A decorator to run a function using TensorFlow's CPU device.
    """

    def wrapper(*args, **kwargs) -> _ReturnType:
        with tf.device("/CPU:0"):
            result = func(*args, **kwargs)
        return result

    return wrapper
