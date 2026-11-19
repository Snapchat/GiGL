from typing import Callable, TypeVar

_ReturnType = TypeVar("_ReturnType")  # Generic Return Type of function for decorator


def tf_on_cpu(func: Callable[..., _ReturnType]) -> Callable[..., _ReturnType]:
    """
    A decorator to run a function using TensorFlow's CPU device.
    """

    def wrapper(*args, **kwargs) -> _ReturnType:
        # Imported here so that importing this module does not load TensorFlow.
        import tensorflow as tf

        with tf.device("/CPU:0"):
            result = func(*args, **kwargs)
        return result

    return wrapper
