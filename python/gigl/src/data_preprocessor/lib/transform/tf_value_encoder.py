from typing import Any, AnyStr, List, Union

import tensorflow as tf


class TFValueEncoder:
    @staticmethod
    def get_value_to_impute(dtype: tf.dtypes.DType) -> Union[int, str, float]:
        """
        Returns the default value to use for a missing field.
        :param dtype:
        :return:
        """

        if dtype.is_integer:
            return 0
        elif dtype.is_bool:
            return 0
        elif dtype.is_floating:
            return 0.0
        else:
            return "MISSING"

    @staticmethod
    def __bytes_values_to_tf_feature(value: List[AnyStr]) -> tf.train.Feature:
        """
        Returns a bytes_list from a string / byte (or list of such).
        """

        if isinstance(value, type(tf.constant(0))):
            value = (
                value.numpy()
            )  # BytesList won't unpack a string from an EagerTensor.

        value_bytes: List[bytes] = []
        for v in value:
            if isinstance(v, str):
                value_bytes.append(v.encode("utf-8"))
            elif isinstance(v, bytes):
                value_bytes.append(v)
            else:
                raise TypeError(f"Got object of type {type(v)} (must be bytes or str)")
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_bytes))

    @staticmethod
    def __float_values_to_tf_feature(value: List[float]) -> tf.train.Feature:
        """
        Returns a float_list from a float / double (or list of such).
        """

        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def __int_values_to_tf_feature(value: List[int]) -> tf.train.Feature:
        """
        Returns an int64_list from a bool / enum / int / uint (or list of such).
        """

        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def encode_value_as_feature(value: Any, dtype: tf.dtypes.DType) -> tf.train.Feature:
        """
        Try to encode a given "raw value" as a tf.train.Feature of the intended type.
        Imputes missing values according to defaults for their dtype.

        :param value:
        :param dtype:
        :return:
        """
        # prepare value
        if value is None:
            value = TFValueEncoder.get_value_to_impute(dtype=dtype)
        if not isinstance(value, list):
            value = [value]

        # encode value
        if dtype.is_integer or dtype.is_bool:
            tf_feature = TFValueEncoder.__int_values_to_tf_feature(value=value)
        elif dtype.is_floating:
            tf_feature = TFValueEncoder.__float_values_to_tf_feature(value=value)
        else:
            tf_feature = TFValueEncoder.__bytes_values_to_tf_feature(value=value)

        return tf_feature
