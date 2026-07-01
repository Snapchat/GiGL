from collections.abc import Sequence

import tensorflow as tf
import tensorflow_transform as tft

from gigl.types.graph import FeatureQuantizationMetadata


def quantize_tft_tensor(
    x: tft.common_types.ConsistentTensorType,
    bits: int,
    dequantized_feature_keys: Sequence[str] = (),
) -> tuple[tf.Tensor, FeatureQuantizationMetadata]:
    """Quantize a dense TFT tensor and pack codes along its last dimension.

    Quantization runs in TensorFlow/TFT preprocessing. GiGL training later unpacks
    and dequantizes the emitted packed tensor with Torch. By convention, 1-bit
    uses two centroids and 2/4/8-bit uses linear buckets.
    """
    VALID_BITS = (1, 2, 4, 8)
    if bits not in VALID_BITS:
        raise ValueError(f"bits must be one of {VALID_BITS}.")
    if isinstance(x, tf.SparseTensor):
        raise ValueError("Feature quantization expects a dense numerical tensor.")
    x = tf.cast(x, tf.float32)
    q = compute_feature_quantization_metadata(
        x=x, bits=bits, dequantized_feature_keys=dequantized_feature_keys
    )
    if bits == 1:
        codes = tf.cast(x > 0, tf.uint8)
    else:
        levels = (1 << bits) - 1
        clipped = tf.clip_by_value(x, q.clip_min, q.clip_max)
        scaled = (clipped - q.clip_min) / (q.clip_max - q.clip_min)
        codes = tf.cast(tf.round(scaled * levels), tf.uint8)
    return pack_tft_feature_codes(codes, bits), q


def compute_feature_quantization_metadata(
    x: tft.common_types.ConsistentTensorType,
    bits: int,
    dequantized_feature_keys: Sequence[str] = (),
) -> FeatureQuantizationMetadata:
    """Compute global metadata: 1-bit centroid, 2/4/8-bit linear buckets."""
    dim = x.shape[-1]
    if dim is None:
        raise ValueError("Feature quantization expects a known final dimension.")
    dim = int(dim)
    per_byte = 8 // bits
    kwargs = dict(
        bits=bits,
        packed_feature_dim=(dim + per_byte - 1) // per_byte,
        dequantized_feature_dim=dim,
        dequantized_feature_keys=tuple(dequantized_feature_keys)
    )
    if bits == 1:
        x_flat = tf.reshape(x, [-1])
        pos = tf.reshape(x > 0, [-1])
        neg = tf.logical_not(pos)
        zeros = tf.zeros_like(x_flat)
        return FeatureQuantizationMetadata(
            **kwargs,
            bucket_0_value=tf.math.divide_no_nan(
                tft.sum(tf.where(neg, x_flat, zeros)), tft.sum(tf.cast(neg, tf.float32))
            ),
            bucket_1_value=tf.math.divide_no_nan(
                tft.sum(tf.where(pos, x_flat, zeros)), tft.sum(tf.cast(pos, tf.float32))
            )
        )

    x_abs = tf.reshape(tf.abs(x), [-1])
    bounds = tft.quantiles(x_abs, num_buckets=1000, epsilon=0.001)
    # TFT returns boundaries for k / 1000, k=1..999; 0.995 is index 994.
    fmax = tf.maximum(bounds[..., 994], 1e-5)
    clip_abs = tf.maximum(fmax, 1e-5)
    return FeatureQuantizationMetadata(**kwargs, clip_min=-clip_abs, clip_max=clip_abs)


def pack_tft_feature_codes(codes: tf.Tensor, bits: int) -> tf.Tensor:
    """Pack integer codes along the last dim, high-bits-first within each byte.

    This must match Torch unpacking in ``torch_ops.py``. Example: 2-bit codes
    [a, b, c, d] are packed as byte ``a << 6 | b << 4 | c << 2 | d``.
    """
    codes = tf.cast(codes, tf.int32)
    per_byte = 8 // bits
    pad = tf.math.floormod(-tf.shape(codes)[-1], per_byte)
    rank = tf.rank(codes)
    leading_padding = tf.zeros(tf.stack([rank - 1, 2]), dtype=tf.int32)
    last_padding = tf.expand_dims(tf.stack([0, pad]), axis=0)
    codes = tf.pad(codes, tf.concat([leading_padding, last_padding], axis=0))

    padded_shape = tf.shape(codes)
    new_shape = tf.concat(
        [padded_shape[:-1], tf.constant([-1, per_byte], dtype=tf.int32)], axis=0
    )
    codes = tf.reshape(codes, new_shape)
    shifts = bits * tf.range(per_byte - 1, -1, -1, dtype=tf.int32)
    weights = tf.bitwise.left_shift(tf.ones_like(shifts), shifts)
    return tf.cast(tf.reduce_sum(codes * weights, axis=-1), tf.uint8)
