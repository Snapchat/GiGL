import tensorflow as tf
import tensorflow_transform as tft

from gigl.types.graph import FeatureQuantizationMetadata


def quantize_tft_tensor(
    x: tft.common_types.ConsistentTensorType,
    bits: int,
) -> tuple[tf.Tensor, FeatureQuantizationMetadata]:
    VALID_BITS = (1, 2, 4, 8)
    if bits not in VALID_BITS:
        raise ValueError(f"bits must be one of {VALID_BITS}, got {bits}")
    if isinstance(x, tf.SparseTensor):
        raise ValueError("Feature quantization expects a dense numerical tensor.")
    x = tf.cast(x, tf.float32)
    q = _build_quantization_metadata(x, bits)
    if bits == 1:
        codes = tf.cast(x > 0, tf.uint8)
    else:
        levels = (1 << bits) - 1
        clipped = tf.clip_by_value(x, q.clip_min, q.clip_max)
        scaled = (clipped - q.clip_min) / (q.clip_max - q.clip_min)
        codes = tf.cast(tf.round(scaled * levels), tf.uint8)
    return _pack_tft_tensor(codes, bits), q


def _build_quantization_metadata(
    x: tft.common_types.ConsistentTensorType,
    bits: int,
) -> FeatureQuantizationMetadata:
    dim = x.shape[-1]
    if dim is None:
        raise ValueError("Feature quantization expects a known final dimension.")
    dim = int(dim)
    per_byte = 8 // bits
    q = FeatureQuantizationMetadata(bits, (dim + per_byte - 1) // per_byte, dim, ())
    if bits == 1:
        flat = tf.reshape(x, [-1])
        neg = tf.cast(flat <= 0, tf.float32)
        pos = tf.cast(flat > 0, tf.float32)
        q.bucket_0_value = tf.math.divide_no_nan(tft.sum(flat * neg), tft.sum(neg))
        q.bucket_1_value = tf.math.divide_no_nan(tft.sum(flat * pos), tft.sum(pos))
    else:
        x_abs = tf.reshape(tf.abs(x), [-1])
        bounds = tft.quantiles(x_abs, num_buckets=1000, epsilon=0.001)
        # TFT returns boundaries for k / 1000, k=1..999; 0.995 is index 994.
        q.clip_max = tf.maximum(bounds[..., 994], 1e-5)
        q.clip_min = -q.clip_max
    return q


def _pack_tft_tensor(codes: tf.Tensor, bits: int) -> tf.Tensor:
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
