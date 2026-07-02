import tensorflow as tf
import tensorflow_transform as tft

from gigl.common.logger import Logger
from gigl.types.graph import FeatureQuantizationMetadata

logger = Logger()


def quantize_tft_tensor(
    x: tft.common_types.ConsistentTensorType,
    bits: int,
) -> tuple[tf.Tensor, FeatureQuantizationMetadata]:
    """Quantize the final tft tensor dimension into packed uint8 codes."""
    VALID_BITS = (1, 2, 4, 8)
    if bits not in VALID_BITS:
        raise ValueError(f"bits must be one of {VALID_BITS}, got {bits}")
    if isinstance(x, tf.SparseTensor):
        raise ValueError("Feature quantization expects a dense numerical tensor.")
    if x.shape.rank != 2:
        raise ValueError(f"Feature quantization expects a 2D tensor, got {x.shape}.")

    x = tf.cast(x, tf.float32)
    q = _build_quantization_metadata(x, bits)
    if bits == 1:
        # 1-bit quantization keeps only sign; values restore from neg/pos means.
        codes = tf.cast(x > 0, tf.uint8)
    else:
        # Linearly map clipped values into integer buckets.
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
    q = FeatureQuantizationMetadata(
        bits=bits,
        packed_feature_dim=(dim + per_byte - 1) // per_byte,
        dequantized_feature_dim=dim,
    )
    if bits == 1:
        # Store mean values for negative and positive buckets.
        x = tf.reshape(x, [-1])
        neg = tf.cast(x <= 0, tf.float32)
        pos = tf.cast(x > 0, tf.float32)
        q.neg_mean = tf.math.divide_no_nan(tft.sum(x * neg), tft.sum(neg))
        q.pos_mean = tf.math.divide_no_nan(tft.sum(x * pos), tft.sum(pos))
        tf.debugging.check_numerics(q.neg_mean, "non-finite neg_mean.")
        tf.debugging.check_numerics(q.pos_mean, "non-finite pos_mean.")
        logger.info(
            "Computed 1-bit quantization stats: "
            f"neg_mean={q.neg_mean}, pos_mean={q.pos_mean}"
        )
    else:
        # Store symmetric clip bounds from the 99.5th abs-value percentile.
        x_abs = tf.reshape(tf.abs(x), [-1])
        bounds = tft.quantiles(x_abs, num_buckets=1000, epsilon=0.001)
        # TFT returns boundaries for k / 1000, k=1..999; 0.995 is index 994.
        q.clip_max = tf.maximum(bounds[..., 994], 1e-5)
        q.clip_min = -q.clip_max
        tf.debugging.check_numerics(q.clip_min, "non-finite clip_min.")
        tf.debugging.check_numerics(q.clip_max, "non-finite clip_max.")
        logger.info(
            f"Computed {bits}-bit quantization stats: "
            f"clip_min={q.clip_min}, clip_max={q.clip_max}"
        )
    logger.info(f"Built feature quantization metadata: {q}")
    return q


def _pack_tft_tensor(codes: tf.Tensor, bits: int) -> tf.Tensor:
    # Cast to int32 for integer shift/multiply/sum packing math.
    codes = tf.cast(codes, tf.int32)
    # Pad only the feature dimension of this 2D [row, feature] tensor.
    per_byte = 8 // bits
    pad = tf.math.floormod(-tf.shape(codes)[-1], per_byte)
    codes = tf.pad(codes, [[0, 0], [0, pad]])
    # Group the padded feature dimension into chunks that each form one byte.
    codes = tf.reshape(codes, [tf.shape(codes)[0], -1, per_byte])
    # Place the first code in each chunk into the highest bits of the byte.
    shifts = bits * tf.range(per_byte - 1, -1, -1, dtype=tf.int32)
    weights = tf.bitwise.left_shift(tf.ones_like(shifts), shifts)
    return tf.cast(tf.reduce_sum(codes * weights, axis=-1), tf.uint8)
