"""NumPy feature quantization helpers for preprocessing.

Quantization runs in the data preprocessor, where feature data is stored as CPU
arrays and torch is not available. Dequantization lives in torch_ops.py because
the dataloader collate path operates on torch tensors that may already be on GPU.
"""

import numpy as np
from jaxtyping import Float, UInt8

from gigl.common.utils.feature_quantization import SUPPORTED_QUANTIZATION_BITS


def quantize_ndarray(
    features: Float[np.ndarray, "entities feature_dim"],
    *,
    bits: int,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> UInt8[np.ndarray, "entities packed_feature_dim"]:
    """Quantize a 2D float array into packed uint8 codes.

    For multi-bit quantization, `clip_min` and `clip_max` are required and
    define the min-max scaling range: values are clipped to that range, scaled
    to `[0, 2**bits - 1]`, rounded to integer codes, then packed into bytes.
    """
    if bits not in SUPPORTED_QUANTIZATION_BITS:
        raise ValueError(
            f"bits must be one of {SUPPORTED_QUANTIZATION_BITS}, got {bits}"
        )
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, got shape {features.shape}.")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite; got NaN or Inf")
    if bits == 1:
        # 1-bit quantization keeps only sign; values restore from neg/pos means.
        codes = (features > 0).astype(np.uint8)
    else:
        # Min-max scale using clipped values and map to integer buckets.
        if clip_min is None or clip_max is None:
            raise ValueError(f"{bits}-bit quantization requires clip_min/clip_max")
        levels = (1 << bits) - 1
        clipped = np.clip(features, clip_min, clip_max)
        scaled = (clipped - clip_min) / (clip_max - clip_min)
        codes = np.rint(scaled * levels).astype(np.uint8)
    return _pack_codes(codes, bits)


def _pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack low-bit feature codes high-bits-first along the final dimension."""
    per_byte = 8 // bits
    pad = (-codes.shape[-1]) % per_byte
    if pad:
        # Pad only the feature dimension of this 2D [row, feature] array.
        codes = np.pad(codes, ((0, 0), (0, pad)), constant_values=0)
    # Group the padded feature dimension into chunks that each form one byte.
    # Valid bit widths pack exactly one byte per group, so the final sum is at
    # most 255. uint16 is a conservative arithmetic dtype that avoids relying on
    # NumPy's uint8 accumulator behavior before the final uint8 cast.
    codes = codes.reshape(codes.shape[0], -1, per_byte).astype(np.uint16)
    shifts = bits * np.arange(per_byte - 1, -1, -1, dtype=np.uint16)
    weights = (1 << shifts).astype(np.uint16)
    return np.sum(codes * weights, axis=-1).astype(np.uint8)
