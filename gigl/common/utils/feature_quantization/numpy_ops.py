from collections.abc import Mapping

import numpy as np


def quantize_ndarray(
    features: np.ndarray, *, bits: int, stats: Mapping[str, float]
) -> np.ndarray:
    """Quantize a 2D float array into packed uint8 codes."""
    if bits not in (1, 2, 4, 8):
        raise ValueError(f"bits must be one of 1, 2, 4, or 8, got {bits}.")
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, got shape {features.shape}.")
    if bits == 1:
        # 1-bit quantization keeps only sign; values restore from neg/pos means.
        codes = (features > 0).astype(np.uint8)
    else:
        # Linearly map clipped values into integer buckets.
        levels = (1 << bits) - 1
        lo, hi = stats["clip_min"], stats["clip_max"]
        clipped = np.clip(features, lo, hi)
        scaled = (clipped - lo) / (hi - lo)
        codes = np.rint(scaled * levels).astype(np.uint8)
    return pack_codes(codes, bits)


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack low-bit feature codes high-bits-first along the final dimension."""
    if bits not in (1, 2, 4, 8):
        raise ValueError(f"bits must be one of 1, 2, 4, or 8, got {bits}.")
    if codes.ndim != 2:
        raise ValueError(f"Expected a 2D code array, got shape {codes.shape}.")
    per_byte = 8 // bits
    pad = (-codes.shape[-1]) % per_byte
    if pad:
        # Pad only the feature dimension of this 2D [row, feature] array.
        codes = np.pad(codes, ((0, 0), (0, pad)), constant_values=0)
    # Group the padded feature dimension into chunks that each form one byte.
    codes = codes.reshape(codes.shape[0], -1, per_byte).astype(np.uint16)
    shifts = bits * np.arange(per_byte - 1, -1, -1, dtype=np.uint16)
    weights = (1 << shifts).astype(np.uint16)
    return np.sum(codes * weights, axis=-1).astype(np.uint8)
