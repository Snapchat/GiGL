"""Torch feature dequantization helpers for dataloader collation.

Quantization lives in numpy_ops.py because preprocessing works with CPU arrays
in an environment without torch. Dequantization runs in the dataloader collate
path, where packed feature data is already represented as torch tensors.
"""

import torch
from jaxtyping import Float32, UInt8

from gigl.types.graph import FeatureQuantizationMetadata


def dequantize_torch_tensor(
    packed_features: UInt8[torch.Tensor, "... packed_feature_dim"],
    metadata: FeatureQuantizationMetadata,
) -> Float32[torch.Tensor, "... {len(metadata.packed_feature_indices)}"]:
    """Reconstruct float features from packed uint8 rows.

    The columns come back in ``metadata.packed_feature_indices`` order: first the n-bit quantized columns
    (approximate), then the lossless columns (exact, except ``bfloat16`` ones).
    """
    q = metadata

    if packed_features.size(-1) != q.packed_feature_dim:
        raise ValueError(
            f"Expected packed feature dim {q.packed_feature_dim} for "
            f"{q.quantized_feature_dim} quantized and {q.lossless_feature_dim} lossless features, got "
            f"{packed_features.size(-1)}."
        )

    regions: list[torch.Tensor] = []
    if q.quantized_feature_dim:
        regions.append(
            _dequantize_region(packed_features[..., : q.quantized_packed_dim], q)
        )
    lossless_codec = q.lossless_codec
    if lossless_codec is not None:
        lossless_rows = packed_features[..., q.lossless_byte_offset :]
        leading_shape = lossless_rows.shape[:-1]
        regions.append(
            lossless_codec.decode(
                lossless_rows.reshape(-1, lossless_rows.size(-1))
            ).reshape(*leading_shape, -1)
        )
    if not regions:
        return packed_features.new_zeros(
            (*packed_features.shape[:-1], 0), dtype=torch.float32
        )
    if len(regions) == 1:
        return regions[0]
    return torch.cat(regions, dim=-1)


def _dequantize_region(
    packed_features: torch.Tensor, q: FeatureQuantizationMetadata
) -> torch.Tensor:
    assert q.bits is not None
    codes = _unpack_torch_tensor(
        packed_features, dim=q.quantized_feature_dim, bits=q.bits
    ).float()
    if q.bits == 1:
        if q.neg_mean is None or q.pos_mean is None:
            raise ValueError("1-bit dequantization requires pos_mean/neg_mean")
        return torch.where(codes.bool(), q.pos_mean, q.neg_mean)
    else:
        if q.clip_min is None or q.clip_max is None:
            raise ValueError(f"{q.bits}-bit dequantization requires clip_min/clip_max")
        levels = (1 << q.bits) - 1
        return q.clip_min + (codes / levels) * (q.clip_max - q.clip_min)


def _unpack_torch_tensor(
    packed_features: torch.Tensor, *, dim: int, bits: int
) -> torch.Tensor:
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    # Extract high-bits-first codes from each packed byte.
    shifts = bits * torch.arange(
        per_byte - 1, -1, -1, device=packed_features.device, dtype=torch.uint8
    )
    codes = (packed_features.unsqueeze(-1) >> shifts).bitwise_and(mask)
    return codes.reshape(*packed_features.shape[:-1], -1)[..., :dim]
