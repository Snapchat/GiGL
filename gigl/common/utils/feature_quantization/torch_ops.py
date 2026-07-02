import math

import torch

from gigl.types.graph import FeatureQuantizationMetadata


def dequantize_torch_tensor(
    packed_features: torch.Tensor,
    metadata: FeatureQuantizationMetadata,
) -> torch.Tensor:
    q = metadata

    VALID_BITS = (1, 2, 4, 8)
    if q.bits not in VALID_BITS:
        raise ValueError(f"bits must be one of {VALID_BITS}, got {q.bits}")

    codes = _unpack_torch_tensor(
        packed_features, dim=q.dequantized_feature_dim, bits=q.bits
    ).float()
    if q.bits == 1:
        if math.isnan(q.bucket_0_value) or math.isnan(q.bucket_1_value):
            raise ValueError("1-bit dequantization requires bucket values.")
        return torch.where(codes.bool(), q.bucket_1_value, q.bucket_0_value)
    else:
        if math.isnan(q.clip_min) or math.isnan(q.clip_max):
            raise ValueError(f"{q.bits}-bit dequantization requires clip values.")
        levels = (1 << q.bits) - 1
        return q.clip_min + (codes / levels) * (q.clip_max - q.clip_min)


def _unpack_torch_tensor(
    packed_features: torch.Tensor, *, dim: int, bits: int
) -> torch.Tensor:
    per_byte = 8 // bits
    packed_dim = (dim + per_byte - 1) // per_byte
    if packed_features.size(-1) != packed_dim:
        raise ValueError(
            f"Expected packed feature dim {packed_dim} for {dim} {bits}-bit "
            f"features, got {packed_features.size(-1)}."
        )
    mask = (1 << bits) - 1
    shifts = bits * torch.arange(per_byte - 1, -1, -1, device=packed_features.device)
    codes = (packed_features.unsqueeze(-1).to(torch.int16) >> shifts).bitwise_and(mask)
    return codes.reshape(*packed_features.shape[:-1], -1)[..., :dim].to(torch.uint8)
