import torch

from gigl.types.graph import FeatureQuantizationMetadata


def dequantize_torch_tensor(
    packed_features: torch.Tensor,
    metadata: FeatureQuantizationMetadata,
) -> torch.Tensor:
    """Reconstruct approximate float features from packed uint8 codes."""
    q = metadata

    VALID_BITS = (1, 2, 4, 8)
    if q.bits not in VALID_BITS:
        raise ValueError(f"bits must be one of {VALID_BITS}, got {q.bits}")
    per_byte = 8 // q.bits
    expected_packed_dim = (q.dequantized_feature_dim + per_byte - 1) // per_byte
    if packed_features.size(-1) != expected_packed_dim:
        raise ValueError(
            f"Expected packed feature dim {expected_packed_dim} for "
            f"{q.dequantized_feature_dim} {q.bits}-bit features, got "
            f"{packed_features.size(-1)}."
        )

    codes = _unpack_torch_tensor(
        packed_features, dim=q.dequantized_feature_dim, bits=q.bits
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
    shifts = bits * torch.arange(per_byte - 1, -1, -1, device=packed_features.device)
    codes = (packed_features.unsqueeze(-1).to(torch.int16) >> shifts).bitwise_and(mask)
    return codes.reshape(*packed_features.shape[:-1], -1)[..., :dim].to(torch.uint8)
