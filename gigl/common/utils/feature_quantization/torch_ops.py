import torch

from gigl.types.graph import FeatureQuantizationMetadata


def dequantize_feature_tensor(
    packed_features: torch.Tensor,
    *,
    metadata: FeatureQuantizationMetadata,
) -> torch.Tensor:
    VALID_BITS = (1, 2, 4, 8)
    if metadata.bits not in VALID_BITS:
        raise ValueError(
            f"Expected bits to be one of {VALID_BITS}, got {metadata.bits}."
        )

    codes = _unpack_features(
        packed_features,
        dim=metadata.dequantized_feature_dim,
        bits=metadata.bits,
    ).float()
    if metadata.bits == 1:
        if metadata.bucket_0_value is None or metadata.bucket_1_value is None:
            raise ValueError(
                "bucket_0_value and bucket_1_value required for 1-bit dequantization."
            )
        return torch.where(
            codes.bool(), metadata.bucket_1_value, metadata.bucket_0_value
        )
    else:
        if metadata.clip_min is None or metadata.clip_max is None:
            raise ValueError(
                f"clip_min and clip_max required for {metadata.bits}-bit dequantization."
            )
        levels = (1 << metadata.bits) - 1
        return metadata.clip_min + (codes / levels) * (
            metadata.clip_max - metadata.clip_min
        )


def _unpack_features(
    packed_features: torch.Tensor, *, dim: int, bits: int
) -> torch.Tensor:
    """Unpack codes written high-bits-first within each byte.

    Example: 2-bit codes [a, b, c, d] must be packed as byte: ``a << 6 | b << 4 | c << 2 | d``.
    """
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
