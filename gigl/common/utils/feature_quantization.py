import torch


def dequantize_feature_tensor(
    packed_features: torch.Tensor,
    *,
    bits: int,
    dequantized_dim: int,
    clip_min: float | None = None, clip_max: float | None = None,
    bucket_0_value: float | None = None, bucket_1_value: float | None = None
) -> torch.Tensor:
    VALID_BITS = (1, 2, 4, 8)
    if bits not in VALID_BITS:
        raise ValueError(f"Expected bits to be one of {VALID_BITS}, got {bits}.")
    if bits == 1:
        if bucket_0_value is None or bucket_1_value is None:
            raise ValueError(
                "Expected bucket_0_value and bucket_1_value for 1-bit dequantization."
            )
    elif clip_min is None or clip_max is None:
        raise ValueError(
            f"Expected clip_min and clip_max for {bits}-bit dequantization."
        )

    codes = _unpack_features(packed_features, dim=dequantized_dim, bits=bits).float()
    if bits == 1:
        return torch.where(codes.bool(), bucket_1_value, bucket_0_value)
    else:
        levels = (1 << bits) - 1
        return clip_min + (codes / levels) * (clip_max - clip_min)


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
