import torch


def dequantize_feature_tensor(
    packed_features: torch.Tensor,
    *,
    bits: int,
    dequantized_dim: int,
    clip_min: float | None = None, clip_max: float | None = None,
    bucket_0_value: float | None = None, bucket_1_value: float | None = None
) -> torch.Tensor:
    codes = _unpack_codes(packed_features, bits=bits, dim=dequantized_dim)
    if bits == 1:
        if bucket_0_value is None or bucket_1_value is None:
            raise ValueError(
                "Expected bucket_0_value and bucket_1_value for 1-bit centroid "
                "dequantization."
            )
        return torch.where(codes.bool(), bucket_1_value, bucket_0_value)

    if clip_min is None or clip_max is None:
        raise ValueError(
            f"Expected clip_min and clip_max for {bits}-bit linear dequantization."
        )
    levels = (1 << bits) - 1
    return clip_min + (codes / levels) * (clip_max - clip_min)


def _unpack_codes(
    packed_features: torch.Tensor, *, bits: int, dim: int
) -> torch.Tensor:
    if bits not in {1, 2, 4, 8}:
        raise ValueError(f"Expected bits to be one of 1, 2, 4, or 8, got {bits}.")

    per_byte = 8 // bits
    packed_dim = (dim + per_byte - 1) // per_byte
    if packed_features.size(1) != packed_dim:
        raise ValueError(
            f"Expected packed feature dim {packed_dim} for {dim} {bits}-bit "
            f"features, got {packed_features.size(1)}."
        )
    if bits == 8:
        return packed_features.to(torch.float32)

    idx = torch.arange(dim, device=packed_features.device)
    byte_idx = torch.div(idx, per_byte, rounding_mode="floor")
    offsets = ((idx % per_byte) * bits).to(torch.int16)
    bytes_ = packed_features[:, byte_idx].to(torch.int16)
    shifted = torch.bitwise_right_shift(bytes_, offsets)
    return torch.bitwise_and(shifted, (1 << bits) - 1).to(torch.float32)
