"""Tensor utilities for feature quantization."""

import torch

from gigl.types.graph import FeatureQuantizationMetadata


def _unpack_quantized_features(
    packed_features: torch.Tensor,
    quantization_metadata: FeatureQuantizationMetadata,
) -> torch.Tensor:
    bits = quantization_metadata.bits
    dequantized_feature_dim = quantization_metadata.dequantized_feature_dim
    if bits not in {1, 2, 4, 8}:
        raise ValueError(f"Expected bits to be one of 1, 2, 4, or 8, got {bits}.")

    if bits == 8:
        if packed_features.size(1) < dequantized_feature_dim:
            raise ValueError(
                f"Packed feature dim {packed_features.size(1)} is smaller than "
                f"dequantized dim {dequantized_feature_dim}."
            )
        return packed_features[:, :dequantized_feature_dim].to(torch.float32)

    values_per_byte = 8 // bits
    feature_indices = torch.arange(
        dequantized_feature_dim, device=packed_features.device
    )
    byte_indices = torch.div(feature_indices, values_per_byte, rounding_mode="floor")
    bit_offsets = ((feature_indices % values_per_byte) * bits).to(torch.int16)
    selected_bytes = packed_features[:, byte_indices].to(torch.int16)
    return torch.bitwise_and(
        torch.bitwise_right_shift(selected_bytes, bit_offsets),
        (1 << bits) - 1,
    ).to(torch.float32)


def dequantize_feature_tensor(
    packed_features: torch.Tensor,
    quantization_metadata: FeatureQuantizationMetadata,
) -> torch.Tensor:
    codes = _unpack_quantized_features(packed_features, quantization_metadata)
    if quantization_metadata.bits == 1:
        if (
            quantization_metadata.bucket_0_value is None
            or quantization_metadata.bucket_1_value is None
        ):
            raise ValueError("Centroid quantization requires both bucket values.")
        return torch.where(
            codes.bool(),
            torch.tensor(
                quantization_metadata.bucket_1_value,
                dtype=torch.float32,
                device=packed_features.device,
            ),
            torch.tensor(
                quantization_metadata.bucket_0_value,
                dtype=torch.float32,
                device=packed_features.device,
            ),
        )

    if quantization_metadata.clip_min is None or quantization_metadata.clip_max is None:
        raise ValueError("Linear quantization requires both clip bounds.")
    levels = (1 << quantization_metadata.bits) - 1
    clip_min = torch.tensor(
        quantization_metadata.clip_min,
        dtype=torch.float32,
        device=packed_features.device,
    )
    clip_max = torch.tensor(
        quantization_metadata.clip_max,
        dtype=torch.float32,
        device=packed_features.device,
    )
    return clip_min + (codes / levels) * (clip_max - clip_min)
