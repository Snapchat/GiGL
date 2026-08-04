# Feature Quantization

This package contains the low-level NumPy and Torch helpers for feature
quantization in GiGL.

Feature quantization is lossy compression: high-precision feature values such as
fp32 are mapped into a lower-precision representation. The current built-in
scheme stores low-bit codes as packed `uint8` bytes, then reconstructs
approximate feature values when a sampled subgraph is materialized for training
or inference.

The motivation is practical scaling. Large-scale GNN training is often
memory-bound:

- feature hydration uses irregular memory access, often over the network;
- hydrated features still need to move to the accelerator;
- large feature stores limit the workloads that fit on a given machine.

Reducing feature size can improve feature-store footprint, network bandwidth,
and PCIe transfer volume. GNNs have also been shown to be relatively tolerant of
input feature quantization in [BiFeat: Supercharge GNN Training via Graph
Feature Quantization](https://arxiv.org/abs/2207.14696), which motivates this as
a useful tradeoff for GiGL.

## Current Built-In Flow

The built-in flow is:

1. The data preprocessor computes feature summary statistics offline.
2. The preprocessor quantizes selected scalar feature columns with NumPy.
3. The packed `uint8` feature sidecar is written to TFRecords.
4. Distributed dataset construction partitions and samples the packed bytes.
5. The dataloader collate path dequantizes sampled packed features with Torch.
6. Dequantized columns are scattered back into the logical `x` feature matrix.

The NumPy/Torch split is intentional:

- `numpy_ops.py` runs in preprocessing, where data is on CPU and Torch may not
  be available.
- `torch_ops.py` runs during dataloader collation, where sampled feature data is
  already represented as Torch tensors and may already be on GPU.

`FeatureQuantizationMetadata` is the contract between those two steps. It records
the bit width, packed feature dimension, logical feature positions, and the
statistics needed to invert the compression step.

## Current Built-In Scheme

The current implementation supports `1`, `2`, `4`, and `8` bit quantization.
Codes are packed high-bits-first into bytes.

For `1` bit, values are represented by sign and reconstructed from the positive
and non-positive means.

For `2`, `4`, and `8` bits, values are clipped to pre-computed bounds and mapped into
uniform integer buckets between those bounds.

## TODO: Pluggable Schemes

The current metadata/proto shape is tied to the built-in quantization scheme. A
useful follow-up is to make the quantization scheme itself pluggable.

One possible design is:

- define a quantizer object tied to `FeatureQuantizationSpec`;
- serialize a stable fully qualified name or registry key for the quantizer;
- serialize the quantizer arguments in the proto;
- rebuild the quantizer from that metadata on the read side;
- require each quantizer to provide matching NumPy `quantize` and Torch
  `dequantize` implementations.

That would let developers add new schemes without threading one-off fields
through every proto and loader path. A registry key is likely safer than
arbitrary imports, but either way the important contract is that the serialized
scheme identifies both the compression step and its inverse.
