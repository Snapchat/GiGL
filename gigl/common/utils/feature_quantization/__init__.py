"""Utilities for lossy node feature quantization in GiGL.

Feature quantization compresses high-precision node feature columns into a
lower-precision representation. GiGL uses this to reduce feature-store size on
disk and in memory, and to reduce feature bandwidth across storage, sampling,
and device-transfer paths.

The current built-in workflow computes summary statistics offline in the
preprocessor, stores packed feature bytes, partitions and samples those bytes,
then dequantizes sampled subgraph features during dataloader collation.
"""
