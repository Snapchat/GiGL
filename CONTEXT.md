# Graph Feature Quantization

This context defines the storage optimization used for graph features while preserving the feature tensors consumed by
models.

## Language

**Feature quantization**: An opt-in representation that stores selected logical feature fields as packed low-bit values
and reconstructs floating-point tensors before model consumption.

**Logical feature**: A model-facing feature in its original position and floating-point representation, independent of
its stored representation.

**Packed feature**: The serialized and distributed-storage representation of one or more quantized logical features.

**Raw feature sidecar**: The unquantized floating-point columns retained beside a packed feature for the same entities.

**Quantization metadata**: The bit width, logical feature positions, and dequantization state required to reconstruct a
logical feature vector.

**Main edge**: An edge in the graph used by neighbor sampling and message passing. _Avoid_: Training edge, regular edge

**Supervision edge**: A positive or negative source-destination pair used as a link-prediction label rather than as part
of the sampled message-passing feature store. _Avoid_: Main edge, message-passing edge

**Sampling weight**: A raw scalar edge value consumed before neighbor selection, distinct from model-facing edge
features materialized after sampling.

**Transparent materialization**: Reconstruction of packed features into their logical floating-point tensor and feature
order before a sampled batch reaches the model. _Avoid_: Model-side dequantization
