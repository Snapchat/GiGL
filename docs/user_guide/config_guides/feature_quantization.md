# Feature Quantization

Feature quantization is an opt-in preprocessing setting that stores selected scalar features in packed low-bit form.
GiGL automatically reconstructs approximate floating-point values before a sampled batch reaches the model.

Use it when features are a meaningful memory or data-transfer cost and a small accuracy tradeoff is acceptable. Start
with a baseline run, then evaluate quantization on your task metric.

For background on feature quantization for GNNs, see
[BiFeat: Supercharge GNN Training via Graph Feature Quantization](https://arxiv.org/abs/2207.14696).

## Enable it

The current API supports node features. Configure it in `get_nodes_preprocessing_spec`, the function in your data
preprocessor config that returns each `NodeDataPreprocessingSpec`. Do not add quantization logic to `preprocessing_fn`.

First, import `FeatureQuantizationSpec`:

```python
from gigl.src.data_preprocessor.lib.types import FeatureQuantizationSpec
```

Then add `feature_quantization_spec` to the existing `NodeDataPreprocessingSpec`. Its `feature_keys` must name fields
produced by `preprocessing_fn` and listed in `features_outputs`.

```python
def get_nodes_preprocessing_spec(self) -> dict[NodeDataReference, NodeDataPreprocessingSpec]:
    # Existing code defines node_data_ref, feature_spec_fn, preprocessing_fn, and node_output_id.
    return {
        node_data_ref: NodeDataPreprocessingSpec(
            feature_spec_fn=feature_spec_fn,
            preprocessing_fn=preprocessing_fn,
            identifier_output=node_output_id,
            features_outputs=["embedding_0", "embedding_1", "embedding_2"],
            feature_quantization_spec=FeatureQuantizationSpec(
                feature_keys=["embedding_0", "embedding_1", "embedding_2"],
                bits=4,
            ),
        ),
    }
```

The keys must be distinct scalar fields. GiGL supports `1`, `2`, `4`, and `8` bits.

After changing this setting, rerun preprocessing and use its output for subsequent training or inference. Do not reuse
artifacts produced with a different quantization setting. GiGL restores the original feature-vector order and dimension,
so model code does not need to change.

## Choose a bit width

Start with `bits=4`. Use `8` when preserving quality matters more than compression; try `2` or `1` only after validating
your task metric. Lower bit widths use less storage and can lose more information. Packing is most efficient when the
number of selected features fills whole bytes.

## Expected upside and tradeoffs

Packing selected features can reduce their stored and transferred payload. GiGL dequantizes sampled features at runtime,
so the model receives approximate `float` values rather than the original values.

Quantization is lossy. It can change model quality and may add preprocessing and runtime work. Its end-to-end effect on
memory, throughput, cost, and task quality depends on the graph, sampled workload, selected columns, and bit width.

TODO: Add published end-to-end benchmark results for memory, transfer, throughput, and task-quality impact.

## Gotchas and FAQ

### What may I quantize?

Choose only finite scalar fields from `features_outputs`. Every key must exist there and be unique. A scalar field has
one number for each entity, such as `age=42`. A vector-valued field has a list or array for each entity, such as
`embedding=[0.1, 0.2, 0.3]`; it cannot be quantized directly. Expose each vector element as a separate scalar output
field before selecting it.

### Do I need to change my model or loader?

No. GiGL saves quantization metadata during preprocessing and dequantizes sampled features before they are passed to the
model.

### Can I quantize only some features?

Yes. Put only the selected fields in `feature_keys`; unselected fields stay unquantized. Compare the task metric with
and without each selection, especially when feature scales or importance differ.

```python
features_outputs=["embedding_0", "embedding_1", "country_id"],
feature_quantization_spec=FeatureQuantizationSpec(
    feature_keys=["embedding_0", "embedding_1"],
    bits=4,
),
```

### Why is end-to-end training not faster?

Quantization reduces the selected feature payload, not all training work. If data loading, feature transfer, or memory
capacity is not the bottleneck, end-to-end training time may not improve. Compare loader time, accelerator utilization,
and memory usage with the unquantized baseline before increasing the quantization level. Inference may see more upside
when it is feature-transfer or memory-bound; measure it separately from training.

### Does this support edge features?

Not yet. The current API supports node features only. This guide will extend to the edge-feature API when it is
available.
