# Feature Quantization

Feature quantization is an opt-in preprocessing setting that stores selected features at reduced precision. Before the
model receives a sampled batch, GiGL reconstructs approximate `float32` versions of the original feature values in their
original order.

Use it when feature storage or transfer is a material constraint and a modest drop in the task metric is acceptable.
Establish an unquantized baseline first, then evaluate the same task with quantization enabled.

For background on feature quantization for GNNs, see
[BiFeat: Supercharge GNN Training via Graph Feature Quantization](https://arxiv.org/abs/2207.14696).

## Enable node-feature quantization

In your preprocessor setup, add `feature_quantization_spec` to the relevant `NodeDataPreprocessingSpec` returned from
`get_nodes_preprocessing_spec`:

```python
from gigl.src.data_preprocessor.lib.types import (
    FeatureQuantizationSpec,
    NodeDataPreprocessingSpec,
)


def get_nodes_preprocessing_spec(
    self,
) -> dict[NodeDataReference, NodeDataPreprocessingSpec]:
    # This example assumes these variables are already defined in this method:
    # node_data_ref, feature_spec_fn, preprocessing_fn, and node_output_id.
    feature_outputs = ["embedding_0", "embedding_1", "embedding_2"]

    return {
        node_data_ref: NodeDataPreprocessingSpec(
            feature_spec_fn=feature_spec_fn,
            preprocessing_fn=preprocessing_fn,
            identifier_output=node_output_id,
            features_outputs=feature_outputs,
            # Quantize all three scalar node features.
            feature_quantization_spec=FeatureQuantizationSpec(
                feature_keys=feature_outputs,
                bits=4,
            ),
        ),
    }
```

`feature_keys` may name all or only some fields in `features_outputs`. Quantized and unquantized fields can be mixed;
GiGL places both back into their original positions in the model input. After changing `feature_quantization_spec`,
rerun preprocessing.

## Enable edge-feature quantization

Add the same `feature_quantization_spec` to a main edge's `EdgeDataPreprocessingSpec` returned from
`get_edges_preprocessing_spec`:

```python
from gigl.src.data_preprocessor.lib.types import (
    EdgeDataPreprocessingSpec,
    FeatureQuantizationSpec,
)


def get_edges_preprocessing_spec(
    self,
) -> dict[EdgeDataReference, EdgeDataPreprocessingSpec]:
    edge_feature_outputs = ["match_score", "recency_days", "event_type"]

    return {
        main_edge_data_ref: EdgeDataPreprocessingSpec(
            feature_spec_fn=feature_spec_fn,
            preprocessing_fn=preprocessing_fn,
            identifier_output=edge_output_id,
            features_outputs=edge_feature_outputs,
            # Keep event_type raw while quantizing two scalar edge features.
            feature_quantization_spec=FeatureQuantizationSpec(
                feature_keys=["match_score", "recency_days"],
                bits=8,
            ),
        ),
    }
```

Edge quantization is supported only for references with `EdgeUsageType.MAIN`. Positive and negative supervision edges
cannot be quantized. A feature used as a sampling weight must also remain raw so the sampler can consume it directly.
GiGL reconstructs selected main-edge fields into their original positions in `edge_attr` before the model receives a
sampled batch.

## Configure each node and edge type independently

`feature_quantization_spec` belongs to one `NodeDataPreprocessingSpec` or `EdgeDataPreprocessingSpec`, not to the graph
as a whole. Each node or main-edge reference can therefore select its own `feature_keys` and `bits`. For example, one
node type can use 4-bit quantization, one main-edge type can use 8-bit quantization for a different feature subset, and
another type can remain unquantized by omitting `feature_quantization_spec`.

## Choose a bit width

GiGL supports `1`-, `2`-, `4`-, and `8`-bit compression per feature. Lower bit widths reduce the feature data more and
preserve less detail. Start with `4` bits. Use `8` bits when task quality is more important than data reduction;
consider `2` or `1` bits only after validating task quality.

## Expected effect and how to evaluate it

Quantization reduces only the stored and transferred data for the selected fields. Unselected features, graph topology,
labels, and model parameters are unchanged.

Quantization can increase sampling-worker throughput when fetching node or edge features, or when transferring hydrated
subgraphs to the GPU is the bottleneck. If training is instead limited by neighborhood sampling, model forward passes,
or backpropagation, do not expect a material throughput increase. Quantization can still reduce peak RAM for graph data,
in proportion to the share occupied by the selected features.

Quantization is lossy, so it can change task quality. Compare task quality with an unquantized baseline before adopting
a setting.

## FAQ

### What can I quantize?

Select distinct floating-point node or main-edge outputs from `preprocessing_fn` that are also listed in
`features_outputs`. Do not select vector-valued outputs. For example, `embedding=[0.1, 0.2, 0.3]` is one
three-dimensional output and cannot be quantized directly. Expose its elements as separate scalar outputs such as
`embedding_0`, `embedding_1`, and `embedding_2` before selecting them.

### Can I quantize only some features?

Yes. Leave fields unquantized when they are especially sensitive to approximation. This configuration quantizes two
embedding coordinates and keeps `country_id` unchanged:

```python
features_outputs = ["embedding_0", "embedding_1", "country_id"]

feature_quantization_spec = FeatureQuantizationSpec(
    feature_keys=["embedding_0", "embedding_1"],
    bits=4,
)
```

### Do I need to change my model or loader?

No model or loader configuration is required. GiGL writes the quantization metadata during preprocessing and
reconstructs approximate floating-point `x` and `edge_attr` tensors before they reach the model.

### Will quantization make training faster?

Not necessarily. Quantization reduces only selected feature data, while preprocessing and loading must also do extra
work. It is most useful when storing or transferring those features is a material part of the workload.

### Will quantization lower GPU requirements?

Not for model inputs. GiGL reconstructs node and edge feature tensors before they reach the model, so their per-batch
dimensions and floating-point memory requirements are unchanged. Quantization can still reduce feature storage,
transfer, and host RAM before that reconstruction.
