# Node Feature Quantization

Quantization stores selected scalar floating-point node features as packed low-bit values. GiGL reconstructs approximate
floating-point values before the model receives a batch.

## Enable quantization

Add `FeatureQuantizationSpec` to the `NodeDataPreprocessingSpec` for each node type you want to quantize, then rerun
data preprocessing.

```python
from gigl.src.data_preprocessor.lib.types import (
    FeatureQuantizationSpec,
    NodeDataPreprocessingSpec,
    NodeOutputIdentifier,
)

node_preprocessing_spec = NodeDataPreprocessingSpec(
    identifier_output=NodeOutputIdentifier("node_id"),
    features_outputs=["embedding_0", "embedding_1", "embedding_2"],
    feature_spec_fn=feature_spec_fn,
    preprocessing_fn=preprocessing_fn,
    feature_quantization_spec=FeatureQuantizationSpec(
        feature_keys=["embedding_0", "embedding_1", "embedding_2"],
        bits=4,
    ),
)
```

`feature_keys` must name distinct scalar output features from `features_outputs`. The supported bit widths are `1`, `2`,
`4`, and `8`.

No model, trainer, sampler, or inference changes are required. GiGL restores the original feature-vector order and
dimension before the batch reaches the model.

## Choose features and bit width

- Select only scalar floating-point outputs. IDs, labels, and non-scalar outputs cannot be quantized.
- Use `8` bits when input precision is more important than payload reduction. Use fewer bits only after evaluating the
  task metric with quantization enabled.
- At `2`, `4`, and `8` bits, GiGL clips every selected feature to one shared range and maps it to uniform levels.
- At `1` bit, GiGL stores only the sign and reconstructs values using the global mean of positive or non-positive
  values. This is the most lossy option.

## Expected benefit

The packed payload for selected features is smaller than 32-bit floats by this factor when the number of selected
features fills whole bytes:

| Bit width | Packed values per byte | Maximum reduction for selected features |
| --------- | ---------------------- | --------------------------------------- |
| 8         | 1                      | 4x                                      |
| 4         | 2                      | 8x                                      |
| 2         | 4                      | 16x                                     |
| 1         | 8                      | 32x                                     |

The final packed byte is padded when the selected feature count does not fill it, so the actual payload reduction can be
smaller. GiGL does not currently publish an end-to-end storage, throughput, or model-quality guarantee.

TODO: Add workload benchmarks for storage, transfer, and task-quality impact.
