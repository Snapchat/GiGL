### Updating Mocked Graph

1. Make necessary changes to ["graph_config.yaml"](./graph_config.yaml) is updated.

2. Potentially, update `MOCK_DATA_GCS_BUCKET` and `MOCK_DATA_BQ_DATASET_NAME` in `gigl/src/mocking/lib/constants.py` to
   upload to resources your custom buckets.

3. Run the following command to upload the relevant mocks to GCS and BQ:

```bash
python -m gigl.src.mocking.dataset_asset_mocking_suite \
--select mock_toy_graph_homogeneous_node_anchor_based_link_prediction_dataset \
--resource_config_uri=examples/toy_visual_example/resource_config.yaml
```

4. Subsequently, update the BQ paths in [task_config.yaml](./task_config.yaml).
