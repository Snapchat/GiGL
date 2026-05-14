# Examples for Supervised Node Classification on Homogeneous Graphs

## Homogeneous (CORA)

We use the CORA dataset as an example for supervised node classification on a homogeneous graph.

[homogeneous_training.py](./homogeneous_training.py) and [homogeneous_inference.py](./homogeneous_inference.py) are
example training and inference loops for the CORA dataset, the MNIST of graph models, and available via the PyG
`Planetoid`
[dataset](https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid).

A `cora.ipynb` walkthrough notebook is planned as a follow-up; for now, run the full e2e pipeline with:

```bash
make run_hom_cora_snc_e2e_test
```

The pipeline will run each component end-to-end: `config_populator` → `data_preprocessor` → `trainer` → `inferencer`,
exporting the per-anchor predicted class label (an integer in `[0, 7)` cast to `FLOAT64`) to a BigQuery table referenced
by `InferenceAssets.get_enumerated_predictions_table_path(...)`.
