# Examples for Training and Inference on Link Prediction GNN models.

## Homogeneous (CORA)

We use the CORA dataset as an example for sampling against a homogeneous dataset.

[homogeneous_inference.py](./homogeneous_inference.py) and [homogeneous_training.py](./homogeneous_training.py) are
example inference and training loops for the CORA dataset, the MNIST of graph models, and available via the PyG
`Planetoid`
[dataset](https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid).

You can follow along with [cora.ipynb](./cora.ipynb) to run an e2e GiGL pipeline on the CORA dataset. It will guide you
through running each component: `config_populator` -> `data_preprocessor` -> `trainer` -> `inferencer`

## Heterogeneous (DBLP)

We use use the DBLP dataset as an example for sampling against a heterogeneous dataset.

[heterogeneous_inference.py](./heterogeneous_inference.py) and [heterogeneous_training.py](./heterogeneous_training.py)
are example inference and training loops for the DBLP dataset. The DBLP dataset is avaialble at the `PyG` `DBLP`
[dataset](https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.datasets.DBLP.html#torch_geometric.datasets.DBLP).

You can follow along with [dblp.ipynb](./dblp.ipynb) to run an e2e GiGL pipeline on the DBLP dataset. It will guide you
through running each component: `config_populator` -> `data_preprocessor` -> `trainer` -> `inferencer`

## Vertex AI TensorBoard

The example trainer configs enable TensorBoard logging with
`trainerConfig.shouldLogToTensorboard: true`.

To surface those events in Vertex AI TensorBoard, set
`tensorboard_resource_name` on the trainer Vertex resource config, use a
regional bucket, and keep the bucket, CustomJob, and TensorBoard instance in
the same region. The attached service account should have
`roles/storage.admin` and `roles/aiplatform.user`.

```{toctree}
:maxdepth: 2
:hidden:
cora.ipynb
dblp.ipynb
```
