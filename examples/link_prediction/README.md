# Examples for Training and Inference on Link Prediction GNN models.

## Homogeneous (CORA)

We use the CORA dataset as an example for sampling against a homogeneous dataset.

[homogeneous_inference.py](./homogeneous_inference.py) is an example inference loop for the CORA dataset, the MNIST of graph models, and available via the
PyG `Planetoid` [dataset](https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid).

You can follow along with [cora.ipynb](./cora.ipynb) to run an e2e GiGL pipeline on the CORA dataset. It will guide you
through running each component: `config_populator` -> `data_preprocessor` -> `trainer` -> `inferencer`

```{toctree}
:maxdepth: 2
:hidden:
cora.ipynb
```
