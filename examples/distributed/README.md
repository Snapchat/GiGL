# Examples for Training and Inference using live subgraph sampling

## Homogeneous (CORA)

We use the CORA dataset as an example for sampling against a homogeneous dataset.

[homogeneous_inference.py](./homogeneous_inference.py) is an example inference loop for the CORA dataset, introduced in
`McCallum, A.K., Nigam, K., Rennie, J. et al. Automating the Construction of Internet Portals with Machine Learning. Information Retrieval 3, 127–163 (2000). https://doi.org/10.1023/A:1009953814988`
and available via the
[PyG `Planetoid` dataset](https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid).

You can follow along with [cora.ipynb](./cora.ipynb) to run an e2e GiGL pipeline on the CORA dataset. It will guide you
through running each component: `config_populator` -> `data_preprocessor` -> `trainer` -> `inferencer`

```{toctree}
:maxdepth: 2
:hidden:
cora.ipynb
```
