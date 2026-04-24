# Deprecated Tabularized Architecture

These pages describe the older GiGL pipeline that materializes sampled subgraphs ahead of time:

```text
Config Populator -> Data Preprocessor -> Subgraph Sampler -> Split Generator -> Trainer -> Inferencer
```

Use these docs only if you are maintaining an existing pipeline that still depends on the tabularized sampling path. New
in-memory subgraph sampling pipelines should start from the [current architecture overview](../architecture.md).

```{toctree}
:maxdepth: 1

subgraph_sampler
split_generator
```
