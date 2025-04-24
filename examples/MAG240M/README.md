## Running the MAG240M experiments on your own GCP project

The following instructions assume you do not have a KFP cluster setup

### 1. (Optional) Pull MAG240M data into your own project

GiGL assumes your data is available in BQ Tables. BQ is ubiqutous to large enterprises as it provides a serverless,
highly scalable, and cost-effective platform for storing and analyzing massive datasets.

We provide a script `fetch_data.ipynb` which you can utilize to load the MAG240M data into BQ tables in your own
project. Alternatively, you can skip this all together since we a copy of this dataset in BQ that can be utilized right
away.

### 2. Run e2e pipeline

Prerequiste: Ensure you have access to your own GCP project, and a service account setup. You should also have gcloud
cli setup locally and/or running the notebook through a GCP VM. Some basic knowledge of GCP may be necessary here.

Note: If you decided to follow step 1. you may need to subesequently modify paths in
`examples/MAG240M/preprocessor_config.py`

Follow along `examples/MAG240M/mag240m.ipynb` to run an e2e GiGL pipeline on the MAG240M dataset. It will guide you
through running each component: `config_populator` -> `data_preprocessor` -> `subgraph_sampler` -> `split_generator` ->
`trainer` -> `inferencer`
