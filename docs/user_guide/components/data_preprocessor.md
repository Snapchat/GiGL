# Data Preprocessor

The Data Preprocessor reads node, edge and respective feature data from a data source, and produces preprocessed /
transformed versions of all this data, for subsequent components to use. It uses Tensorflow Transform to achieve data
transformation in a distributed fashion, and allows for transformations like categorical encoding, scaling,
normalization, casting and more.

## Input

- **job_name** (AppliedTaskIdentifier): which uniquely identifies an end-to-end task.
- **task_config_uri** (Uri): Path which points to a "frozen" `GbmlConfig` proto yaml file - Can be either manually
  created, or `config_populator` component (recommended approach) can be used which can generate this frozen config from
  a template config.
- **resource_config_uri** (Uri): Path which points to a `GiGLResourceConfig` yaml
- **Optional: custom_worker_image_uri**: Path to docker file to be used for dataflow worker harness image

## What does it do?

The Data Preprocessor undertakes the following actions

- Reads frozen `GbmlConfig` proto yaml, which contains a pointer to a user-defined instance of the
  `DataPreprocessorConfig` class (see `dataPreprocessorConfigClsPath` field of `datasetConfig.dataPreprocessorConfig`).
  This class houses logic for

  - Preparing datasets for ingestion and transformation (see `prepare_for_pipeline`)
  - Defining transformation imperatives for different node types (`get_nodes_preprocessing_spec`)
  - Defining transformation imperatives for different edge types (`get_edges_preprocessing_spec`)

  Custom arguments can also be passed into the `DataPreprocessorConfig` class by including them in the
  `dataPreprocessorArgs` field inside `datasetConfig.dataPreprocessorConfig` section of `GbmlConfig`.

- Builds a `GraphMetadata` proto instance, which contains information about the node types (e.g. “user”) and edge types
  in the graph (e.g. “user-friends-user”), and assigns them corresponding “condensed” integer node and edge types.

- Runs an “enumeration” step to internally map all the node ids to integers to mitigate space overhead. Other components
  operate on these enumerated identifiers to reduce storage footprint, memory overhead, and network traffic.

- For each node and edge type, spins up a Dataflow job which manifests a Tensorflow Transform pipeline to operationalize
  the user-defined transformations specified in the `get_nodes_preprocessing_spec` and `get_edges_preprocessing_spec`
  functions inside the user-specified `DataPreprocessorConfig` instance. The pipelines write out transformed features as
  TFRecords, and a schema to help parse them, the inferred Tensorflow transform function for each feature-set, and other
  metadata to GCS.

## How do I run it?

**Import GiGL**

```python
from gigl.src.data_preprocessor.data_preprocessor import DataPreprocessor
from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier

data_preprocessor = DataPreprocessor()

data_preprocessor.run(
    applied_task_identifier=AppliedTaskIdentifier("sample_job_name"),
    task_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml"),
    resource_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/resource_config.yaml")
    custom_worker_image_uri="gcr.io/project/directory/dataflow_image:x.x.x",  # Optional
)
```

**Command Line**

```bash
python -m \
    gigl.src.data_preprocessor.data_preprocessor \
    --job_name="sample_job_name" \
    --task_config_uri="gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml" \
    --resource_config_uri="gs://MY TEMP ASSETS BUCKET/resource_config.yaml"
```

## Output

Upon completing the Dataflow jobs referenced in the last bullet point of [What](#what-does-it-do) above, the component
writes out a `PreprocessedMetadata` proto to URI specified by the `preprocessedMetadataUri` field in the `sharedConfig`
section of the frozen `GbmlConfig` i.e. the frozen task spec specified by `task_config_uri`.

This proto houses information about

- The inferred `GraphMetadata`
- A map of all condensed node types to `NodeMetadataOutput` protos
- A map of all condensed edge types to `EdgeMetadataOutput` protos

`NodeMetadataOutput` and `EdgeMetadataOutput` protos store information about the paths mentioned in the above bullet
point, and relevant metadata including the fields in each TFExample which store node/edge identifiers, feature keys,
labels, etc. `PreprocessedMetadata` will be read from this URI by other components.

## Custom Usage

- The actions this component undertakes are largely determined by the imperative transformation logic specified in the
  user-provided `DataPreprocessorConfig` class instance. This leaves much to user control. Please take a look at the
  instance provided at the `dataPreprocessorConfigClsPath` field of `datasetConfig`.`dataPreprocessorConfig` in order to
  learn more. For an example `dataPreprocessorConfig`, see
  [here](../../../../python/gigl/src/mocking/mocking_assets/passthrough_preprocessor_config_for_mocked_assets.py)

- In order to customize transformation logic for existing node features, take a look at preprocessing functions in
  [Tensorflow Transform ](https://www.tensorflow.org/tfx/transform/get_started) documentation. In order to add or remove
  node and edge features, you can modify the logic in `feature_spec_fn` and `preprocessing_fn` housed by
  `NodeDataPreprocessingSpec` and `EdgeDataPreprocessingSpec`. You can use the `build_ingestion_feature_spec_fn`
  function to conveniently generate feature specs which allow you to ingest and then transform these fields

- Note that the identifier fields (indicating node id, edge src node id, or edge dst node id) are always designated as
  integer types due to the enumeration steps which precedes the Tensorflow Transform jobs.

## Other

- **Design**: The design of this component is intended to leave maximal flexibility to the user in defining how they
  want to preprocess and transform their data. These steps are unlikely to be the same across many different custom
  pipelines (e.g. which fields to categorically encode, which to normalize, etc.) and thus we opted for a user-defined
  class to house as much code as possible which could be natively written by someone familiar with Tensorflow Transform.

- **Debugging**: The core logic of this component executes in Dataflow. A link to the Dataflow job will be printed in
  the logs of the component, which can be used to navigate to the Dataflow console and see fine-grained logging of the
  Dataflow pipeline.
