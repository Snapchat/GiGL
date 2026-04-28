# Post Processor

The Post Processor is the final stage in the GiGL pipeline. It runs after inference, performs shared cleanup, and can
optionally execute user-defined post-processing logic.

## Input

- **job_name** (AppliedTaskIdentifier): which uniquely identifies an end-to-end task.
- **task_config_uri** (Uri): Path which points to a frozen `GbmlConfig` proto yaml file.
- **resource_config_uri** (Uri): Path which points to a `GiGLResourceConfig` yaml.

## What does it do?

The Post Processor undertakes the following actions:

- Reads the frozen `GbmlConfig`.
- When the in-memory subgraph sampling path is enabled (`featureFlags.should_run_glt_backend`), automatically
  unenumerates inference outputs by joining the enumerated embedding or prediction tables with the enumeration mapping
  tables written by the Data Preprocessor.
- Instantiates the user-defined post processor class from `postProcessorConfig.postProcessorClsPath`, if one is
  configured.
- Writes any returned evaluation metrics to the configured post-processor metrics path.
- Cleans up temporary assets unless `should_skip_automatic_temp_asset_cleanup` is set.

## Why unenumeration lives here

The Data Preprocessor enumerates node IDs into compact integers so later stages can use more efficient internal IDs. For
in-memory inference, embeddings and predictions are written for those enumerated IDs first. The Post Processor then
restores the original user-facing IDs before downstream systems consume the outputs.

This makes the Post Processor part of the core in-memory subgraph sampling architecture rather than an optional
afterthought.

## How do I run it?

**Import GiGL**

```python
from gigl.src.post_process.post_processor import PostProcessor
from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier

post_processor = PostProcessor()

post_processor.run(
    applied_task_identifier=AppliedTaskIdentifier("sample_job_name"),
    task_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml"),
    resource_config_uri=UriFactory.create_uri("gs://MY TEMP ASSETS BUCKET/resource_config.yaml"),
)
```

**Command Line**

```bash
python -m \
    gigl.src.post_process.post_processor \
    --job_name="sample_job_name" \
    --task_config_uri="gs://MY TEMP ASSETS BUCKET/frozen_task_config.yaml" \
    --resource_config_uri="gs://MY TEMP ASSETS BUCKET/resource_config.yaml"
```

## Output

The Post Processor may produce:

- unenumerated embedding tables,
- unenumerated prediction tables,
- user-defined post-processed assets,
- post-processor evaluation metrics.

## Custom Usage

If `postProcessorConfig.postProcessorClsPath` is set, GiGL instantiates that class and runs it after shared
unenumeration logic completes. Custom post processors should inherit from
{py:class}`gigl.src.post_process.lib.base_post_processor.BasePostProcessor`.

## Other

- The shared unenumeration logic is currently tied to the in-memory subgraph sampling path.
- If no user-defined post processor is configured, GiGL still runs the shared cleanup and automatic unenumeration steps
  for in-memory sampling outputs.
