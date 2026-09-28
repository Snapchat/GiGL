# Composing task and resource configs with Hydra

Task and resource configs can use [Hydra Defaults Lists](https://hydra.cc/docs/advanced/defaults_list/) by adding a
top-level `defaults` list. `ProtoUtils.read_proto_from_yaml` composes each YAML config when it reads it, then parses the
resolved mapping into the requested protobuf. Configs without a `defaults` list also pass through Hydra composition.
Composition preserves any active Hydra context, so reads work inside a user application under `@hydra.main`.

## Config root

For a local `*.yaml` primary, its parent directory is the Hydra config root. Hydra 1.3 resolves local config names with
its `.yaml` convention. Remote primaries, and local files not named `*.yaml`, are staged to a temporary `.yaml` file and
composed standalone.

```text
configs/
├── task.yaml
├── resource.yaml
├── shared/
│   └── directed.yaml
└── compute/
    └── local.yaml
```

GiGL does not search for a repository root or a directory named `configs`. Local configs retain their parent as the
config root so they can select sibling fragments. GCS and HTTP primaries are downloaded to a temporary local file before
composition; relative Defaults List entries are not downloaded with them. A remote primary can use an installed config
package as described below.

## Remote primary with local shared configs

GCS and HTTPS primaries can select shared configs installed in the runtime image by adding a Hydra search path:

```yaml
hydra:
  searchpath:
    - pkg://my_project.configs

defaults:
  - shared@_global_: common
  - _self_
```

Prefer `pkg://` roots so the same config works across machines and containers. The referenced package and its YAML files
must be installed in every environment that composes the primary. Hydra removes its own `hydra` metadata from the
composed mapping before GiGL parses the protobuf.

GiGL does not download Defaults List entries relative to a GCS prefix or HTTPS location. Without an explicit search
path, a remote primary is composed as a standalone file.

## Task config example

```yaml
# task.yaml
defaults:
  - shared@sharedConfig: directed
  - _self_
```

```yaml
# shared/directed.yaml
isGraphDirected: true
```

The package target after `@` places the selected group at the corresponding protobuf field. A fragment containing fields
for the whole protobuf can use `@_global_`.

Include `_self_` explicitly so it is clear whether values in the primary override group values or are overridden by
them.

## Resource config example

```yaml
# resource.yaml
defaults:
  - compute@shared_resource_config: local
  - _self_

shared_resource_config:
  common_compute_config:
    region: us-east1
```

```yaml
# compute/local.yaml
common_compute_config:
  project: example-project
  region: us-central1
  temp_regional_assets_bucket: gs://example-bucket
```

`compute@shared_resource_config: local` places all fields from `compute/local.yaml` under `shared_resource_config`. The
primary file can then set any fields in that section. Here, `_self_` is last, so the composed
`shared_resource_config.common_compute_config.region` is `us-east1`; `project` and `temp_regional_assets_bucket` still
come from the fragment. Put a setting in the fragment for its protobuf section, and use the primary file for any
pipeline-specific override.

Outside a KFP pipeline, resolvers such as `now`, `git_hash`, and `oc.env` are evaluated on each read. Environment
variables and installed packages referenced by a config must be available in the reading process. A dynamic value may
differ between reads.

## KFP pipeline snapshots

At submission, KFP composes the resource source to select its project, region, service account, and staging bucket.
ConfigValidator composes both source configs inside the pipeline, resolves any external shared resource config, and
writes plain protobuf YAML snapshots. It initializes its runtime from those snapshots and validates the same resolved
protobufs before publishing the snapshot URIs. Every downstream component receives those URIs, and the validator also
publishes its GLT backend decision from the resolved task config.

The validator accepts `source_task_config_uri` and `source_resource_config_uri`. It publishes:

- `composed_task_config_snapshot_uri`
- `composed_resource_config_snapshot_uri`
- `should_use_glt_backend`

Each snapshot starts with a provenance comment naming its source; a container-local source includes the docker image
name.

KFP cache hits reuse the prior validator outputs and do not compose again. If ConfigValidator reexecutes, it composes
the sources again. Its snapshot paths are based on `job_name`, so a reexecution with the same name can overwrite those
paths. Use a distinct job name when a run needs distinct snapshot paths.

The final composed mapping must still be a valid `GbmlConfig` or `GiglResourceConfig`. Protobuf parsing remains the
schema and type validation boundary.

## Boundaries

- GiGL does not consume Hydra command-line overrides, multirun, launchers, or output-directory behavior.
- GiGL does not automatically download config fragments next to a GCS or HTTPS primary.
- Treat config bundle write access as trusted access. GiGL configs can reference importable classes and commands.
