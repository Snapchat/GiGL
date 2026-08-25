# Composing task and resource configs with Hydra

Task and resource configs can use [Hydra Defaults Lists](https://hydra.cc/docs/advanced/defaults_list/) by adding a
top-level `defaults` list. ConfigValidator composes every source config before publishing plain YAML snapshots.
`ProtoUtils.read_proto_from_yaml` is the single URI-agnostic reader: every YAML config it reads is composed with Hydra.
Configs without a `defaults` list compose to themselves, so plain configs and materialized snapshots read unchanged.
Composition snapshots and restores any active foreign Hydra context, so reads also work inside a user application under
`@hydra.main`.

## Config root

The primary file's parent directory is its Hydra config root. Hydra 1.3 resolves local config names with its `.yaml`
convention. Remote primaries, and local files not named `*.yaml`, are staged to a temporary `.yaml` file and composed
standalone.

```text
configs/
├── task.yaml
├── resource.yaml
├── shared/
│   └── directed.yaml
└── compute/
    └── local.yaml
```

Keep primary files directly in the bundle root. GiGL does not search for a repository root or a directory named
`configs`. Local configs retain their parent as the config root so they can select sibling fragments. GCS and HTTP
primaries are downloaded to a temporary local file before composition; relative Defaults List entries are not downloaded
with them. Remote primaries can opt into an existing local config root as described below.

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
```

```yaml
# compute/local.yaml
common_compute_config:
  project: example-project
  region: us-central1
  temp_regional_assets_bucket: gs://example-bucket
```

KFP submission composes a local resource source once to select the project, region, service account, and staging bucket.
ConfigValidator composes it again inside the pipeline, and that validation-time result becomes the resource config used
by downstream components. Resolvers such as `now`, `git_hash`, and `oc.env` are supported; environment-based resolvers
must be available wherever the source config is composed.

## Pipeline behavior

ConfigValidator composes both primary configs, writes fully resolved plain-protobuf YAML snapshots, initializes its
runtime from those exact snapshots, and validates them before publishing its outputs. Source comments do not survive
composition; each snapshot starts with a `# Resolved Hydra config from: <source uri>` provenance comment instead, where
a container-local source is prefixed with its docker image. Every downstream pipeline component receives those snapshot
URIs. ConfigValidator follows the run's KFP caching settings: relaunching a pipeline with the same job name reuses the
previous resolved snapshots instead of recomposing. After editing config fragments, use a new job name (or disable
caching for the run) to recompose.

The final composed mapping must still be a valid `GbmlConfig` or `GiglResourceConfig`. Protobuf parsing remains the
schema and type validation boundary.

## Repository example

GiGL's three E2E resource configs share their infrastructure and preprocessing sections:

- [`e2e_cicd_resource_config.yaml`](../../../deployment/configs/e2e_cicd_resource_config.yaml)
- [`e2e_glt_resource_config.yaml`](../../../deployment/configs/e2e_glt_resource_config.yaml)
- [`e2e_glt_gs_resource_config.yaml`](../../../deployment/configs/e2e_glt_gs_resource_config.yaml)

Each primary selects [`e2e/shared.yaml`](../../../deployment/configs/e2e/shared.yaml) and
[`e2e/preprocessor.yaml`](../../../deployment/configs/e2e/preprocessor.yaml), then defines only its pipeline-specific
resources. Their names intentionally omit `resource_config` so repository validation does not mistake these partial
fragments for complete resource configs. The unit-test resource config remains standalone because its buckets, datasets,
and runner intentionally differ.

## Boundaries

- GiGL does not consume Hydra command-line overrides, multirun, launchers, or output-directory behavior.
- GiGL does not automatically download config fragments next to a GCS or HTTPS primary.
- Treat config bundle write access as trusted access. GiGL configs can reference importable classes and commands.
