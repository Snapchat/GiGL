# Composing task and resource configs with Hydra

GiGL task and resource configs can opt into [Hydra Defaults Lists](https://hydra.cc/docs/advanced/defaults_list/) by
adding a top-level `defaults` list. Configs without `defaults` continue through the existing YAML and OmegaConf loader.

## Config root

The primary file's parent directory is its Hydra config root. Composed primary files and selected fragments must use
Hydra's `.yaml` extension. Plain legacy `.yml` files without `defaults` remain supported.

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
`configs`. Composition currently requires a local primary file. GCS and HTTP primaries remain supported only as plain,
single-file YAML without a Defaults List.

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

Resource config is read once before pipeline submission to select the project, region, service account, and staging
bucket. ConfigValidator resolves it again inside the pipeline and verifies that both resolutions have the same protobuf
fingerprint. Dynamic resolvers such as `now`, `git_hash`, and `oc.env` are therefore rejected in resource configs.
Deterministic references to another config value remain supported.

## Pipeline behavior

ConfigValidator composes both primary configs, writes fully resolved plain-protobuf YAML snapshots, initializes its
runtime from those exact snapshots, and validates them before publishing its outputs. Every downstream pipeline
component receives those snapshot URIs. ConfigValidator caching is disabled so changing a selected fragment cannot reuse
stale resolved outputs.

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
- GCS and HTTP configs remain supported only when they are plain, single-file YAML.
- Treat config bundle write access as trusted access. GiGL configs can reference importable classes and commands.
