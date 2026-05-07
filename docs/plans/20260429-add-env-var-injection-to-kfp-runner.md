# Plan: Add env-var injection to the KFP runner

## 1. Background

Today, every container launched by the GiGL Kubeflow pipeline (config_validator, config_populator, data_preprocessor, subgraph_sampler, split_generator, trainer, inferencer, post_processor — see `SPECED_COMPONENTS` at `gigl/orchestration/kubeflow/kfp_pipeline.py:32-41`) inherits its environment from the image. There is no first-class way for a caller of `runner.py` to set arbitrary `ENV` values on those containers at compile time.

Wiring each downstream consumer in as a typed flag (e.g. one flag per image URI, one per Python-interop toggle, one per protobuf-implementation switch) does not scale and leaks consumer-specific concepts into the GiGL surface area. A single generic `--env KEY=VALUE` flag — threaded into `KfpOrchestrator.compile(...)` and applied per-task via the KFP v2 SDK's [`PipelineTask.set_env_variable(name, value)`](https://kubeflow-pipelines.readthedocs.io/en/stable/source/dsl.html#kfp.dsl.PipelineTask.set_env_variable) — is the cleanest carrier: GiGL knows nothing about the contents, callers own the meaning. GiGL's own code paths that need env config will continue to use the `GIGL_*` prefix; everything else is opaque transport.

## 2. API surface

**New CLI flag on `runner.py`** — repeatable, mirrors `--run_labels` exactly:

```
--env KEY=VALUE
```

Argparse spec — added next to `--run_labels` in `_get_parser` (`runner.py:319-328`):

- `action="append"`, `default=[]`, `help` describes the format and that values flow to every component at compile time.

**Parser helper** — added next to `_parse_labels` (`runner.py:212-225`); identical signature shape:

```
def _parse_env_vars(env_vars: list[str]) -> dict[str, str]
```

The template to mirror is the existing `_parse_labels` body verbatim — `split("=", 1)` on each entry, populate `dict[str, str]`, log the parsed result. Same error semantics: a malformed entry (no `=`) raises `ValueError` from `str.split` unpacking, exactly like `_parse_labels` does today.

**Plumbing path:**

1. `runner.py:__main__` calls `parsed_env_vars = _parse_env_vars(args.env)` alongside `parsed_additional_job_args` / `parsed_labels` (`runner.py:346-347`).
2. Both `KfpOrchestrator.compile` call sites (`runner.py:385` for the RUN action, `runner.py:412` for the COMPILE action) gain `env_vars=parsed_env_vars`.
3. `KfpOrchestrator.compile` (`kfp_orchestrator.py:51-109`) gains `env_vars: Optional[dict[str, str]] = None`, packs it into `CommonPipelineComponentConfigs` (a new field — see §3), and passes it through `generate_pipeline(...)`.
4. `kfp_pipeline.py:_generate_component_task` (`kfp_pipeline.py:54-128`) — after `add_task_resource_requirements(...)` is called and before `return component_task`, loop the dict and call `component_task.set_env_variable(name=k, value=v)` per entry.

Per the [KFP v2 API](https://kubeflow-pipelines.readthedocs.io/en/stable/source/dsl.html#kfp.dsl.PipelineTask.set_env_variable), `set_env_variable` takes one `name`/`value` pair per call and returns the task for chaining; we must loop, not pass a dict.

## 3. Where the env vars get applied

**All eight components in `SPECED_COMPONENTS`.** Justification:

- The whole point of a generic carrier is uniformity — callers expect that if they say `--env FOO=bar`, *every* container they're paying for sees `FOO=bar`.
- `_generate_component_task` is the single funnel through which every `PipelineTask` in `SPECED_COMPONENTS` is constructed (`kfp_pipeline.py:54-128`). Adding the loop there covers config_validator, config_populator, data_preprocessor, subgraph_sampler, split_generator, trainer, inferencer, post_processor in one place.
- The two non-`SPECED_COMPONENTS` tasks in the pipeline are `check_glt_backend_eligibility_component` (`utils/glt_backend.py`) and `log_metrics_to_ui` (`utils/log_metrics.py`). Decision: include them too, for a complete answer to "every container my pipeline launches sees these vars." Concretely, both are constructed inside `kfp_pipeline.py` (`_generate_component_tasks` for the GLT check, `_create_trainer_task_op` / `_create_post_processor_task_op` for log metrics). Apply the same loop right after each is built.
- Implementation choice that keeps the callsite count low: introduce a private helper `_apply_env_vars(task, env_vars)` inside `kfp_pipeline.py` (parallel in spirit to `add_task_resource_requirements`) and call it from `_generate_component_task` plus the two non-SPECED sites. One source of truth, no risk of drift.

The `env_vars` dict rides on `CommonPipelineComponentConfigs` (`gigl/common/types/resource_config.py:8-17`), added as `env_vars: dict[str, str] = field(default_factory=dict)` — same shape as the existing `additional_job_args` field.

## 4. Compile-time vs run-time

**Compile-time bake-in.** `set_env_variable` records the name/value into the compiled pipeline IR (the YAML at `dst_compiled_pipeline_path`); the values are baked at compile time, not resolved per run. That is the right choice here:

- For `--action=run`, `runner.py:384-395` *recompiles* on every invocation before calling `orchestrator.run`, so each run gets a fresh bake of whatever `--env` values were passed on that invocation. No UX loss.
- For `--action=compile`, `runner.py:411-422` produces a static artifact — the user explicitly wanted those values frozen in.
- For `--action=run_no_compile`, the user is opting into a pre-compiled pipeline; whatever envs were baked at compile time is what runs. This is consistent with how `additional_job_args` already behaves.

A run-time-resolved alternative would require declaring a KFP pipeline parameter (e.g. `env_vars: dict`) on the `@kfp.dsl.pipeline` function and passing it through every component op. KFP v2 has poor ergonomics for `dict` pipeline parameters and the immediate use case (callers injecting a config value per compile) does not benefit from late binding. We do *not* take that path; if a future caller needs per-run env, that's a separate proposal.

## 5. Validation / failure modes

Mirror `_parse_labels` exactly so behavior is predictable across all three flags:

- Malformed entry (no `=`): `str.split("=", 1)` returns one element, the unpack to `(name, value)` raises `ValueError`. Same as `_parse_labels` today — propagate, do not catch.
- Empty value (`--env FOO=`): valid, `value=""`. Mirrors `_parse_labels` — that flag also accepts empty values, so we stay consistent.
- Empty key (`--env =bar`): not validated by `_parse_labels` either; KFP's own `set_env_variable` will reject it downstream. Consistent failure surface; do not pre-empt.
- Duplicate keys across multiple `--env` invocations: last one wins (dict overwrite), same as `_parse_labels`.
- Reserved names: do *not* enforce a denylist inside GiGL. KFP itself reserves a small set (e.g. `KFP_*`); leaving validation to KFP keeps GiGL generic. Document the caveat in the help text.
- Cross-flag conflict: `_assert_required_flags` (`runner.py:134-172`) does not need a new check. `--env` is valid for all three actions (`run`, `run_no_compile`, `compile`), unlike `--run_labels` which is run-only. Document this difference.

## 6. Step-by-step implementation plan

Each bullet is one commit-sized unit:

1. **Add `env_vars` field to `CommonPipelineComponentConfigs`** at `gigl/common/types/resource_config.py:8-17`. Default-empty dict.
2. **Add `_apply_env_vars(task, env_vars)` helper in `kfp_pipeline.py`** — single-responsibility, loops the dict and calls `task.set_env_variable(name=k, value=v)`.
3. **Wire the helper into `_generate_component_task`** at `kfp_pipeline.py:54-128`, right after `add_task_resource_requirements(...)`. Also call it on `check_glt_backend_eligibility_component` and `log_metrics_to_ui` task results.
4. **Thread `env_vars: Optional[dict[str, str]] = None`** through `KfpOrchestrator.compile` (`kfp_orchestrator.py:51-109`) into `CommonPipelineComponentConfigs(...)` (existing site at `kfp_orchestrator.py:83-88`).
5. **Add `--env` arg to `_get_parser`** (`runner.py:_get_parser`) — `action="append"`, `default=[]`, help text describing the format and noting GiGL does not interpret values.
6. **Add `_parse_env_vars` helper** next to `_parse_labels` (`runner.py:212-225`).
7. **Plumb `parsed_env_vars` into both `KfpOrchestrator.compile(...)` call sites** in `runner.py` (`runner.py:385-395` and `runner.py:411-422`).
8. **Update the `runner.py` module docstring** (lines 1-71): document `--env` under both `RUN` and `COMPILE` action sections; note compile-time bake-in semantics.
9. **Unit test: parser** in `tests/unit/orchestration/kubeflow/runner_test.py` (new file, mirror layout of `kfp_orchestrator_test.py`). Cover: single var, multiple vars, value containing `=`, malformed entry raises `ValueError`, empty list returns empty dict.
10. **Unit test: compile-time injection** extending `kfp_orchestrator_test.py:KfpOrchestratorTest` — call `KfpOrchestrator.compile(..., env_vars={"FOO": "bar"})` writing to a tmp path, parse the resulting YAML, assert that every component spec under the compiled pipeline IR has an `env` entry with `name: FOO`, `value: bar`. The IR shape is stable for KFP v2 (`spec.executors.<id>.container.env`).

## 7. Test strategy

- **Parser unit test** (step 9 above): pure function, no I/O. Exhaustive on malformed inputs since this is user-facing CLI.
- **Compile integration test** (step 10): writes the compiled pipeline to a temp file, loads with `yaml.safe_load`, walks every executor in `pipelineSpec.deploymentSpec.executors`, asserts the env list contains all expected pairs. This proves the loop ran on every component, not just one. Run via `make unit_test_py PY_TEST_FILES="kfp_orchestrator_test.py"`.
- **Smoke test, manual**: from a downstream `Makefile`, add `--env=KEY1=value1 --env=KEY2=value2` to one `compile_*_kubeflow_pipeline` target, compile, and `grep -A2 "name: KEY1" build/gigl_pipeline_gnn.yaml` to confirm presence on every executor.

## 8. Rollout

GiGL-side callers of `gigl.orchestration.kubeflow.runner` to inventory:

- `Makefile:258 compile_gigl_kubeflow_pipeline`
- `Makefile:283 run_dev_gnn_kubeflow_pipeline`
- `Makefile:307 compile_simple_gigl_kubeflow_pipeline`
- `Makefile:328-ish run_dev_simple_kubeflow_pipeline` (the simple-GiGL run target — verify exact line)
- `Makefile:352, 360, 368, 376, 384, 392, 400, 407` — the e2e test targets that depend on `compile_gigl_kubeflow_pipeline`. They inherit whatever the compile target sets; no per-target change unless they need different envs.

GiGL repo: no callers beyond this `runner.py` module need changes. The flag is opt-in, default empty, fully backward-compatible.

Downstream repos that vendor GiGL as a submodule: a separate follow-up commit (out of scope of the GiGL PR) wires `--env=...` into whichever Makefile targets need it. That commit lives in the consumer repo, not GiGL.

## 9. Open questions

1. **Pipeline-parameter env**: do we expect any caller to want envs that vary *per run* of the same compiled pipeline? Not for the immediate use case driving this work; flag if that changes.
2. **VertexNotificationEmailOp env**: `_generate_component_tasks` wraps everything in an `ExitHandler(VertexNotificationEmailOp(...))` (`kfp_pipeline.py:252-256`). The notification op is a Google-managed component; should env vars be applied to it? Default answer: no — it's a managed op outside the user's control surface, applying envs is at best inert and at worst rejected. Confirm during implementation by checking whether `set_env_variable` on it raises.
3. **Naming — `--env` vs `--env_var` vs `--env_vars`**: `--env` is shortest and matches `docker run --env`; `--env_var` is more discoverable in `--help`; `--env_vars` would be most consistent with the existing plural `--run_labels`. Repeatable flags in this file use both styles (`--run_labels` plural, `--notification_emails` plural, `--additional_job_args` plural). Suggest `--env_vars` and document it as repeatable. Confirm with reviewer.
4. **Should the helper land on `CommonPipelineComponentConfigs` or be a separate parameter to `generate_pipeline`?** Going via `CommonPipelineComponentConfigs` is consistent with `additional_job_args` and minimizes signature churn. No open question; calling out the design choice.

### Critical Files for Implementation

- `gigl/orchestration/kubeflow/runner.py`
- `gigl/orchestration/kubeflow/kfp_orchestrator.py`
- `gigl/orchestration/kubeflow/kfp_pipeline.py`
- `gigl/common/types/resource_config.py`
- `tests/unit/orchestration/kubeflow/kfp_orchestrator_test.py`
