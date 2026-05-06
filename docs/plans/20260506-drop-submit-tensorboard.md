# Drop `submit(tensorboard=...)`, single-uploader TB

Date: 2026-05-06
Predecessor PR: https://github.com/Snapchat/GiGL/pull/603

## Goal

Eliminate Vertex AI's auto-uploader. Keep only the chief-rank
`aiplatform.start_upload_tb_log` uploader for both live streaming and multi-run
comparison. Single uploader, single experiment, less plumbing.

## Why

PR #603 ships a dual-uploader design: Vertex's auto-uploader (gated on
`submit(tensorboard=...)`) plus a chief-rank `start_upload_tb_log` uploader.
That's because the SDK forces `submit(tensorboard=)` and `submit(experiment=)`
to be mutually exclusive, so getting both R1 (per-job UI link) and R2 (multi-run
comparison) required two parallel uploaders streaming from the same log dir.

We want to keep streaming and multi-run comparison, but we don't actually need
R1 (the "Open TensorBoard" button on the Vertex job page) — we can replace it
with a logged URL in trainer stdout. Dropping `submit(tensorboard=...)` removes
the dual-uploader oddity and most of the supporting plumbing in
`vertex_ai.py`.

## Step 0 — Constraint check (resolved via docs)

**Question:** does Vertex AI populate `AIP_TENSORBOARD_LOG_DIR` inside the
worker container when `baseOutputDirectory` is set on `CustomJobSpec` but
`submit(tensorboard=)` is NOT?

**Answer: yes.** Vertex's training-code-requirements doc
(https://cloud.google.com/vertex-ai/docs/training/code-requirements) is
unambiguous: when `baseOutputDirectory` is configured, Vertex AI sets
`AIP_MODEL_DIR`, `AIP_CHECKPOINT_DIR`, and `AIP_TENSORBOARD_LOG_DIR` env vars
unconditionally. The `tensorboard` field on `CustomJobSpec` is not a
prerequisite. **Step 4 below is not required** and is dropped from this plan.

(If smoke testing later reveals a discrepancy, Step 4 can be re-introduced as
a fallback.)

## Step 1 — Tighten validation: both fields or neither

**File:** `gigl/src/validation_check/libs/gbml_and_resource_config_compatibility_checks.py`

In `check_vertex_ai_trainer_tensorboard_compatibility`, replace the current
"experiment name requires resource name" rule with "both must be set together
(or both unset)":

- If exactly one of `tensorboard_resource_name` /
  `tensorboard_experiment_name` is set, raise.
- Add the Vertex resource-ID regex check on `tensorboard_experiment_name` here
  (moved from `_submit_job`).

This shifts the precondition out of submit-time into the validation-check
stage, where the rest of the resource-config rules live.

**Backwards compat:** zero risk. Both proto fields landed in PR #603 (this
branch); neither exists on `main`. No production config has
`tensorboard_resource_name` set without `tensorboard_experiment_name`.

**File:** `tests/unit/src/validation/lib/gbml_and_resource_config_compatibility_checks_test.py`
- Add a "resource_name set, experiment_name unset" failure test.
- Existing "experiment_name set, resource_name unset" failure test stays.
- Add a regex-failure test for an invalid experiment name.

## Step 2 — Drop the `submit(tensorboard=...)` path

**File:** `gigl/common/services/vertex_ai.py`

In `_submit_job` (around lines 411-440):
- Delete `tensorboard=job_config.tensorboard_resource_name or None` kwarg from
  `job.submit(...)`.
- Delete the experiment-name regex precondition block (lines 411-424); moved
  to validation in Step 1.

URL logging (lines 450-470):
- Delete the per-job URL log (lines 450-459).
- Keep the cross-job URL log (lines 460-470). Validation now guarantees both
  names are present whenever either is, so the inner `if` simplifies.

`VertexAiJobConfig` (around lines 213-214):
- Delete `tensorboard_resource_name` and `tensorboard_experiment_name` fields.
  They were carriers from launcher into `_submit_job`; nothing reads them now.

`_VERTEX_RESOURCE_ID_PATTERN` constant: delete from this file (only used by
validation now, which has its own copy or imports it).

## Step 3 — Stop wiring TB names into VertexAiJobConfig (launcher)

**File:** `gigl/src/common/vertex_ai_launcher.py`

- `_build_job_config` (around lines 405-412): drop `tensorboard_resource_name=...`
  and `tensorboard_experiment_name=...` kwargs to `VertexAiJobConfig`.
- Env-var injection block (lines 339-369): keep. The "both set" guard at line
  357-358 simplifies — since validation now enforces all-or-nothing, it's
  exactly one condition (either field set implies both).
- `baseOutputDirectory` plumbing: unchanged.

## Step 4 — Surface the named-experiment URL where users will see it

**File:** `gigl/utils/tensorboard_writer.py`

In `_maybe_start_uploader`, after `aiplatform.start_upload_tb_log(...)`
succeeds, log the cross-job experiment URL using the same format as
`vertex_ai.py:_build_tensorboard_experiment_url`. Either move that helper to a
shared location (`gigl/common/services/vertex_ai_url_helpers.py` or similar)
or inline the format string in the writer — it's three lines, duplication is
fine.

Compensates for losing the Vertex UI's "Open TensorBoard" button by putting
the link in trainer stdout, where engineers already look.

## Step 5 — Tests

**File:** `tests/unit/src/common/vertex_ai_launcher_test.py`
- Drop assertions on `cfg.tensorboard_resource_name` /
  `cfg.tensorboard_experiment_name` (the dataclass fields are gone). Env-var
  injection assertions stay and become the primary contract test.

**File:** `tests/unit/utils/tensorboard_writer_test.py`
- Add coverage for the URL log line emitted by `_maybe_start_uploader` on
  success (Step 4).

## Step 6 — Verification

- `make type_check` clean.
- Per-file: `make unit_test_py PY_TEST_FILES="vertex_ai_launcher_test.py"`,
  `tensorboard_writer_test.py`,
  `gbml_and_resource_config_compatibility_checks_test.py`.
- Smoke: rerun the same two-runs-on-one-experiment smoke from PR #603. Confirm:
  - Vertex job page no longer renders a TB button (expected regression).
  - Trainer stdout logs the named-experiment URL.
  - Both runs land on the same TB page side-by-side.
  - `printenv | grep AIP_` confirms `AIP_TENSORBOARD_LOG_DIR` is set even
    without `submit(tensorboard=)` (sanity check on the Step 0 doc claim).
- Full e2e CORA pipeline regression.

## Risk and rollback

- **Step 0's claim is load-bearing.** Resolved via docs, but the smoke run in
  Step 6 should cross-check `AIP_TENSORBOARD_LOG_DIR` actually appears in the
  worker container before relying on it in production.
- **UX regression on the Vertex UI button.** Mitigated by Step 4's stdout
  logging. Call out in the PR description so reviewers aren't surprised.
- **Rollback:** single PR, easy to revert. Proto is unchanged; both fields
  stay as carriers for the chief-rank uploader. Reverting just adds back the
  `submit(tensorboard=...)` kwarg and the dropped `VertexAiJobConfig` fields.

## Critical files

- `gigl/common/services/vertex_ai.py` — drop submit kwarg, drop dataclass
  fields, drop URL helpers (Step 2).
- `gigl/src/common/vertex_ai_launcher.py` — drop dataclass kwargs (Step 3).
- `gigl/utils/tensorboard_writer.py` — surface URL on uploader start (Step 4).
- `gigl/src/validation_check/libs/gbml_and_resource_config_compatibility_checks.py`
  — tighten to all-or-nothing + regex check (Step 1).
- Tests under `tests/unit/src/common/`, `tests/unit/utils/`,
  `tests/unit/src/validation/lib/`.

## Out of scope

- Structured "trainer output metadata" file for KFP UI surfacing of the TB
  URL. Considered useful but separate; defer.
- Removing `tensorboard_resource_name` field entirely. The chief-rank uploader
  needs it (it's how `start_upload_tb_log` knows which `Tensorboard` instance
  to write to), so the field stays.

## References

- Vertex AI training code requirements (env vars):
  https://cloud.google.com/vertex-ai/docs/training/code-requirements
- `CustomJobSpec` REST (`baseOutputDirectory`):
  https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec
- TB data model:
  https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview
- `aiplatform.start_upload_tb_log`:
  https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform#google_cloud_aiplatform_start_upload_tb_log
