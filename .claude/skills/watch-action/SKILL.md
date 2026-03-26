______________________________________________________________________

## description: Monitor a GitHub Actions workflow run end-to-end through GH Actions, CloudBuild, and Vertex AI Pipelines. Invoke with a run URL or run ID. argument-hint: "<run-url-or-id> [--repo owner/repo]"

# Watch Action

Monitor a GitHub Actions CI/CD run through every layer of the stack: GitHub Actions jobs, Google Cloud Build, and Vertex
AI Pipelines. Reports status, investigates failures, and proposes fixes.

## Instructions

When this skill is invoked with `$ARGUMENTS`, execute the following sections in order.

______________________________________________________________________

### 1. Parse arguments

Extract the run ID and repository from `$ARGUMENTS`:

| Input                                                              | Run ID         | Repo                                             |
| ------------------------------------------------------------------ | -------------- | ------------------------------------------------ |
| `https://github.com/{owner}/{repo}/actions/runs/{id}`              | `{id}`         | `{owner}/{repo}`                                 |
| `https://github.com/{owner}/{repo}/actions/runs/{id}/attempts/{n}` | `{id}`         | `{owner}/{repo}`                                 |
| `{numeric-id}`                                                     | `{numeric-id}` | From `--repo` flag, or current `gh` repo context |
| `{numeric-id} --repo owner/repo`                                   | `{numeric-id}` | `owner/repo`                                     |

Validate the run exists:

```bash
gh run view {RUN_ID} --repo {REPO} --json status,conclusion
```

If this fails, tell the user the run was not found and stop.

______________________________________________________________________

### 2. Initial assessment

Fetch all jobs and build a status table:

```bash
gh run view {RUN_ID} --repo {REPO} --json status,conclusion,jobs,createdAt \
  --jq '{
    status: .status,
    conclusion: .conclusion,
    created: .createdAt,
    jobs: [.jobs[] | {name, status, conclusion, databaseId}]
  }'
```

Present a markdown status table to the user:

```
| Job | Status | Conclusion |
|-----|--------|------------|
| lint-test | completed | success |
| unit-test-python | in_progress | — |
| ... | ... | ... |
```

Categorize each job into a **duration tier** for polling:

- **Fast** (~10-15 min): jobs with names containing `lint`, `scala`
- **Medium** (~20-40 min): jobs with names containing `unit`, `integration` (but not `e2e`)
- **Slow** (~60-155 min): jobs with names containing `e2e`
- **Unknown**: everything else, treat as Medium

If the run is already completed, skip to **Section 7 (Final report)**.

If any jobs have already failed, immediately investigate them per **Section 5**.

______________________________________________________________________

### 3. Discover CloudBuild jobs

GH Actions job logs are often unavailable via API until jobs complete (returns 404). Instead, discover CloudBuild jobs
directly.

Extract the run's creation timestamp from the initial assessment, then query:

```bash
gcloud builds list \
  --project=external-snap-ci-github-gigl \
  --limit=20 \
  --filter="createTime>='{run_created_at}' AND status!='CANCELLED'" \
  --format="table(id, status, createTime, substitutions._CMD)"
```

Map CloudBuild jobs to GH Actions jobs using the `_CMD` substitution:

| `_CMD` contains                    | GH Actions job       |
| ---------------------------------- | -------------------- |
| `unit_test_py`                     | unit-test-python     |
| `unit_test_scala`                  | unit-test-scala      |
| `integration_test` (without `e2e`) | integration-test     |
| `run_all_e2e_tests`                | integration-e2e-test |
| `lint_test`                        | lint-test            |
| `run_all_notebook_e2e_tests`       | notebooks-e2e-test   |

Store this mapping — you will use CloudBuild IDs to fetch logs when GH Actions logs are unavailable.

**Note:** The CloudBuild project is `external-snap-ci-github-gigl`. If your `gcloud` auth doesn't have access, fall back
to GH Actions logs only and note this limitation to the user.

______________________________________________________________________

### 4. Monitor loop

Poll until all active jobs reach `completed` status.

**Polling strategy:**

- While any **Fast**-tier job is active: poll every **3 minutes**
- While only **Medium**-tier or slower jobs remain: poll every **5 minutes**
- While only **Slow**-tier jobs remain: poll every **5 minutes**, but also check Vertex AI pipelines (Section 6)

**On each poll:**

1. Fetch job statuses:
   ```bash
   gh run view {RUN_ID} --repo {REPO} --json jobs \
     --jq '[.jobs[] | {name, status, conclusion}]'
   ```
2. Compare against previous poll to detect changes
3. For **newly completed** jobs: report pass/fail
4. For **newly failed** jobs: immediately investigate (Section 5)
5. For **e2e jobs still in progress**: check Vertex AI pipelines (Section 6)
6. Present updated status table

**Long-running jobs:** If only Slow-tier jobs remain and estimated wait is >30 minutes, suggest to the user:

```
The remaining e2e jobs may take another 30-90 minutes.
You can use `/loop` to monitor in the background:
/loop "Check GH Actions run {RUN_ID} status and report any changes" --every 5m
```

______________________________________________________________________

### 5. Investigate failures

When a job fails, drill down through the stack layers:

**Layer 1 — GitHub Actions logs:**

```bash
gh api repos/{OWNER}/{REPO}/actions/jobs/{DATABASE_ID}/logs 2>&1 | tail -200
```

If this returns 404 (`BlobNotFound`), proceed to Layer 2.

**Layer 2 — CloudBuild logs:**

Using the CloudBuild ID mapped in Section 3:

```bash
gcloud builds log {BUILD_ID} --project=external-snap-ci-github-gigl 2>&1 | tail -300
```

Look for error messages, stack traces, failing test names.

**Layer 3 — Vertex AI logs (for e2e failures only):**

If the e2e CloudBuild job failed, check which Vertex AI pipeline failed:

```bash
gcloud ai pipelines runs list \
  --project=external-snap-ci-github-gigl \
  --region=us-central1 \
  --filter="createTime>='{run_created_at}'" \
  --format="table(name, displayName, state, createTime)"
```

For failed pipelines, check the custom jobs within them:

```bash
gcloud ai custom-jobs list \
  --project=external-snap-ci-github-gigl \
  --region=us-central1 \
  --filter="createTime>='{run_created_at}'" \
  --format="table(name, displayName, state, createTime, error.message)"
```

**After identifying the error:**

1. Read the relevant source files referenced in the error
2. Present findings with file paths and line numbers
3. Propose a concrete fix with rationale
4. Suggest local validation commands:
   - For Python test failures: `make unit_test_py PY_TEST_FILES="failing_test.py"`
   - For type errors: `make type_check`
   - For lint failures: `make check_format`

______________________________________________________________________

### 6. Monitor Vertex AI pipelines (e2e jobs only)

When the integration-e2e-test job is in progress and its CloudBuild ID is known:

**Discover pipeline runs:**

```bash
gcloud ai pipelines runs list \
  --project=external-snap-ci-github-gigl \
  --region=us-central1 \
  --filter="createTime>='{run_created_at}'" \
  --format="table(name, displayName, state, createTime)"
```

**Expected pipelines** (from `tests/e2e_tests/e2e_tests.yaml`):

- `cora-nalp-test`, `cora-snc-test`, `cora-udl-test`, `dblp-nalp-test`
- `hom-cora-sup-test`, `het-dblp-sup-test`
- `hom-cora-sup-gs-test`, `het-dblp-sup-gs-test`

Note: Pipeline job names use hyphens, not underscores (converted via `str(applied_task_identifier).replace("_", "-")` in
`gigl/orchestration/kubeflow/kfp_orchestrator.py:179`).

**Track pipeline states:**

| State                      | Meaning             |
| -------------------------- | ------------------- |
| `PIPELINE_STATE_RUNNING`   | Still executing     |
| `PIPELINE_STATE_SUCCEEDED` | Passed              |
| `PIPELINE_STATE_FAILED`    | Needs investigation |
| `PIPELINE_STATE_CANCELLED` | Was cancelled       |

Present a pipeline status table alongside the main job status table when Vertex AI pipelines are active.

**On pipeline failure:** Immediately investigate per Section 5, Layer 3.

______________________________________________________________________

### 7. Final report

When all jobs have completed, present a summary:

```
## CI/CD Run Summary — {RUN_ID}

| Job | Result |
|-----|--------|
| lint-test | passed |
| unit-test-python | passed |
| ... | ... |

### Failures
(For each failed job: root cause summary, relevant file paths, suggested fix)

### Next Steps
- (If all passed): "All checks passed. Ready to merge."
- (If failures): "Fix the issues above, push, and re-trigger with `/all_test`."
- (If flaky): "Consider re-running: `gh run rerun {RUN_ID} --repo {REPO} --failed`"
```

______________________________________________________________________

### Permissions note

This skill requires the following commands to be allowlisted in `.claude/settings.local.json` under
`"permissions.allow"`:

```json
"Bash(gh run view:*)",
"Bash(gh run watch:*)",
"Bash(gh api:*)",
"Bash(gcloud builds list:*)",
"Bash(gcloud builds log:*)",
"Bash(gcloud ai pipelines:*)",
"Bash(gcloud ai custom-jobs:*)"
```

If a command is blocked, note the limitation and continue with available data sources.

______________________________________________________________________

### Key codebase references

These files are useful context when investigating failures on this project:

- `.github/workflows/on-pr-comment.yml` — Workflow definition, job names and commands
- `.github/actions/run-cloud-run-command-on-active-checkout/action.yml` — CloudBuild submission action
- `.github/cloud_builder/run_command_on_active_checkout.yaml` — CloudBuild config (`logging: CLOUD_LOGGING_ONLY`)
- `tests/e2e_tests/e2e_test.py` — E2E test runner, submits Vertex AI PipelineJobs
- `tests/e2e_tests/e2e_tests.yaml` — E2E test definitions (8 pipeline tests)
- `gigl/orchestration/kubeflow/kfp_orchestrator.py` — Pipeline submission and wait logic
- `gigl/common/services/vertex_ai.py` — Vertex AI service (pipeline URLs at line 382, custom job URLs at line 316)
- `deployment/configs/e2e_cicd_resource_config.yaml` — GCP project and region config
