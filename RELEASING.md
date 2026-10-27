# Releasing GiGL

## Two-wheel model

GiGL is distributed as two wheels that are always installed together:

- **`gigl`** — pure Python (same wheel for CPU and CUDA users)
- **`gigl-core`** — compiled C++/CUDA extensions, ABI-bound to the torch variant

Both wheels are always versioned and released together. `bump_version.py` updates both versions and keeps the
`gigl-core` pin in `pyproject.toml` in sync automatically.

## Release process

A full release involves two GitHub Actions workflows run in sequence.

### Step 1 — Create Release (`create_release.yml`)

This workflow bumps the version, creates the release branch and tag, releases the KFP pipeline, and opens the merge-back
PR. Trigger it manually:

1. Go to the **Actions** tab in the GitHub repository.
2. Select **Create Release** from the left sidebar.
3. Click **Run workflow**, choose the bump type (`major`, `minor`, or `patch`), and click **Run workflow**.

The workflow will:

- Bump the version in `pyproject.toml` (both `gigl` and `gigl-core`) and commit it to a new `release/vX.Y.Z` branch.
- Release the GiGL KFP pipeline from that branch.
- Create and push a version tag `vX.Y.Z`.
- Open a PR to merge `release/vX.Y.Z` back to `main`.

### Step 2 — Release GiGL (`release.yml`)

This workflow builds and publishes the `gigl` and `gigl-core` wheels. Trigger it from the release branch created in Step
1:

1. Go to the **Actions** tab.
2. Select **Release GiGL** from the left sidebar.
3. Click **Run workflow**, select the `release/vX.Y.Z` branch from the branch dropdown, and click **Run workflow**.

The workflow runs two jobs in parallel — one on a CPU runner, one on a GPU runner — and publishes both wheels to both
variant registries.

### Step 3 — Merge the PR

Once both workflows succeed, merge the PR opened by **Create Release** to bring the version bump back into `main`.

## Refreshing pinned CI dependencies

Third-party GitHub Actions under `.github/` are pinned to full commit SHAs (with the release tag in a trailing comment).
Dependabot (`.github/dependabot.yml`) checks daily and opens one grouped PR when any pinned action has a newer release;
review and merge it like any other PR.

Dependabot does not cover the Cloud Build helper image `multiarch/qemu-user-static`, which is pinned by digest in
`.github/cloud_builder/run_command_on_active_checkout.yaml`. To refresh it, look up the current digest and update it
there:

```sh
docker buildx imagetools inspect multiarch/qemu-user-static:latest
```

## Nightly releases

Nightly builds are triggered automatically by `nightly_release_&_test.yml`, which calls `create_release.yml` with
`bump_type=nightly`. The **Release GiGL** wheel-publish step is not part of the nightly flow.

## Release identity and permissions

CI authenticates to GCP through Workload Identity Federation with two service accounts:

| Service account                  | Used by                                                                    | Artifact Registry access                                                        |
| -------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| PR/test (`untrusted-external-…`) | Every workflow job that does not run in the `release` environment          | `writer` on `gigl-base-images`; `reader` on `public-gigl`                       |
| Release (`gigl-release`)         | `create_release.yml` job `release_gigl_kfp_pipeline`; all of `release.yml` | `writer` on `public-gigl`, `gigl`, `gigl-cu128`; `reader` on `gigl-base-images` |

All workflows read the service account email from the secret `GCP_SERVICE_ACCOUNT_EMAIL`. The `release` GitHub
environment defines its own copy of that secret holding the release account, and an environment secret overrides the
repository secret of the same name. So a job gets the release account only when it declares `environment: release`.

The real control is on the GCP side. The release account's `roles/iam.workloadIdentityUser` binding matches only OIDC
tokens whose `sub` is `repo:Snapchat/GiGL:environment:release` and whose `job_workflow_ref` is `create_release.yml` on
`refs/heads/main` or `release.yml` on a `refs/heads/release/v*` branch. A PR job cannot obtain that token, so it cannot
impersonate the release account even if it names it.

Two consequences. `create_release.yml` must be dispatched from `main`. The `release` environment's branch policy must
admit `main` for that job to start at all; a dispatch from any other branch fails at the environment check, or at the
WIF token exchange if the branch is a `release/v*` branch. And the gate's floor is repo write access: anyone who can
create a `release/v*` branch can run `release.yml` from it. Closing that needs a ruleset that restricts creation of
branches matching the same pattern as the environment policy, and `bump_version` pushing with a dedicated GitHub App
token that is the only bypass actor. Every workflow's `GITHUB_TOKEN` acts as the same GitHub Actions app, so a bypass
for that app would cover every workflow.

When you add a new publish target (a registry, a bucket path), grant it to the release account only, and run the
publishing step in a job that declares `environment: release`. Do not widen the PR/test account.

## What gets published

Each release run publishes to two self-contained registries:

| Registry                     | Packages                                             |
| ---------------------------- | ---------------------------------------------------- |
| `gcp-release-registry-cpu`   | `gigl` (pure Python) + `gigl-core` (CPU wheel)       |
| `gcp-release-registry-cu128` | `gigl` (pure Python) + `gigl-core` (CUDA 12.8 wheel) |

Users install from exactly one registry based on their variant — see
[installation docs](docs/user_guide/getting_started/installation.md).
