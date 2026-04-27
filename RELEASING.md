# Releasing GiGL

## Two-wheel model

GiGL is distributed as two wheels that are always installed together:

- **`gigl`** — pure Python (same wheel for CPU and CUDA users)
- **`gigl-core`** — compiled C++/CUDA extensions, ABI-bound to the torch variant

Both wheels are always versioned and released together. `bump_version.py` updates both versions and keeps the
`gigl-core` pin in `gigl/pyproject.toml` in sync automatically.

## Release process

A full release involves two GitHub Actions workflows run in sequence.

### Step 1 — Create Release (`create_release.yml`)

This workflow bumps the version, creates the release branch and tag, releases the KFP pipeline, and opens the
merge-back PR. Trigger it manually:

1. Go to the **Actions** tab in the GitHub repository.
2. Select **Create Release** from the left sidebar.
3. Click **Run workflow**, choose the bump type (`major`, `minor`, or `patch`), and click **Run workflow**.

The workflow will:

- Bump the version in `pyproject.toml` (both `gigl` and `gigl-core`) and commit it to a new `release/vX.Y.Z` branch.
- Release the GiGL KFP pipeline from that branch.
- Create and push a version tag `vX.Y.Z`.
- Open a PR to merge `release/vX.Y.Z` back to `main`.

### Step 2 — Release GiGL (`release.yml`)

This workflow builds and publishes the `gigl` and `gigl-core` wheels. Trigger it from the release branch created in
Step 1:

1. Go to the **Actions** tab.
2. Select **Release GiGL** from the left sidebar.
3. Click **Run workflow**, select the `release/vX.Y.Z` branch from the branch dropdown, and click **Run workflow**.

The workflow runs two jobs in parallel — one on a CPU runner, one on a GPU runner — and publishes both wheels to both
variant registries.

### Step 3 — Merge the PR

Once both workflows succeed, merge the PR opened by **Create Release** to bring the version bump back into `main`.

## Nightly releases

Nightly builds are triggered automatically by `nightly_release_&_test.yml`, which calls `create_release.yml` with
`bump_type=nightly`. The **Release GiGL** wheel-publish step is not part of the nightly flow.

## What gets published

Each release run publishes to two self-contained registries:

| Registry                     | Packages                                              |
| ---------------------------- | ----------------------------------------------------- |
| `gcp-release-registry-cpu`   | `gigl` (pure Python) + `gigl-core` (CPU wheel)        |
| `gcp-release-registry-cu128` | `gigl` (pure Python) + `gigl-core` (CUDA 12.8 wheel)  |

Users install from exactly one registry based on their variant — see
[installation docs](docs/user_guide/getting_started/installation.md).
