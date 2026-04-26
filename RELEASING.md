# Releasing GiGL

## Two-wheel model

GiGL is distributed as two wheels that are always installed together:

- **`gigl`** — pure Python (same wheel for CPU and CUDA users)
- **`gigl-core`** — compiled C++/CUDA extensions, ABI-bound to the torch variant

Both wheels are always versioned and released together. `bump_version.py` updates both versions and keeps the
`gigl-core` pin in `gigl/pyproject.toml` in sync automatically.

## How to trigger a release

Releases are triggered manually via the **Release GiGL** GitHub Actions workflow (`release.yml`):

1. Go to the **Actions** tab in the GitHub repository.
2. Select **Release GiGL** from the left sidebar.
3. Click **Run workflow** (top right of the workflow runs table).
4. Click **Run workflow** to start.

The workflow runs two jobs in parallel — one on a CPU runner, one on a GPU runner — and publishes both `gigl` and
`gigl-core` wheels to both variant registries.

## Releasing GiGL

1. Run `bump_version.py` (or let `create_release.yml` handle it automatically for nightly builds).
2. Open a PR with the version bump, get it merged to `main`.
3. Trigger the **Release GiGL** workflow from the release branch.

## What gets published

Each release run publishes to two self-contained registries:

| Registry                     | Packages                                              |
| ---------------------------- | ----------------------------------------------------- |
| `gcp-release-registry-cpu`   | `gigl` (pure Python) + `gigl-core` (CPU wheel)        |
| `gcp-release-registry-cu128` | `gigl` (pure Python) + `gigl-core` (CUDA 12.8 wheel)  |

Users install from exactly one registry based on their variant — see
[installation docs](docs/user_guide/getting_started/installation.md).
