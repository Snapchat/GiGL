# Releasing GiGL

## Two-wheel model

GiGL is distributed as two wheels that are always installed together:

- **`gigl`** — pure Python (same wheel for CPU and CUDA users)
- **`gigl-core`** — compiled C++/CUDA extensions, ABI-bound to the torch variant

Each has its own version. `gigl-core` is versioned **independently** — it only needs a new release when C++ code under `gigl-core/` actually changes. `gigl` pins an exact `gigl-core` version in its `pyproject.toml`; that pin is updated in the same PR as any C++ change.

## How to trigger a release

Releases are triggered manually via the **Release GiGL** GitHub Actions workflow (`release.yml`):

1. Go to the **Actions** tab in the GitHub repository.
2. Select **Release GiGL** from the left sidebar.
3. Click **Run workflow** (top right of the workflow runs table).
4. A form appears with one input:
   - **Release gigl-core wheel** — check this only if C++ code changed (see below).
5. Click **Run workflow** to start.

The workflow runs two jobs in parallel — one on a CPU runner, one on a GPU runner — and publishes to both variant registries.

## Releasing gigl (Python-only change)

Use this when no files under `gigl-core/` were modified.

1. Bump the version in `gigl/pyproject.toml` (`version = "X.Y.Z"`).
2. Open a PR, get it merged to `main`.
3. Trigger the **Release GiGL** workflow with **Release gigl-core wheel** unchecked.

Both the CPU and CUDA `gigl` wheels are built and published. The existing `gigl-core` wheel in the registry continues to satisfy the pinned dependency — no `gigl-core` action needed.

## Releasing gigl-core (C++ changed)

Use this when files under `gigl-core/csrc/` or `gigl-core/CMakeLists.txt` were modified.

1. Bump the version in `gigl-core/pyproject.toml` (`version = "X.Y.Z"`).
2. Update the matching pin in `gigl/pyproject.toml`: `"gigl-core==X.Y.Z"`.
3. Open a PR with both version changes, get it merged to `main`.
4. Trigger the **Release GiGL** workflow with **Release gigl-core wheel** checked.

This publishes new `gigl-core` wheels (cpu + cu128) **and** new `gigl` wheels (which carry the updated pin).

## What gets published

Each release run publishes to two self-contained registries:

| Registry | Packages |
| -------- | -------- |
| `gcp-release-registry-cpu` | `gigl` (pure Python) + `gigl-core` (CPU wheel, if releasing) |
| `gcp-release-registry-cu128` | `gigl` (pure Python) + `gigl-core` (CUDA 12.8 wheel, if releasing) |

Users install from exactly one registry based on their variant — see [installation docs](docs/user_guide/getting_started/installation.md).
