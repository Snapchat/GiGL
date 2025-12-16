## Adding a new Package as a Dependency

This document outlines the steps required to add a new Python package as a dependency in our codebase.

### 1. Package Addition

Open `pyproject.toml` file --> Add the package to its correct location (dev, transform, etc.). Guidance:

1. Freeze (set explicit versions e.g. `my-lib==1.2.3`) versions of doc deps as version updates changes how the doc pages
   are rendered, files are formatted, etc.
2. Freeze versions for deps that don't do a great job of maintaining in their wheels what libs / libraries they are
   compatible with - this is a problem with packages like tensorflow.
3. Everything else, let it auto resolve so we can keep flexibility for GiGL lib users.

### 2. Hash Generation

Run `uv lock` to generate a new `uv.lock` file, and create a PR with the changes.

### 3. Push new base docker images

- Navigate to GiGL's [Actions tab](https://github.com/Snapchat/GiGL/actions) and find the
  [build-base-docker-images workflow](https://github.com/Snapchat/GiGL/actions/workflows/build-base-docker-images.yml).
- Run the workflow with the PR# (You may need to ask a repo maintainer to do this). You can leave the branch name as
  `main`.
- Navigate back to your PR, you should see a message from `github-actions[bot]` specifying that your docker images are
  being built. The bot will push a new commit to the PR once it is done.

### 4. End-to-End Pipeline Test

To test that everything is working correctly, you should pre-emptively [run tests](../../../README.md#tests-) to ensure
that everything is still working.
