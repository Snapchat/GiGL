## Adding a new Package as a Dependency

This document outlines the steps required to add a new Python package as a dependency in our codebase.

### 1. Package Addition

Open `pyproject.toml` file --> Add the package to its correct location (dev, transform, etc.).

### 2. Hash Generation

- Create a PR with the changes to the deps, take note of your PR#
- Once PR is created, navigate to GiGL's Actions tab and find the
  [generate-hashed-reqs workflow](https://github.com/Snapchat/GiGL/actions/workflows/generate-hashed-requirements.yml).
- Run the workflow with the PR# (You may need to ask a repo maintainer to do this). You can leave the branch name as
  `main`.
- Navigate back to your PR, you should see a message from `github-actions[bot]` specifying that your hashed requirements
  are being built. The bot will push a new commit to the PR once it is done.

### 3. Push new base docker images

- Similar to the Hash Generation step, navigate to GiGL's Actions tab and find the
  [build-base-docker-images workflow](https://github.com/Snapchat/GiGL/actions/workflows/build-base-docker-images.yml).
- Run the workflow with the PR# (You may need to ask a repo maintainer to do this). You can leave the branch name as
  `main`.
- Navigate back to your PR, you should see a message from `github-actions[bot]` specifying that your docker images are
  being built. The bot will push a new commit to the PR once it is done.

### 4. End-to-End Pipeline Test

To test that everything is working correctly, you should pre-emptively [run tests](../../../README.md#tests-) to ensure
that everything is still working.
