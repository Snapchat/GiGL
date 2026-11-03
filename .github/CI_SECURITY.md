# CI security: PR-command TOCTOU hardening

This repo runs privileged CI against pull-request code (tests and base-image builds). Because contributors work from
forks, those workflows are hardened against a time-of-check-to-time-of-use (TOCTOU) race where code pushed *after* a
maintainer authorizes a run would execute in place of the reviewed code, with GCP Workload Identity Federation
credentials.

## How the hardening works

1. **Freeze the head SHA once.** The PR-comment workflow (`on-pr-comment.yml`) has a `prepare` job that resolves the PR
   head commit once (via `actions/get-pr-src-branch`) and posts it to the PR. Every command job pins to that single
   frozen SHA.
2. **Human approval bound to the SHA — for fork PRs only.** For a cross-repository (fork) PR, the command jobs attach to
   the `pr-command-execution` GitHub Environment (see setup below) and a reviewer approves the run knowing the frozen
   SHA posted by `prepare`. For same-repo PRs the environment resolves to an empty string (no approval gate): only
   collaborators can push branches to this repo, so the author is already trusted and there is no untrusted party to
   race the checkout. This keeps the normal `/all_test` flow friction-free for maintainers while gating only untrusted
   fork code. The collaborator check and the SHA pin still apply to both paths.
3. **Pinned, verified checkout.** `actions/checkout-pr-branch` takes a required `sha` input, checks out the PR head, and
   **fails closed** if `git rev-parse HEAD` is not that SHA. A push that races the approval moves the tip and aborts the
   job before any PR code runs.
4. **Least privilege + a writeback job that runs no PR code.** `checkout-pr-branch` always checks out PR code with
   `persist-credentials: false`. In `build-base-docker-images.yml`, the `build-and-commit-base-images` job holds the
   `contents: write` token, so it never executes PR-authored code: it checks out the PR at the reviewed `commit_sha`
   (verified) and treats it as data only, then checks out the workflow's own commit (`github.sha`) into `.trusted` and
   runs the image-ref update script from there. The commit step has no token in its environment and checks that only
   `gigl/dep_vars.env` and `.github/cloud_builder/run_command_on_active_checkout.yaml` changed. Every git call disables
   hooks and `core.fsmonitor`, and only the final push step sees `GITHUB_TOKEN`, supplied inline via `git -c` rather
   than persisted to `.git/config`. Fork PRs skip writeback (see below). Jobs also declare scoped `permissions`.
5. **Fail-closed authorization.** `actions/assert-is-collaborator` now denies on any result other than an explicit 204
   (previously a 403 was logged and allowed through).

The manually dispatched `build-base-docker-images.yml` workflow is `workflow_dispatch`-gated (maintainer-triggered) and
takes a required **`commit_sha`** input: paste the exact commit you reviewed. All its jobs pin to it and abort if the PR
head has moved.

## Required repository setup (one-time, done in Settings — not in YAML)

Create a GitHub **Environment** named **`pr-command-execution`** (Settings → Environments →
`https://github.com/<owner>/<repo>/settings/environments`):

1. **New environment** → name `pr-command-execution`.
2. Enable **Required reviewers** and add the maintainers/team allowed to approve fork-PR command runs.
3. (Recommended) Confirm the Workload Identity Federation provider's attribute condition binds to this repository (and
   ideally to this environment), and that the mapped service account has minimal roles.

> **This is a hard precondition — the approval gate fails OPEN without it.** If a workflow references an environment
> that does not exist, GitHub **auto-creates it with no protection rules**, so fork-PR command jobs would run
> immediately with no approval (the freeze→execute race would still be open). The gate only exists once this environment
> has a **required reviewers** rule (creating the environment with no protection rules does nothing). Use a **team** as
> the reviewer so you are not capped at 6 individuals — any one team member can approve, and only one approval is needed
> per run.

> **This is NOT the same as the Actions setting "Require approval for … external contributors"** (Settings → Actions →
> General → Fork pull request workflows). That setting gates workflow runs triggered by the **`pull_request`** event
> from forks. Our command flow is triggered by **`issue_comment`**, which that setting does **not** gate (this is
> exactly why the IssueOps TOCTOU exists). The two are complementary: keep the external-contributor setting for the
> `pull_request`/CI path, and add this Environment for the comment-command path.

**Mechanically enforced:** the `prepare` job runs a preflight (for fork PRs only) that calls
`GET /repos/{owner}/{repo}/environments/pr-command-execution` and **fails closed** if the environment is missing or has
no required-reviewers rule — so a removed/misconfigured environment blocks fork commands instead of silently failing
open. The `prepare` job explicitly grants its workflow token `actions: read`, which is required to inspect the
environment. If that API call ever fails, the preflight fails closed and the error explains why.

## Known limitation

For `issue_comment` events the Environment approval UI shows the base-branch commit, not the frozen fork-head SHA. The
frozen SHA is surfaced in the `prepare` job's PR comment; reviewers must approve based on that comment. This is a
process control, not a purely mechanical guarantee.

## Writeback works only for same-repo PRs

`build-base-docker-images.yml` commits generated files back to the PR branch. `GITHUB_TOKEN` cannot push to a fork, so
the writeback job detects fork PRs up front and skips every writeback step. Instead it comments on the PR with the built
image refs, and the author must update `gigl/dep_vars.env` and
`.github/cloud_builder/run_command_on_active_checkout.yaml` themselves.

## Deferred (defense-in-depth follow-up)

Registry hardening for `build-base-docker-images.yml` (quarantine/SHA-scoped image tags → scan → promote to
`public-gigl`; isolated build project/service account) and self-hosted runner isolation are not included here. Under the
assumption that maintainers are trusted, these contain blast radius if a bad image is ever built rather than preventing
the primary race, which the `commit_sha` pin closes.
