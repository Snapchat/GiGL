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
   SHA posted by `prepare`. For same-repo PRs the environment resolves to an empty string (no approval gate): the
   maintainer who commented has reviewed the frozen SHA, only collaborators can push to the branch, and the SHA pin
   aborts the run if anyone pushes after the comment. This keeps the normal `/all_test` flow friction-free for
   maintainers while gating only untrusted fork code. The maintainer check and the SHA pin still apply to both paths.
3. **Pinned, verified checkout.** `actions/checkout-pr-branch` takes a required `sha` input, checks out the PR head, and
   **fails closed** if `git rev-parse HEAD` is not that SHA. A push that races the approval moves the tip and aborts the
   job before any PR code runs.
4. **Least privilege + a writeback job that runs no PR code.** Jobs declare scoped `permissions`, and
   `checkout-pr-branch` always checks out PR code with `persist-credentials: false`. In `build-base-docker-images.yml`,
   the `build-and-commit-base-images` job holds the `contents: write` token, so it never executes PR-authored code:
   - It checks out the PR at the reviewed `commit_sha` (verified) and treats it as data only.
   - It checks out the workflow's own commit (`github.sha`) into `.trusted` and runs the image-ref update script from
     there.
   - The commit step has no token in its environment. It fails if any file other than `gigl/dep_vars.env` and
     `.github/cloud_builder/run_command_on_active_checkout.yaml` changed.
   - Every git call disables hooks and `core.fsmonitor`.
   - Only the final push step sees `GITHUB_TOKEN`. It passes the token to git through an inline credential helper that
     reads it from the step's env, so the token is not in argv or persisted to `.git/config`.
   - Fork PRs skip writeback (see below).
5. **Fail-closed authorization.** `actions/assert-is-maintainer` allows a command only when the commenter's repository
   role is `maintain` or `admin`. Write-role collaborators, and any API error, are denied.

The manually dispatched `build-base-docker-images.yml` workflow takes a required **`commit_sha`** input: paste the exact
commit you reviewed. All its jobs pin to it and abort if the PR head has moved. Its four build jobs run PR code with GCP
credentials, so they always attach to the `pr-command-execution` environment, for same-repo and fork PRs alike. A
reviewer must approve the run before any build starts. The first job fails closed if that environment has no
required-reviewers rule or allows self-review.

## Required repository setup (one-time, done in Settings — not in YAML)

Create a GitHub **Environment** named **`pr-command-execution`** (Settings → Environments →
`https://github.com/<owner>/<repo>/settings/environments`):

1. **New environment** → name `pr-command-execution`.
2. Enable **Required reviewers** and add the maintainers/team allowed to approve fork-PR command runs and base-image
   builds.
3. Enable **Prevent self-review**, so the person who triggered a run cannot approve it.
4. Give everyone who should run `/all_test` and other commands the **Maintain** (or Admin) role. Write is not enough.
5. (Recommended) Confirm the Workload Identity Federation binding for the PR/test service account allows only the
   workflow files and refs that need it (see `job_workflow_ref` in the GitHub OIDC token), and that the account has
   minimal roles.

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

**Mechanically enforced:** `actions/assert-environment-protected` calls
`GET /repos/{owner}/{repo}/environments/pr-command-execution` and **fails closed** if the environment is missing or has
no required-reviewers rule, or if that rule does not have **Prevent self-review** enabled — so a removed or
misconfigured environment blocks the gated jobs instead of silently failing open. It runs in the `prepare` job of
`on-pr-comment.yml` (fork PRs only) and in the first job of `build-base-docker-images.yml` (every run). Both jobs grant
their workflow token `actions: read`, which is required to inspect the environment. If that API call ever fails, the
preflight fails closed and the error explains why.

## Known limitation

For `issue_comment` events the Environment approval UI shows the base-branch commit, not the frozen fork-head SHA. The
frozen SHA is surfaced in the `prepare` job's PR comment; reviewers must approve based on that comment. The same applies
to `build-base-docker-images.yml`: the approval UI shows the dispatched ref, so reviewers approve based on the
`commit_sha` posted in the PR comment. This is a process control, not a purely mechanical guarantee.

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
