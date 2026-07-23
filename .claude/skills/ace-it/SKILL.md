---
name: ace-it
description: Use when writing, adding, or porting any test — before creating a test file or test method, and before reaching for mock.patch or assert_called_with.
---

# ace-it — Writing Tests

## Overview

A test earns its place by catching a real behavior regression. If a test can only fail when someone rewrites the test
itself (because it asserts how the code is wired, not what it produces), it is noise — delete it, don't write it.

**Core principle:** Prefer one test that exercises real behavior over many that mock collaborators. Match the number of
tests to the risk of the change, not to the number of functions you touched.

## When to Use

Invoke this skill **before writing any test** — a new file, a new method, or porting tests from another branch. Also
invoke it the moment you reach for `mock.patch` or `assert_called_with`.

## The Recipe: what a good test is

A good test in this repo does three things, in order:

1. **Arrange real inputs** — build the real object, write a real (small) YAML/fixture, construct a real `nn.Module`.
   Mock only true external boundaries (subprocess, GCP / Vertex / GCS clients, `torch.distributed`, network) — this is
   the "Mock external services" rule from `CLAUDE.md`, and nothing more.
2. **Act** — call the real function or method under test.
3. **Assert on real outputs** — the returned value, the built object's fields, the raised error. **Assert as many real
   facts as belong together in one method** — this is the blessed style here, not a smell:

```python
def test_from_yaml_loads_values(self) -> None:
    hyperparams = HyperParams.from_yaml(UriFactory.create_uri(self.yaml_path))
    self.assertEqual(hyperparams.graph_sampler_config.main_batch_size, 32)
    self.assertEqual(hyperparams.optim_config.name, "AdamW")  # default backfilled
    self.assertEqual(hyperparams.seed, 7)
```

One real behavior test like this beats five that patch collaborators.

## What deserves a test (match count to risk)

| Change                                                                              | Tests to write                                          |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Real logic: parsing, transforms, partitioning, math, state transitions, error paths | Test it — the higher the branching/risk, the more cases |
| Thin / mechanical change (rename, config value, forwarding a new kwarg)             | Usually **zero** new tests                              |
| **Glue**: reads inputs, composes args, calls a submit/CLI/client                    | **No unit test.** That's e2e's job — see below          |

**Glue gets no unit test.** If the function's whole job is to assemble arguments and hand them to a launcher, submitter,
or CLI, do not mock that collaborator to prove "it called X with Y." That test just restates the implementation and
breaks on every refactor. Cover glue at the e2e/integration layer where the real call runs.

## Mock smell

**If a test body is mostly `mock.patch(...)` + `assert_called_with(...)` / `assert_called_once()`, don't write it.**

```python
# ❌ Mock smell — mocks the submit, then reads call_args to inspect what it built.
# This block gets copy-pasted across many methods; each asserts plumbing, not behavior.
with (
    mock.patch.dict(os.environ, _env(GiGLComponents.Trainer), clear=True),
    mock.patch("gigl.orchestration.launch.GbmlConfigPbWrapper.get_gbml_config_from_uri", ...),
    mock.patch("gigl.orchestration.launch.submit_job") as mock_submit,
):
    run_component(_base_args(), GiGLComponents.Trainer)
job_spec = mock_submit.call_args.args[0]               # <- reaching into the mock
self.assertEqual(job_spec.image, _CUDA_URI)
```

**The tell: if you read `mock.call_args` to reach the object the function built, that object should be *returned* (or
built by a pure helper) so you can assert on it directly.** Then you don't mock `submit_job` at all:

```python
# ✅ Build with a pure helper, mock only the true external edge (the GCS config read),
#    and assert on the real returned object. One method covers many facts.
spec = build_job_spec(_base_args(image_kind="cpu"), GiGLComponents.Trainer, wrapper)
self.assertEqual(spec.image, _CPU_URI)
self.assertEqual(spec.job_name, "trainer-cora-test-on-123")   # normalized
self.assertLessEqual(len(spec.job_name), 63)                  # under the job-name length limit
```

Also **delete pure-wiring tests** — e.g. a dispatch test that mocks `CliApp.run` and the component runner just to
`assert_called_once_with(...)` tests only that the code calls what the code calls. And **collapse redundant cases**: two
tests asserting the same image-kind→URI mapping for trainer vs inferencer are one behavior, not two.

Exception — mocking the external edge is fine **when the interaction with that edge is the behavior under test** (e.g.
the launcher must set specific env vars for the subprocess). Even then, prefer asserting on observable state over
`assert_called`. A good exemplar mocks only the GCP exporter, then exercises real instruments and asserts the real
recorded values.

**Worth keeping vs. noise** — from a launch/orchestration glue module: the real logic (name normalization, image-kind
mapping, env/component validation via `assertRaises`) earns a few tests against the real built object; the
entrypoint-string composition and flag pass-through are glue — cover them e2e, not with `call_args`.

## Ported / inherited tests aren't exempt

When porting tests from another branch, evaluate each one against the rules above and **delete the mock-heavy / glue
ones**. "It already existed" is not a reason to keep a test that only asserts plumbing. Port the real-behavior tests;
drop the noise.

## Repo conventions

- Filename `*_test.py`; class-based (`class XxxTest(...)`, methods `test_*`).
- Base class: `tests.test_assets.test_case.TestCase`, **not** `unittest.TestCase` (per `CLAUDE.md`).
- Error paths: `self.assertRaises` / `assertRaisesRegex`. Don't assert exact message strings unless the message is
  load-bearing.
- Run: `make unit_test_py PY_TEST_FILES="my_thing_test.py"` (filename only, not path).

## Rationalization table

| Excuse                                                | Reality                                                                                            |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| "Every function should have a test"                   | Only behavior with regression risk needs one. Glue and mechanical changes often need zero.         |
| "I'll mock the collaborator so it's a real unit test" | Mocking internal collaborators tests wiring, not behavior. Test the real thing or leave it to e2e. |
| "assert_called_with proves it works"                  | It proves the code calls what the code calls. It breaks on refactor and catches no real bug.       |
| "This test already existed on the other branch"       | Ported ≠ exempt. Delete it if it's mock plumbing.                                                  |
| "More tests = safer"                                  | Noisy tests hide the ones that matter and rot on every refactor. Fewer, real tests are safer.      |
| "One assertion per test is cleaner"                   | No such rule here. Assert every related fact in one method.                                        |

## Red Flags — STOP

- The test body is mostly `@patch` decorators and `assert_called*`.
- You're reading `mock.call_args` / `call_args.args[0]` to reach the object the function built.
- The same multi-`patch` `with` block is copy-pasted across many methods.
- You're mocking a collaborator you wrote (not an external service).
- The test would still pass if the function's real logic were wrong, and only fail if someone changed which helper it
  calls.
- You're adding a unit test for a function whose only job is to submit/launch/forward.
- You're keeping a ported test only because it was already there.

**All of these mean: don't write it (or delete it). Reach for a real-behavior test, or push coverage to e2e.**
