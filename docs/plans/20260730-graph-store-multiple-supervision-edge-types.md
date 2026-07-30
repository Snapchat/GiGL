# GraphStore multiple-supervision-edge support

## Objective

Enable one GraphStore `DistABLPLoader` to fetch, sample, and return labels for multiple heterogeneous supervision edge
types that share one anchor node type, without regressing the existing one-edge, labeled-homogeneous, sharded-fetch, or
incoming-edge behavior.

The task metadata and graph materialization layers are already plural: the task proto stores repeated supervision edge
types, its wrapper returns a list, and the dataset factory passes that list into the splitter
(`proto/snapchat/research/gbml/gbml_config.proto:18-34`, `gigl/src/common/types/pb_wrappers/task_metadata.py:58-88`,
`gigl/distributed/dataset_factory.py:568-585`). The singular RPC/fetch boundary is the main TODO
(`gigl/distributed/graph_store/remote_dist_dataset.py:380-446`), but correct per-type negative output also requires a
loader change because positive and negative results are currently flattened independently
(`gigl/distributed/dist_ablp_neighborloader.py:876-886`).

## Scope and compatibility decisions

### In scope

1. Widen `RemoteDistDataset.fetch_ablp_input()` from one supervision edge type to one or many, while keeping scalar
   input source-compatible (`gigl/distributed/graph_store/remote_dist_dataset.py:447-586`).
2. Batch all requested edge types into the existing one-request-per-storage-server fan-out, rather than issuing one RPC
   per edge type (`gigl/distributed/graph_store/remote_dist_dataset.py:388-426`).
3. Return the existing `ABLPInputNodes` schema, whose `labels` member is already a dictionary from supervision edge type
   to `(positive, optional negative)` tensors (`gigl/utils/sampling.py:91-142`).
4. Remove the GraphStore-only one-edge loader restriction and validate the full per-server schema before constructing
   sampler inputs (`gigl/distributed/dist_ablp_neighborloader.py:676-713`).
5. Support per-edge negative-label availability. Today the loader compares the whole `(positive, negative)` tuple with
   `None`, so any label entry makes the global `has_negatives` flag true
   (`gigl/distributed/dist_ablp_neighborloader.py:684-703`, `gigl/utils/sampling.py:139-142`).
6. Prove both the RPC data path and final multi-key `y_positive`/`y_negative` output, including `edge_dir="in"`,
   sharding, and an edge type without hard negatives.
7. Align the preferred public API with colocated mode's anchor-outward supervision types while retaining the existing
   incoming GraphStore scalar call as a compatibility form.

### Direction contract

The preferred public contract will match colocated mode: caller-provided supervision types are anchor-outward, so every
type has `src_node_type == anchor_node_type`. Colocated mode already enforces that rule and reverses the types
internally for incoming sampling (`gigl/distributed/dist_ablp_neighborloader.py:471-490`). Canonical dataset
construction also starts from the task-metadata list and reverses label edges before registering an incoming graph
(`gigl/distributed/dataset_factory.py:568-585`, `gigl/types/graph.py:110-131`).

For `edge_dir="in"`, `RemoteDistDataset` will therefore translate the canonical outward list to registered sampling
orientation before constructing the internal request:

```text
public/task type       registered request key       final loader key
A --r1--> B            B --r1--> A                  A --r1--> B
A --r2--> C            C --r2--> A                  A --r2--> C
```

The final reversal already occurs during collation (`gigl/distributed/dist_ablp_neighborloader.py:905-921`). For
`edge_dir="out"`, the public, registered, and final keys are identical.

Backward compatibility is necessary because the current GraphStore example treats the destination of `paper -> author`
as the anchor and passes that stored orientation directly
(`examples/link_prediction/graph_store/heterogeneous_training.py:180-202`); its storage config documents a manual
reverse of the splitter input
(`examples/link_prediction/graph_store/configs/e2e_het_dblp_sup_gs_task_config.yaml:55-67`). For incoming graphs only,
accept a legacy request when all supplied types have `dst_node_type == anchor_node_type`; those types are already
registered orientation and are not reversed again. Reject a list that mixes canonical and legacy orientations. When both
endpoints have the anchor type, reversal produces the same node-type endpoints, so resolve by positive-label topology
and reject only if neither candidate exists. This codifies an implicit convention as new validation; the current client
checks only optional-argument pairing (`gigl/distributed/graph_store/remote_dist_dataset.py:545-558`).

### Explicit non-goals

- No proto, partition format, or graph registration change. `EdgeType` is already hashable, graph labels are stored by
  edge type, and partitioned label maps are already dictionaries (`gigl/src/common/types/graph_data.py:24-33`,
  `gigl/types/graph.py:60-91`, `gigl/distributed/dist_partitioner.py:150-178`).
- No change to neighbor/PPR sampling algorithms. `ABLPNodeSamplerInput` already slices and shares every dictionary entry
  (`gigl/distributed/sampler.py:13-75`), and the base sampler already visits all positive and negative label types when
  building seeds and metadata (`gigl/distributed/base_sampler.py:129-199`).
- No promise that the example trainer can optimize multiple objectives. It still rejects task metadata containing more
  than one type and assumes `main_data.y_positive` is a single tensor
  (`examples/link_prediction/graph_store/heterogeneous_training.py:927-934`, `:297-315`). Updating loss aggregation,
  random-negative semantics, and trainer configuration should be a follow-up after the core loader contract is proven.
- No mixed-version compute/storage rollout. GraphStore RPC sends a Python callable and Python arguments directly, so the
  request/response dataclass change assumes compute and storage use the same release
  (`gigl/distributed/graph_store/compute.py:99-124`).

## Current data flow and the actual choke points

1. `RemoteDistDataset._fetch_ablp_input()` builds one request per selected server, but each request contains one
   `supervision_edge_type` and each future returns one `(anchors, positive, negative)` tuple
   (`gigl/distributed/graph_store/remote_dist_dataset.py:380-444`).
2. `FetchABLPInputRequest` encodes that singular field (`gigl/distributed/graph_store/messages.py:90-117`).
3. `DistServer.get_ablp_input()` fetches anchors once, resolves one positive and optional negative label topology, and
   returns one label pair (`gigl/distributed/graph_store/dist_server.py:542-576`).
4. The public fetch method wraps that one pair into an `ABLPInputNodes.labels` dictionary, despite the dictionary
   already being able to carry several keys (`gigl/distributed/graph_store/remote_dist_dataset.py:545-586`,
   `gigl/utils/sampling.py:139-142`).
5. GraphStore loader setup reads schema only from the first server, computes a faulty global negative flag, then
   explicitly rejects any key count other than one (`gigl/distributed/dist_ablp_neighborloader.py:676-713`).
6. Everything after that guard is already structured as dictionaries: loader conversion loops over all label entries
   (`gigl/distributed/dist_ablp_neighborloader.py:718-762`), sampling adds every label type to metadata
   (`gigl/distributed/base_sampler.py:152-199`), remapping handles dictionaries
   (`gigl/distributed/utils/ablp.py:97-160`), and output is flattened only for one result while multiple positive
   results remain dictionaries (`gigl/distributed/dist_ablp_neighborloader.py:838-886`).
7. Positive and negative outputs are flattened independently. With two positive types but hard negatives for only one
   type, `y_negative` becomes unkeyed and loses its relation (`gigl/distributed/dist_ablp_neighborloader.py:876-886`).

The colocated path demonstrates the intended cardinality behavior. It normalizes a scalar or non-empty list
(`gigl/distributed/dist_ablp_neighborloader.py:263-274`), resolves and fetches labels once per type
(`gigl/distributed/dist_ablp_neighborloader.py:538-570`), and already has positive-only, positive-and-negative,
same-endpoint/different-relation, and incoming-direction multi-type cases
(`tests/unit/distributed/dist_ablp_neighborloader_test.py:1095-1277`).

## Proposed implementation

### 1. Make the RPC contract plural and reuse `ABLPInputNodes`

Files:

- `gigl/distributed/graph_store/messages.py`
- `gigl/distributed/graph_store/dist_server.py`
- `tests/unit/distributed/graph_store/messages_test.py`
- `tests/unit/distributed/dist_server_test.py`

Change `FetchABLPInputRequest.supervision_edge_type` to `supervision_edge_types: tuple[EdgeType, ...]`. A tuple keeps
the frozen request structurally immutable. The field represents registered sampling-orientation types; public
orientation resolution happens before request construction. Update its documentation/examples.

Change `DistServer.get_ablp_input()` to return `ABLPInputNodes`:

1. Before fetching anchors, reject an empty tuple, duplicates, and a registered type whose effective anchor endpoint is
   wrong (`src` for `out`, `dst` for `in`). This repeats validation at the authoritative RPC boundary because callers
   can invoke the server helper without the public normalizer (`gigl/distributed/graph_store/compute.py:99-124`).
2. Fetch and slice anchors exactly once.
3. Iterate `request.supervision_edge_types` in caller order.
4. For each type, call `select_label_edge_types()` against the registered edge types, then
   `get_labels_for_anchor_nodes()` with the same anchors and `max_labels_per_anchor_node`.
5. Store the returned pair under the registered supervision edge type in `labels`.
6. Return one `ABLPInputNodes(anchor_node_type=request.node_type, ...)`.

`get_labels_for_anchor_nodes()` already defines the required per-type contract: positive and optional negative tensors
are `[N, M]`, padding uses `-1`, and an empty anchor set returns `[0, 0]` tensors while preserving whether a negative
topology exists (`gigl/utils/data_splitters.py:607-682`).

This preserves one RPC per selected server. The client currently dispatches one future per server
(`gigl/distributed/graph_store/remote_dist_dataset.py:415-426`); an edge-by-server fan-out would multiply calls while
the client RPC pool defaults to four workers (`gigl/distributed/graph_store/compute.py:127-175`).

### 2. Normalize and validate the public fetch API

Files:

- `gigl/distributed/graph_store/remote_dist_dataset.py`
- `gigl/distributed/graph_store/dist_server.py`

Widen the existing keyword without renaming it:

```python
supervision_edge_type: Optional[Union[EdgeType, list[EdgeType]]] = None
```

Normalize `None` to `[DEFAULT_HOMOGENEOUS_EDGE_TYPE]`, a scalar to a one-item list, and a list to a copied list. Reject:

- an empty list;
- duplicate edge types;
- only one of `anchor_node_type` and `supervision_edge_type` being supplied;
- a heterogeneous list that is neither uniformly canonical anchor-outward nor uniformly legacy incoming registered
  orientation;
- a list mixing canonical and legacy incoming orientations;
- more than one type for the labeled-homogeneous default.

Fetch `edge_dir` and registered edge types once. Resolve canonical outward inputs to registered orientation, verify
positive-label topology for every resolved type, and record per-type negative-topology presence with
`select_label_edge_types()` (`gigl/types/graph.py:349-372`). Preserve a uniformly legacy incoming list as described
above. Pass only the resolved tuple into every per-server request, and have `_fetch_ablp_input()` return
`dict[int, ABLPInputNodes]` directly.

Keep the public fetch result dense for compatibility, even though the loader can already accept an in-range sparse
mapping and synthesize missing-rank inputs (`gigl/distributed/dist_ablp_neighborloader.py:663-675`, `:718-762`). For
every unrequested rank, construct a topology-complete placeholder: empty anchors, one `[0, 0]` positive tensor per
resolved type, and either a `[0, 0]` negative tensor or `None` according to that type's registered negative topology.
This removes the need to infer placeholder provenance later; a provided `None` where topology requires negatives is
uniformly invalid.

Retain request order in all dictionaries for deterministic logs and tests, while using set equality only for schema
validation.

### 3. Make GraphStore loader setup schema-safe and per-edge

File: `gigl/distributed/dist_ablp_neighborloader.py`

Replace the first-entry/global-boolean logic with full validation:

1. Require a non-empty `input_nodes` mapping and non-empty label schema.
2. Establish the expected `anchor_node_type` and ordered supervision-key tuple from the first entry.
3. For every server entry, require the same anchor type, the same supervision key set, a one-dimensional integral anchor
   tensor, and two-dimensional integral positive/negative tensors whose first dimension equals the anchor count.
   Remapping indexes dimension one, so accepting a one-dimensional label tensor would only defer failure
   (`gigl/distributed/utils/ablp.py:62-76`).
4. Validate every supervision type has the effective anchor endpoint required by the dataset's `edge_dir`.
5. Resolve each positive and optional negative label edge type independently against `edge_types` using
   `select_label_edge_types()`. This makes topology, not “some server returned a tensor,” the source of truth.
6. Uniformly require a positive tensor and require a negative tensor exactly when that edge type has registered negative
   topology; remote placeholders are already schema-complete.
7. Remove the heterogeneous GraphStore `len(...) != 1` guard at `gigl/distributed/dist_ablp_neighborloader.py:706-713`.
8. Preserve the later labeled-homogeneous exactly-one invariant
   (`gigl/distributed/dist_ablp_neighborloader.py:923-930`).

This also fixes mixed negative availability: `_negative_label_edge_types` becomes the subset selected from graph
topology, and each `ABLPNodeSamplerInput` receives only its actual negative keys. No single `has_negatives` flag should
control all edge types.

Prefer extracting the validation/normalization into a small private helper that returns a validated schema plus the
dense `list[ABLPNodeSamplerInput]`. That makes failure cases testable without starting sampling workers.

Move local validation before GraphStore worker-option creation and port allocation. The current order fetches a port
before the input schema is fully checked (`gigl/distributed/dist_ablp_neighborloader.py:643-678`,
`gigl/distributed/base_dist_loader.py:600-624`). Gather a compact local success/error result across compute ranks; if
any rank failed validation, make all ranks raise before collective port allocation or backend registration.

Continue accepting sparse in-range `input_nodes` mappings. Only negative or out-of-range server ranks are rank errors;
missing ranks are not.

### 4. Preserve relation keys for mixed negative output

File: `gigl/distributed/dist_ablp_neighborloader.py`

Change `_set_labels()` flattening to depend on supervision cardinality:

- if the loader has one supervision type, retain the existing bare tensor/ragged-dictionary `y_positive` and
  `y_negative` API;
- if it has more than one supervision type, keep both outputs edge-keyed; `y_negative` contains only the types with
  negative topology, even when that subset has length one;
- if no type has negatives, continue omitting `y_negative`.

This is required because the current independent `len(output_negative_labels)` check loses the only negative edge key in
the mixed case (`gigl/distributed/dist_ablp_neighborloader.py:876-886`).

### 5. Update API documentation and logging

Files:

- `gigl/distributed/graph_store/remote_dist_dataset.py`
- `gigl/distributed/graph_store/messages.py`
- `gigl/distributed/graph_store/dist_server.py`
- `gigl/distributed/dist_ablp_neighborloader.py`

Document:

- scalar and list inputs;
- canonical anchor-outward input, legacy incoming compatibility, and rejection of mixed orientation;
- the exact public key -> registered key -> final key mapping for `edge_dir="in"`;
- dictionary output for multiple final label types;
- per-type optional negatives;
- labeled-homogeneous remaining single-type.

Log the ordered list of requested supervision types and the independently resolved positive/negative label types. Do not
log label tensors or node IDs.

## Test plan

### Unit: request and server fetch contract

Extend `tests/unit/distributed/graph_store/messages_test.py`, whose current ABLP case asserts the singular request
(`tests/unit/distributed/graph_store/messages_test.py:33-55`):

- plural tuple construction, equality, frozen behavior, and optional slice;
- one-item tuple as the backward-compatible normalized form.

Update every existing ABLP case in `tests/unit/distributed/dist_server_test.py`. That tracked module already pins the
singular request and tuple response contract (`tests/unit/distributed/dist_server_test.py:345-595`). Add:

- anchors are fetched/sliced once and reused for two supervision types;
- exact positive and negative tensor values are keyed by both types;
- one type with negatives plus one without negatives;
- empty anchors retain every key and the correct `None` versus `[0, 0]` negative shape;
- empty tuple, duplicates, wrong registered anchor endpoint, and missing label topology raise with the offending type.

### Unit: remote fetch API and sharding

Extend `tests/unit/distributed/graph_store/remote_dist_dataset_test.py`. Its existing cases already cover singleton
fetches, split selection, label caps, defaults, mismatched optional arguments, and exact sharding requests
(`tests/unit/distributed/graph_store/remote_dist_dataset_test.py:440-590`, `:651-735`, `:1029-1277`).

Add:

- scalar input produces a one-item request tuple and unchanged output;
- canonical incoming types are reversed before request construction, while the current scalar legacy incoming form
  remains unchanged;
- mixed canonical/legacy incoming lists fail before dispatch;
- two types produce one request per assigned server, not one request per type/server pair;
- exact key order and exact positive/negative values for every server;
- dense empty placeholders preserve both positive keys and topology-correct negative tensors;
- empty list, duplicates, inconsistent anchor endpoint, missing type, and homogeneous multi-type errors;
- `edge_dir="out"` keeps canonical keys; `edge_dir="in"` proves both canonical translation and legacy compatibility;
- fractional server slicing applies identical row selection to every type.

### Unit: GraphStore loader conversion and output

Extend `tests/unit/distributed/dist_ablp_neighborloader_test.py` with tests around the extracted pure GraphStore setup
helper:

- consistent two-type inputs become one `ABLPNodeSamplerInput` per server;
- mixed negative availability produces the exact positive and negative edge-type sets;
- empty unassigned servers receive schema-correct empty tensors;
- mismatched anchor types, key sets, out-of-range ranks, dimensions, integer dtypes, row counts, and negative topology
  fail before worker startup;
- sparse in-range server mappings remain valid;
- labeled-homogeneous still rejects multiple types.

Run shared loader/collation cases for each direction and for both `use_edge_index_output` modes. Assert:

- both final `y_positive` keys and exact values;
- hard negatives on both types;
- the mixed case where only one type has hard negatives remains a one-key `y_negative` dictionary rather than a bare
  value;
- incoming registered keys are reversed back to the exact canonical outward keys.

Also fix the colocated multi-edge helper so the negative assertion is inside a loop over every negative edge type. It
currently loops over positives but checks only the final `edge_type` for negatives
(`tests/unit/distributed/dist_ablp_neighborloader_test.py:347-366`), despite its fixture declaring negative expectations
for both types (`tests/unit/distributed/dist_ablp_neighborloader_test.py:1152-1193`).

### Integration: real GraphStore RPC and sampler

Extend `tests/integration/distributed/graph_store/graph_store_integration_test.py`. The current test checks singleton
remote input and loader startup (`tests/integration/distributed/graph_store/graph_store_integration_test.py:252-325`);
the heterogeneous loader test is skipped and exercises only a neighbor loader
(`tests/integration/distributed/graph_store/graph_store_integration_test.py:1023-1049`).

Do not rely on the current mocked-asset model: it stores only one `sample_edge_type`, and frozen config generation emits
a singleton supervision list (`gigl/src/mocking/lib/mocked_dataset_resources.py:142-151`,
`gigl/src/mocking/dataset_asset_mocker.py:316-327`). Instead:

1. Generalize the test-only `create_heterogeneous_dataset_for_ablp()` builder to accept per-edge-type positive/negative
   maps while retaining its scalar convenience form (`tests/test_assets/distributed/test_dataset.py:236-300`).
2. Add a top-level, multiprocessing-picklable builder for a tiny two-type dataset with one shared anchor type.
3. Add an optional test-only dataset-builder field to `ServerProcessArgs`; after initializing the storage process group,
   `_run_storage_main_process()` uses it instead of `build_storage_dataset()`. The current harness otherwise always
   builds from a task-config URI (`tests/integration/distributed/graph_store/graph_store_integration_test.py:709-728`,
   `:761-767`).

Use the existing incoming GraphStore direction for the real-RPC acceptance path. Do not reuse the skipped heterogeneous
neighbor-loader case, whose Cloud Build failure is unresolved
(`tests/integration/distributed/graph_store/graph_store_integration_test.py:1023-1049`). Through real RPC:

1. Fetch the same split with two canonical outward task types.
2. Assert each storage response has the exact anchors, exact two registered-key label schema, and exact padded
   positive/negative rows.
3. Start `DistABLPLoader` with `use_edge_index_output=True`, consume at least one batch, and assert both canonical
   outward output keys and their exact global positive/negative pairs, including negatives for only one type.
4. Exercise a sharded compute rank with an unassigned storage server.

Keep the existing singleton and homogeneous integration cases as regression coverage. Do not treat the current
multi-type dataset-storage test alone as sufficient: its supervision types have different anchor types
(`tests/integration/distributed/distributed_dataset_test.py:155-191`), so they cannot feed one ABLP loader under the
shared-anchor contract.

## Verification commands

Run targeted tests first:

```bash
.venv/bin/python -m unittest \
  tests.unit.distributed.graph_store.messages_test \
  tests.unit.distributed.graph_store.remote_dist_dataset_test \
  tests.unit.distributed.dist_server_test \
  tests.unit.distributed.dist_ablp_neighborloader_test
```

Then run repository gates and the GraphStore integration target:

```bash
make type_check
make check_format
make integration_test PY_TEST_FILES="graph_store_integration_test.py"
```

Finally run the full unit suite through `make unit_test_py` once the current worktree's unrelated dev-utils import is
repaired or removed.

## Rollout risks and mitigations

| Risk                                                                                    | Mitigation                                                                                                                                                         |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Incoming-edge orientation is interpreted differently by callers                         | Prefer canonical anchor-outward input, translate to registered orientation, retain the uniform legacy incoming form, reject mixed forms, and pin exact final keys. |
| A malformed or stale server response reaches sampling workers                           | Validate every rank's anchor type, key set, dimensions, integer dtype, and topology before collective worker setup; repeat request validation server-side.         |
| A type without hard negatives inherits another type's negative schema                   | Derive negative availability independently with `select_label_edge_types()` and test mixed availability.                                                           |
| Mixed negative output loses its only relation key                                       | Flatten only for a single-supervision loader; keep multi-supervision outputs keyed.                                                                                |
| Sharded ranks lose per-server alignment                                                 | Keep remote fetch dense with topology-complete placeholders while preserving the loader's existing sparse-input support.                                           |
| RPC count grows with the number of supervision types                                    | Batch all types in the one existing request to each selected server.                                                                                               |
| Integration coverage depends on singleton mocked assets or a skipped heterogeneous test | Inject a tiny programmatic multi-type dataset through the test harness.                                                                                            |
| Example users infer that multi-objective training is now supported                      | Keep the example trainer limitation explicit and track it as a separate follow-up.                                                                                 |

## Acceptance criteria

1. The old incoming scalar GraphStore call remains accepted and yields the same one-key `ABLPInputNodes` and final
   tensor-shaped `y_positive`/`y_negative` behavior.
2. A canonical anchor-outward list taken from task metadata is translated to registered incoming orientation and fetched
   in one RPC per selected storage server; mixed canonical/legacy lists are rejected.
3. Every server response and every `ABLPNodeSamplerInput` carries both positive label types; negative label keys match
   registered topology per type.
4. A real-RPC incoming GraphStore `DistABLPLoader` batch exposes both canonical outward output keys with correct global
   label pairs. Direct server/client and shared collation tests cover `edge_dir="out"` as well.
5. Rank/world-size sharding preserves identical anchor-row selection across all requested label tensors; remote fetch
   remains dense and manual sparse loader input remains valid.
6. Empty lists, duplicates, inconsistent anchor endpoints, inconsistent per-server schemas, malformed tensor
   dimensions/dtypes, and labeled-homogeneous multi-type requests fail with actionable errors before collective
   sampling-worker setup.
7. With multiple supervision types and negatives for only one, `y_positive` and `y_negative` remain edge-keyed;
   singleton loaders retain the existing flattened API.
8. Targeted unit tests, type checking, formatting, and the GraphStore integration test pass.

## Baseline verification before implementation

- `tests.unit.distributed.graph_store.remote_dist_dataset_test` passes directly.
- `tests.unit.distributed.dist_ablp_neighborloader_test` passes directly: 21 tests ran in 1001.292 seconds with one
  skip. This exercises the existing colocated multi-edge cases at
  `tests/unit/distributed/dist_ablp_neighborloader_test.py:1095-1332`.
- The normal `make unit_test_py` entry point is currently blocked before tests by an unrelated untracked file,
  `gigl/utils/dev/tb_smoke_main.py:24`, importing a missing `gigl.utils.tensorboard_writer`. This is a worktree baseline
  issue, not part of the GraphStore plan.

## Dual-review resolution

The initial draft received two independent read-only reviews:

- SOL: `.claude/tmp/codex-verify/20260730-graph-store-multi-edge-plan/review.md`
- Claude Fable: `.claude/tmp/claude-fable-review/20260730-graph-store-multi-edge-plan/review.md`

Both reviewers found the tracked server-test omission and placeholder ambiguity
(`.claude/tmp/codex-verify/20260730-graph-store-multi-edge-plan/review.md:124-153`,
`.claude/tmp/claude-fable-review/20260730-graph-store-multi-edge-plan/review.md:125-168`). The revised plan now updates
`tests/unit/distributed/dist_server_test.py` unconditionally and creates topology-complete placeholders.

SOL identified three blocking design gaps: canonical incoming task types were not accepted, mixed negatives lost their
output key, and the proposed integration fixture was not constructible with singleton mocked assets
(`.claude/tmp/codex-verify/20260730-graph-store-multi-edge-plan/review.md:64-123`). The revised plan adopts
canonical-outward translation with legacy compatibility, changes flattening semantics for multi-type loaders, and
specifies a concrete programmatic integration-dataset hook.

The plan also adopts the reviewers' non-blocking hardening findings: strict dimension/dtype validation, server-side
request validation, sparse loader input compatibility, validation before collective initialization, exact incoming key
mapping, and a test for mixed negatives through final collation
(`.claude/tmp/codex-verify/20260730-graph-store-multi-edge-plan/review.md:155-237`,
`.claude/tmp/claude-fable-review/20260730-graph-store-multi-edge-plan/review.md:171-236`).
