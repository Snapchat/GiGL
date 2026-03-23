
# Edge Direction and Supervision Edge Types for Link Prediction

This document explains how the `edge_dir` parameter interacts with supervision edge types for the link prediction task.

## What `edge_dir` means

In GLT / GiGL, a distributed graph stores message passing edges in one canonical direction
and `edge_dir` tells the sampler which direction to walk them:

- `"out"` — standard forward traversal: neighbours of node `v` are the nodes that `v`
  points **to** (i.e. `v → neighbour`).
- `"in"` — reverse traversal: neighbours of node `v` are the nodes that point **into**
  `v` (i.e. `neighbour → v`, read as "incoming edges of `v`").

## Choosing `edge_dir`

The right choice depends on the semantics of your edges and which direction of traversal
gives meaningful neighbours for your anchor nodes.

Consider a social graph where edges represent "A follows B" (A→B), and you are predicting
future follow edges from users:

**`edge_dir="out"` (default)** — the sampler walks edges in their stored direction.
Starting from User A, hop 1 finds A's followees (nodes A points to); hop 2 finds their
followees; and so on. This builds a representation of A based on *who A follows and what
those accounts do*.

```
User A ──follows──► User B ──follows──► User D
       ──follows──► User C ──follows──► User E
```

**`edge_dir="in"`** — the sampler walks edges in reverse (against their stored direction).
Starting from User A, hop 1 finds A's followers (nodes that point to A); hop 2 finds their
followers. This builds a representation of A based on *who follows A and what those
accounts do*.

```
User X ──follows──► User A ◄──follows── User Y
User Z ──follows──► User X
```

Generally, use `"out"` when the signal flows forward along your edges (e.g. a user's taste is defined
by what they follow). Use `"in"` when the signal flows backward (e.g. a user's influence
is defined by who follows them).

---

## Two types of edges

Additionally, there are two types of edges which GiGL distinguishes for the link prediction task:

- **Message passing edges** — the graph topology used for GNN neighbourhood aggregation. These are the edges in `edge_indices` (e.g. `USER→ITEM`, `ITEM→USER`). `edge_dir` controls which direction the sampler walks these edges during training and inference.

- **Supervision edges** — the positive and negative edges used as the training signal.
  They describe *what the model is trying to predict*, not the graph topology.
  **The supervision edge types must always be supplied in the outgoing (src→dst) direction,
  regardless of `edge_dir`.**

---

## Supplying Supervision Edges

GiGL supports two paradigms for providing supervision edges.

### Self-supervised labels

GiGL derives positive supervision labels from the message passing edges at dataset build time. A random subset — controlled by `_ssl_positive_label_percentage` — is sampled as positive labels. No separate label tensors are needed.

You still must pass `supervision_edge_type` in the **outgoing direction** to both `DistNodeAnchorLinkSplitter` and `DistABLPLoader`.

```python
supervision_edge_type = EdgeType(USER, Relation("to"), ITEM)  # always outgoing

splitter = DistNodeAnchorLinkSplitter(
    sampling_direction=edge_dir,
    supervision_edge_types=[supervision_edge_type],
)
dataset = build_dataset(
    serialized_graph_metadata=...,
    sample_edge_direction=edge_dir,
    splitter=splitter,
    _ssl_positive_label_percentage=0.05,  # 5% of message passing edges used as positive labels
)
loader = DistABLPLoader(
    dataset=dataset,
    supervision_edge_type=supervision_edge_type,  # outgoing, regardless of edge_dir
    ...
)
```

### Explicitly provided labels

Users register positive and optionally negative label edges in `DataPreprocessorConfig.get_edges_preprocessing_spec()` using `EdgeUsageType.POSITIVE` and `EdgeUsageType.NEGATIVE`. These are written as TFRecords by the data preprocessor and loaded automatically when the dataset is built — no extra loading code is needed.

Label edges must be in the **outgoing (src→dst) direction**. GiGL handles any internal reversal required for `edge_dir="in"` automatically (see [What happens internally](#what-happens-internally)).

```python
# In DataPreprocessorConfig.get_edges_preprocessing_spec():
def get_edges_preprocessing_spec(self):
    supervision_edge_type = EdgeType(USER, Relation("to"), ITEM)  # outgoing direction

    positive_edge_ref = BigqueryEdgeDataReference(
        reference_uri=self.positive_edges_table,
        edge_type=supervision_edge_type,
        edge_usage_type=EdgeUsageType.POSITIVE,
    )
    negative_edge_ref = BigqueryEdgeDataReference(  # optional
        reference_uri=self.negative_edges_table,
        edge_type=supervision_edge_type,
        edge_usage_type=EdgeUsageType.NEGATIVE,
    )
    ...

# At training time, labels are loaded automatically from TFRecords:
supervision_edge_type = EdgeType(USER, Relation("to"), ITEM)  # always outgoing

splitter = DistNodeAnchorLinkSplitter(
    sampling_direction=edge_dir,
    supervision_edge_types=[supervision_edge_type],
)
dataset = build_dataset(
    serialized_graph_metadata=...,
    sample_edge_direction=edge_dir,
    splitter=splitter,
    # no _ssl_positive_label_percentage — labels are loaded from TFRecords
)
loader = DistABLPLoader(
    dataset=dataset,
    supervision_edge_type=supervision_edge_type,  # outgoing, regardless of edge_dir
    ...
)
```

---

## What happens internally

This section describes how both supervision edges and message passing edges are stored in a `DistDataset`. This is purely internal detail — you do not need to implement this yourself, but it is useful context when reading the code or debugging.

There are two principles which our internal codebase currently operates under:
- Both supervision and message passing edges are stored in a single `dict[EdgeType, Tensor]` edge index object. Supervision edges are injected into this dict under derived edge types using a `to_gigl_positive` or `to_gigl_negative` relation (e.g. `("user", "to_gigl_positive", "item")`), keeping them distinct from the message passing edge type keys.
- All edges stored in the `DistDataset` must share the same edge direction (either both `"in"` or both `"out"`).

Since supervision edges are always provided in the outgoing direction, but `edge_dir="in"` requires all stored edges to point inward, we reverse the supervision edges before storing them alongside the message passing edges. When `edge_dir="out"`, no reversal is needed since both are already outgoing. The reversal (both the edge type and the COO row flip) is implemented in `_get_label_edges` (`gigl/types/graph.py:110`). `DistNodeAnchorLinkSplitter` (`gigl/utils/data_splitters.py:250`) and `DistABLPLoader` (`gigl/distributed/dist_ablp_neighborloader.py:462`) independently apply the same edge type reversal to derive the correct lookup key when accessing supervision edges from the stored graph.

For example, if a user provides `supervision_edge_type = ("user", "to", "item")` with `edge_dir="in"`, we internally reverse the supervision edge type so that all edges can be stored in the same edge index object. The dataset will store an injected `("item", "to_gigl_positive", "user")` edge type, with the edge index rows flipped accordingly.

### `DistNodeAnchorLinkSplitter` and `edge_dir`

The splitter reads anchor nodes directly from the stored supervision edge index. It uses `edge_dir` to determine two things:

- **Which edge type to look up** — it derives the stored key the same way as storage (reversing for `edge_dir="in"`), so the lookup always matches.
- **Which end of the supervision edge holds the anchor nodes** — `edge_dir` determines which endpoint the splitter reads anchor nodes from: the source end for `"out"`, the destination end for `"in"`, since the stored edge type is reversed for `"in"` and the anchor node type shifts accordingly. For example, with `supervision_edge_type = ("user", "to", "item")`: `edge_dir="out"` reads user IDs from the source end of `("user", "to_gigl_positive", "item")`; `edge_dir="in"` reads user IDs from the destination end of `("item", "to_gigl_positive", "user")`.

The invariant: anchor nodes (users in the above example) are always recoverable and used for splitting regardless of `edge_dir`.

---

### Worked example

Consider a USER→ITEM graph. A user wants to predict which items users will interact with
next and provides:

```python
supervision_edge_type = EdgeType(USER, Relation("to"), ITEM)

positive_labels = {
    supervision_edge_type: torch.tensor([
        [0, 0, 1, 1, 2, 2 ],  # src: user IDs
        [10,11,11,12,12,13],  # dst: item IDs
    ])
}
negative_labels = {
    supervision_edge_type: torch.tensor([
        [0,  1,  2 ],  # src: user IDs
        [12, 13, 10],  # dst: item IDs
    ])
}
```

These inputs are **identical regardless of `edge_dir`**. What changes is how they are
stored internally in the `DistDataset`.

**`edge_dir="out"`** — supervision and message passing edges already share the same
direction, so no reversal is needed:

```
  User 0 ──pos──► Item 10
         ──pos──► Item 11
         ··neg··► Item 12

  User 1 ──pos──► Item 11
         ──pos──► Item 12
         ··neg··► Item 13

  User 2 ──pos──► Item 12
         ──pos──► Item 13
         ··neg··► Item 10

  injected positive edge type  = ("user", "to_gigl_positive", "item")
  injected positive edge index = [[0,  0,  1,  1,  2,  2 ],   ← user IDs  (row 0 = src)
                                   [10, 11, 11, 12, 12, 13]]   ← item IDs  (row 1 = dst)

  injected negative edge type  = ("user", "to_gigl_negative", "item")
  injected negative edge index = [[0,  1,  2 ],   ← user IDs  (row 0)
                                   [12, 13, 10]]   ← item IDs  (row 1)
```

**`edge_dir="in"`** — supervision edges are reversed before storage so they match the
inward orientation of the message passing edges:

```
  Item 10 ──pos──► User 0
          ··neg··► User 2

  Item 11 ──pos──► User 0
          ──pos──► User 1

  Item 12 ──pos──► User 1
          ──pos──► User 2
          ··neg··► User 0

  Item 13 ──pos──► User 2
          ··neg··► User 1

  injected positive edge type  = ("item", "to_gigl_positive", "user")
  injected positive edge index = [[10, 11, 11, 12, 12, 13],   ← item IDs  (row 0 = src)
                                    [0,  0,  1,  1,  2,  2 ]]  ← user IDs  (row 1 = dst)

  injected negative edge type  = ("item", "to_gigl_negative", "user")
  injected negative edge index = [[12, 13, 10],   ← item IDs  (row 0)
                                    [0,  1,  2 ]]  ← user IDs  (row 1)
```

In both cases the user IDs (anchor nodes) are recoverable from the stored edge index, and
the user-provided `supervision_edge_type` and edge index tensors are unchanged.
