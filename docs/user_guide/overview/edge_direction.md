
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

The right choice depends on which nodes the GNN should aggregate information **from** when
building representations for your anchor nodes.

**Use `edge_dir="out"` (default)** when the anchor node's useful neighbourhood is found
by following edges in their stored direction. For a USER→ITEM graph, sampling outward from
a user walks USER→ITEM edges to reach items, then ITEM→USER edges to reach other users.
This is the standard case.

**Use `edge_dir="in"`** when the anchor node's useful neighbourhood is found by following
edges in the *reverse* direction. For a USER→ITEM graph, sampling inward from a user walks
ITEM→USER edges first (i.e. "which items point to this user?"), then USER→ITEM edges from
those items. This is useful when your graph is directed and you want to aggregate from
nodes that "chose" the anchor rather than nodes the anchor "chose".

A common real-world signal for `edge_dir="in"`: if edges represent a user following an
item (USER→ITEM) but you want the item-side neighbourhood first (co-followed items), use
`edge_dir="in"`. If edges represent an item being clicked by a user and you want the
user-side neighbourhood first (co-clicking users), use `edge_dir="out"`.

---

## Two types of edges

Additionally, there are two types of edges which GiGL distinguishes for the link prediction task:

- **Message passing edges** — the graph topology used for GNN neighbourhood aggregation.
  These are the edges in `edge_indices` (e.g. `USER→ITEM`, `ITEM→USER`). `edge_dir`
  controls which direction the sampler walks these edges during training and inference.

- **Supervision edges** — the positive and negative edges used as the training signal.
  These are expressed as an adjacency list (`{src_id: [dst_id, ...]}`), paired with a
  `supervision_edge_type` that names the relationship. Each entry in the adjacency list
  represents one or more directed edges `src → dst`. They describe *what the model is
  trying to predict*, not the graph topology.

`edge_dir` affects **only** message passing edges. Supervision edges are always expressed
in the outgoing direction regardless of `edge_dir`.

---


## What a user need to provide for Anchor Based Link Prediction (ABLP)

**Supervision edges are always provided in the outgoing direction, regardless of `edge_dir`.**

For a USER→ITEM graph:

```python
# Names the relationship being predicted
supervision_edge_type = EdgeType(USER, Relation("to"), ITEM)   # always USER→ITEM

# Adjacency lists: each key is a src node ID, each value is the list of dst node IDs
# it connects to — i.e. each entry represents one or more src→dst edges
positive_labels = {user_id: [item_id, ...], ...}               # always src→dst
negative_labels = {user_id: [item_id, ...], ...}               # always src→dst
```

Changing `edge_dir` changes how the GNN samples the message passing graph — it does not
change how you express which USER→ITEM edges to predict. The same `supervision_edge_type`
and adjacency lists are correct for both `"in"` and `"out"`.

---

## What happens internally

This section describes how both supervision edges and message passing edges are stored in a `DistDataset`. This is purely internal detail — you do not need to implement this yourself, but it is useful context when reading the code or debugging.

There are two principles which our internal codebase currently operates under:
- Supervision edge types and message passing edge types are stored in the same edge index tensor. This is done by injecting an edge type with a custom `to_gigl_positive` or `to_gigl_negative` relation.
- All edges stored in the `DistDataset` must share the same edge direction (both `"in"` or both `"out"`).

Since supervision edges are always provided in the outgoing direction, but `edge_dir="in"` requires all stored edges to point inward, we reverse the supervision edges before storing them alongside the message passing edges. When `edge_dir="out"`, no reversal is needed since both are already outgoing.

For example, if a user provides `supervision_edge_type = ("user", "to", "item")` with `edge_dir="in"`, we internally reverse the supervision edge type so that all edges can be stored in the same edge index object. The dataset will store an injected `("item", "to_gigl_positive", "user")` edge type, with the edge index rows flipped accordingly.

---

### Worked example

Consider a USER→ITEM graph. A user wants to predict which items users will interact with
next and provides:

```python
supervision_edge_type = EdgeType(USER, Relation("to"), ITEM)

positive_labels = {0: [10, 11], 1: [11, 12], 2: [12, 13]}  # user→item edges to predict
negative_labels = {0: [12],     1: [13],      2: [10]}       # user→item non-edges
```

These inputs are **identical regardless of `edge_dir`**. What changes is how they are
stored internally in the `DistDataset`.

**`edge_dir="out"`** — supervision and message passing edges already share the same
direction, so no reversal is needed:

```
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
injected positive edge type  = ("item", "to_gigl_positive", "user")
injected positive edge index = [[10, 11, 11, 12, 12, 13],   ← item IDs  (row 0 = src)
                                  [0,  0,  1,  1,  2,  2 ]]  ← user IDs  (row 1 = dst)

injected negative edge type  = ("item", "to_gigl_negative", "user")
injected negative edge index = [[12, 13, 10],   ← item IDs  (row 0)
                                  [0,  1,  2 ]]  ← user IDs  (row 1)
```

In both cases the user IDs (anchor nodes) are recoverable from the stored edge index, and
the user-provided `supervision_edge_type` and label adjacency lists are unchanged.
