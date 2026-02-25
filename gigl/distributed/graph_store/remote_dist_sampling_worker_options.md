# `RemoteDistSamplingWorkerOptions` Deep Dive

## Class Definition

Defined in the installed GLT package at:
`graphlearn_torch/distributed/dist_options.py` (lines 210-291)

It extends `_BasicDistSamplingWorkerOptions` (lines 26-117) and is designed for **Graph Store (server-client) mode**,
where sampling workers run on remote storage servers and results are sent back to compute nodes.

## All Fields/Knobs

### Inherited from `_BasicDistSamplingWorkerOptions` (lines 26-117)

| Field | Type | Default | Description |
|---|---|---|---|
| `num_workers` | `int` | `1` | Number of sampling worker subprocesses to launch on the server for this client |
| `worker_devices` | `list[torch.device] \| None` | `None` | Device assignment per worker; auto-assigned if `None` |
| `worker_concurrency` | `int` | `4` | Max concurrent seed batches each worker processes simultaneously (clamped to [1, 32]) |
| `master_addr` | `str` | env `MASTER_ADDR` | Master address for RPC init of the sampling worker group |
| `master_port` | `int` | env `MASTER_PORT` + 1 | Master port for RPC init of the sampling worker group |
| `num_rpc_threads` | `int \| None` | `None` | RPC threads per sampling worker; auto-set to `min(num_partitions, 16)` if `None` |
| `rpc_timeout` | `float` | `180` | Timeout (seconds) for all RPC requests during sampling |

### Specific to `RemoteDistSamplingWorkerOptions` (lines 210-291)

| Field | Type | Default | Description |
|---|---|---|---|
| `server_rank` | `int \| list[int]` | auto-assigned | Which storage server(s) to create sampling workers on |
| `buffer_size` | `int \| str` | `"{num_workers * 64}MB"` | Size of server-side shared-memory buffer for sampled messages |
| `buffer_capacity` | computed | `num_workers * worker_concurrency` | Max messages the server-side buffer can hold |
| `prefetch_size` | `int` | `4` | Max prefetched messages on the **client** side (must be <= `buffer_capacity`) |
| `worker_key` | `str \| None` | `None` | Deduplication key -- same key reuses existing producer on server |
| `use_all2all` | `bool` | `False` | Use all2all collective for feature collection instead of point-to-point RPC |
| `glt_graph` | any | `None` | GraphScope only (not used by GiGL) |
| `workload_type` | `str \| None` | `None` | GraphScope only (not used by GiGL) |

## How Each Field Is Used & Client vs. Server

### `server_rank` -- CLIENT side

The client reads this to know which servers to talk to:

- `dist_loader.py:170-171` -- expanded to `_server_rank_list`
- `dist_loader.py:178` -- `request_server(self._server_rank_list[0], DistServer.get_dataset_meta)` to fetch metadata
- `dist_loader.py:188` -- loops over servers calling `DistServer.create_sampling_producer`
- `dist_loader.py:194` -- passed to `RemoteReceivingChannel` for receiving results
- `dist_loader.py:305-306` -- `request_server(server_rank, DistServer.start_new_epoch_sampling, ...)`

### `num_workers` -- SERVER side

Serialized and sent to the server via RPC. On the server:

- `dist_sampling_producer.py:184` -- `self.num_workers = self.worker_options.num_workers`
- `dist_sampling_producer.py:208` -- spawns that many subprocesses via `mp_context.Process(...)`
- Also drives the defaults for `buffer_size` (line 281) and `buffer_capacity` (line 279)

### `worker_devices` -- SERVER side

- Auto-assigned via `_assign_worker_devices()` (`dist_options.py:113-116`) if `None`
- Used in `_sampling_worker_loop` (line 87): `current_device = worker_options.worker_devices[rank]`

### `worker_concurrency` -- SERVER side

Controls async parallelism within each sampling worker:

- `_sampling_worker_loop:106` -- passed to `DistNeighborSampler(..., concurrency=worker_options.worker_concurrency)`
- In `ConcurrentEventLoop` (`event_loop.py:47`): creates a `BoundedSemaphore(concurrency)` limiting concurrent seed batches
- Also drives `buffer_capacity = num_workers * worker_concurrency` (line 279)

### `master_addr` / `master_port` -- SERVER side

Used by sampling worker subprocesses to form their own RPC group for cross-partition sampling:

- `_sampling_worker_loop:93-98` -- `init_rpc(master_addr=..., master_port=..., ...)`

### `num_rpc_threads` -- SERVER side

- `_sampling_worker_loop:82-85` -- if `None`, auto-set to `min(data.num_partitions, 16)`
- Line 91: `torch.set_num_threads(num_rpc_threads + 1)`
- Line 93: passed to `init_rpc(...)` which sets `TensorPipeRpcBackendOptions.num_worker_threads`

### `rpc_timeout` -- SERVER side

- `_sampling_worker_loop:97` -- `init_rpc(..., rpc_timeout=...)`
- Sets the timeout for RPCs made by sampling workers when fetching graph partitions from other servers

### `buffer_size` -- SERVER side

- GLT `dist_server.py:158`: `ShmChannel(worker_options.buffer_capacity, worker_options.buffer_size)`
- GiGL `dist_server.py:456-457`: same usage
- Controls the total bytes of shared memory allocated for the message queue

### `buffer_capacity` -- SERVER side

- Computed as `num_workers * worker_concurrency` (`dist_options.py:279`)
- Passed as the first arg to `ShmChannel(capacity, size)` -- max messages before producers block

### `prefetch_size` -- CLIENT side

- `dist_loader.py:196` -- `RemoteReceivingChannel(..., prefetch_size)`
- In `remote_channel.py:47`: `self.prefetch_size = prefetch_size`
- Line 56: `queue.Queue(maxsize=self.prefetch_size * len(self.server_rank_list))`
- Lines 120-131: controls how many async RPC fetch requests are in-flight per server at any time

### `worker_key` -- SERVER side (during producer creation)

- GLT `dist_server.py:152`: `producer_id = self._worker_key2producer_id.get(worker_options.worker_key)` -- if already exists, reuses the producer
- GiGL `dist_server.py:444-453`: same pattern with per-producer locks

### `use_all2all` -- SERVER side

- `_sampling_worker_loop:73-80` -- if True, initializes `torch.distributed` process group (gloo backend)
- `dist_neighbor_sampler.py:749-753` -- switches from per-type `async_get()` to `get_all2all()` for feature collection

## Client vs. Server Summary

| Field | Side | Purpose |
|---|---|---|
| `server_rank` | **Client** | Which servers to send RPCs to |
| `num_workers` | **Server** | Sampling subprocesses per server |
| `worker_devices` | **Server** | Device per subprocess |
| `worker_concurrency` | **Server** | Concurrent batches per subprocess |
| `master_addr` / `master_port` | **Server** | RPC group for cross-partition sampling |
| `num_rpc_threads` | **Server** | RPC threads per sampling subprocess |
| `rpc_timeout` | **Server** | Timeout for cross-partition RPCs |
| `buffer_size` | **Server** | Shared-memory buffer bytes |
| `buffer_capacity` | **Server** | Shared-memory buffer message count |
| `prefetch_size` | **Client** | Prefetched messages per server |
| `worker_key` | **Server** | Producer deduplication |
| `use_all2all` | **Server** | Collective vs point-to-point features |

The entire options object is **serialized and sent via RPC** from client to server (at `dist_loader.py:188` via
`DistServer.create_sampling_producer`). The server reads the server-side fields; the client reads `server_rank` and
`prefetch_size` locally.

## How GiGL Uses It

### `DistNeighborLoader._setup_for_graph_store()`

**File:** `gigl/distributed/distributed_neighborloader.py:386-395`

```python
worker_options = RemoteDistSamplingWorkerOptions(
    server_rank=list(range(dataset.cluster_info.num_storage_nodes)),
    num_workers=num_workers,
    worker_devices=[torch.device("cpu") for i in range(num_workers)],
    master_addr=dataset.cluster_info.storage_cluster_master_ip,
    buffer_size=channel_size,        # defaults to "4GB"
    master_port=sampling_port,
    worker_key=worker_key,           # unique per compute rank + loader instance
    prefetch_size=prefetch_size,     # default 4
)
```

GiGL talks to **all** storage servers (`server_rank=list(range(num_storage_nodes))`), always uses **CPU** sampling, and
assigns a unique `worker_key` per compute rank + loader instance (`distributed_neighborloader.py:384`).

Notably, GiGL **bypasses GLT's `DistLoader.__init__`** in `_init_graph_store_connections()` (lines 609-837),
dispatching `create_sampling_producer` RPCs sequentially per compute node to avoid GLT's `ThreadPoolExecutor` deadlock
at large scale.

### `DistABLPLoader._setup_for_graph_store()`

**File:** `gigl/distributed/dist_ablp_neighborloader.py:799-808`

```python
worker_options = RemoteDistSamplingWorkerOptions(
    server_rank=list(range(dataset.cluster_info.num_storage_nodes)),
    num_workers=num_workers,
    worker_devices=[torch.device("cpu") for _ in range(num_workers)],
    worker_concurrency=worker_concurrency,
    master_addr=dataset.cluster_info.storage_cluster_master_ip,
    master_port=sampling_port,
    worker_key=worker_key,
    prefetch_size=prefetch_size,
)
```

Nearly identical, except:

- Explicitly passes `worker_concurrency` (default `4`)
- Does **not** set `buffer_size` (uses GLT default of `num_workers * 64 MB` instead of GiGL's `4GB`)
- Uses `ThreadPoolExecutor` for setup (lines 1002-1015) rather than the sequential barrier approach
