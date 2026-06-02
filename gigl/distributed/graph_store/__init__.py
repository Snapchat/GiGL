"""
Public API for GiGL's graph-store deployment mode.

Graph-store mode separates storage and compute clusters: storage nodes build and serve
a partitioned dataset, while compute nodes connect over RPC via a ``RemoteDistDataset``.

This module is the stable import surface for that workflow; helpers, RPC utilities, and
server lifecycle internals remain in private submodules.
"""

__all__ = [
    "GraphStoreInfo",
    "RemoteDistDataset",
    "build_storage_dataset",
    "get_graph_store_info",
    "init_compute_process",
    "run_storage_server",
    "shutdown_compute_process",
]

from gigl.distributed.utils import GraphStoreInfo, get_graph_store_info

from .compute import init_compute_process, shutdown_compute_process
from .remote_dist_dataset import RemoteDistDataset
from .storage_utils import build_storage_dataset, run_storage_server
