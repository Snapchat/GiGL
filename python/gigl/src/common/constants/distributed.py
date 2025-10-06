"""Constants for distributed workloads."""
from typing import Final

# The env vars where the ranks of the leader workers are stored for the storage and compute clusters
# Only applicable in multipool workloads.
STORAGE_CLUSTER_MASTER_KEY: Final[str] = "GIGL_STORAGE_CLUSTER_MASTER_RANK"
COMPUTE_CLUSTER_MASTER_KEY: Final[str] = "GIGL_COMPUTE_CLUSTER_MASTER_RANK"

STORAGE_CLUSTER_NUM_NODES_KEY: Final[str] = "GIGL_STORAGE_CLUSTER_NUM_NODES"
COMPUTE_CLUSTER_NUM_NODES_KEY: Final[str] = "GIGL_COMPUTE_CLUSTER_NUM_NODES"
