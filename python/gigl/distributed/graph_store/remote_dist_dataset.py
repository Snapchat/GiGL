from typing import Literal, Optional, Union

import torch
from graphlearn_torch.distributed import async_request_server, request_server

from gigl.common.logger import Logger
from gigl.distributed.graph_store.remote_dataset import (
    get_edge_dir,
    get_edge_feature_info,
    get_node_feature_info,
    get_node_ids_for_rank,
)
from gigl.distributed.utils.networking import get_free_ports_from_master_node
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import FeatureInfo

logger = Logger()


class RemoteDistDataset:
    def __init__(self, local_rank: int, cluster_info: GraphStoreInfo):
        """
        Represents a dataset that is stored on a difference storage cluster.
        *Must* be used in the GiGL graph-store distributed setup.

        Args:
            local_rank (int): The local rank of the process.
            cluster_info (GraphStoreInfo): The cluster information.
        """
        self._local_rank = local_rank
        self._cluster_info = cluster_info

    def _get_storage_shard(self) -> int:
        """
        Sharded server rank for the current process.
        We do this so we don't overload a specific server.

        Returns:
            int: The sharded server rank.
        """
        return (
            self._cluster_info.compute_cluster_rank(self._local_rank)
            % self._cluster_info.num_storage_nodes
        )

    @property
    def cluster_info(self) -> GraphStoreInfo:
        return self._cluster_info

    @property
    def local_rank(self) -> int:
        return self._local_rank

    def get_node_feature_info(
        self,
    ) -> Union[FeatureInfo, dict[NodeType, FeatureInfo], None]:
        """Get node feature information from the registered dataset.

        Returns:
            Node feature information, which can be:
            - A single FeatureInfo object for homogeneous graphs
            - A dict mapping NodeType to FeatureInfo for heterogeneous graphs
            - None if no node features are available
        """
        return request_server(
            self._get_storage_shard(),
            get_node_feature_info,
        )

    def get_edge_feature_info(
        self,
    ) -> Union[FeatureInfo, dict[EdgeType, FeatureInfo], None]:
        """Get edge feature information from the registered dataset.

        Returns:
            Edge feature information, which can be:
            - A single FeatureInfo object for homogeneous graphs
            - A dict mapping EdgeType to FeatureInfo for heterogeneous graphs
            - None if no edge features are available
        """
        return request_server(
            self._get_storage_shard(),
            get_edge_feature_info,
        )

    def get_edge_dir(self) -> Union[str, Literal["in", "out"]]:
        """Get the edge direction from the registered dataset.

        Returns:
            The edge direction.
        """
        return request_server(
            self._get_storage_shard(),
            get_edge_dir,
        )

    def get_node_ids(
        self,
        node_type: Optional[NodeType] = None,
    ) -> list[torch.Tensor]:
        """
        Fetches node ids from the storage nodes for the current compute rank.

        The returned list are the node ids for each storage rank, by storage rank.

        For example, if there are two storage ranks, and four compute ranks, and 16 total nodes,
        In this scenario, the node ids are sharded as follows:
        Storage rank 0: [0, 1, 2, 3, 4, 5, 6, 7]
        Storage rank 1: [8, 9, 10, 11, 12, 13, 14, 15]

        Then, for compute rank 0 (node 0, process 0), the returned list will be:
            [
                [0, 1], # From storage rank 0
                [8, 9] # From storage rank 1
            ]

        Args:
            node_type (Optional[NodeType]): The type of nodes to get.

        Returns:
            list[torch.Tensor]: A list of node IDs for the given node type.
        """
        futures: list[torch.futures.Future[torch.Tensor]] = []
        rank = (
            self.cluster_info.compute_node_rank
            * self.cluster_info.num_processes_per_compute
            + self._local_rank
        )
        logger.info(
            f"Getting node ids for rank {rank} / {self.cluster_info.compute_cluster_world_size} with node type {node_type}"
        )
        for server_rank in range(self.cluster_info.num_storage_nodes):
            futures.append(
                async_request_server(
                    server_rank,
                    get_node_ids_for_rank,
                    rank=rank,
                    world_size=self.cluster_info.compute_cluster_world_size,
                    node_type=node_type,
                )
            )
        node_ids = torch.futures.wait_all(futures)
        return node_ids

    def get_free_ports(self, num_ports: int) -> list[int]:
        """
        Get free ports from the storage master node for the current compute rank.

        Args:
            num_ports (int): Number of free ports to get.
        """
        return request_server(
            0,  # We use the master node for the free ports
            get_free_ports_from_master_node,
            num_ports=num_ports,
        )
