import abc
from typing import Optional

import torch
from graphlearn_torch.distributed import async_request_server

from gigl.common.logger import Logger
from gigl.distributed.graph_store.remote_dataset import get_node_ids_for_rank
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.graph_data import NodeType

logger = Logger()


class RemoteDataset(abc.ABC):
    """
    Base class for remote datasets.
    """

    @property
    @abc.abstractmethod
    def cluster_info(self) -> GraphStoreInfo:
        """Get the cluster information.

        Returns:
            GraphStoreInfo: The cluster information.
        """

    @abc.abstractmethod
    def get_node_ids(self, node_type: Optional[NodeType] = None) -> list[torch.Tensor]:
        """Get the node IDs for a given node type.

        Args:
            node_type (Optional[NodeType]): The type of nodes to get.
        """


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

    @property
    def cluster_info(self) -> GraphStoreInfo:
        return self._cluster_info

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
