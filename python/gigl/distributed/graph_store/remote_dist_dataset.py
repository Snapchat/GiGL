import time
from multiprocessing.managers import DictProxy
from typing import Literal, Optional, Union

import torch
from graphlearn_torch.distributed import async_request_server, request_server

from gigl.common.logger import Logger
from gigl.distributed.graph_store.storage_utils import (
    get_edge_dir,
    get_edge_feature_info,
    get_node_feature_info,
    get_node_ids_for_rank,
)
from gigl.distributed.utils.networking import get_free_ports
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import FeatureInfo

logger = Logger()


class RemoteDistDataset:
    def __init__(
        self,
        cluster_info: GraphStoreInfo,
        local_rank: int,
        mp_sharing_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        """
        Represents a dataset that is stored on a difference storage cluster.
        *Must* be used in the GiGL graph-store distributed setup.

        This class *must* be used on the compute (client) side of the graph-store distributed setup.

        Args:
            cluster_info (GraphStoreInfo): The cluster information.
            local_rank (int): The local rank of the process on the compute node.
            mp_sharing_dict (Optional[dict[str, torch.Tensor]]):
                (Optional) If provided, will be used to share tensors across the local machine.
                e.g. for `get_node_ids`.
                If provided, *must* be a `DictProxy` e.g. the return value of a mp.Manager.
                ex. torch.multiprocessing.Manager().dict().
        """
        self._cluster_info = cluster_info
        self._local_rank = local_rank
        self._mp_sharing_dict = mp_sharing_dict
        if self._mp_sharing_dict is not None and not isinstance(
            self._mp_sharing_dict, DictProxy
        ):
            raise ValueError(
                f"When using mp_sharing_dict, you must pass in a `DictProxy` e.g. mp.manager().dict(). Recieved a {type(self._mp_sharing_dict)}"
            )

    @property
    def cluster_info(self) -> GraphStoreInfo:
        return self._cluster_info

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
            0,
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
            0,
            get_edge_feature_info,
        )

    def get_edge_dir(self) -> Union[str, Literal["in", "out"]]:
        """Get the edge direction from the registered dataset.

        Returns:
            The edge direction.
        """
        return request_server(
            0,
            get_edge_dir,
        )

    def _get_node_ids(self, node_type: Optional[NodeType] = None) -> list[torch.Tensor]:
        """Fetches node ids from the storage nodes for the current compute node (machine)."""
        futures: list[torch.futures.Future[torch.Tensor]] = []
        rank = self.cluster_info.compute_node_rank
        world_size = self.cluster_info.num_storage_nodes
        print(
            f"Getting node ids for rank {rank} / {world_size} with node type {node_type}"
        )

        for server_rank in range(self.cluster_info.num_storage_nodes):
            futures.append(
                async_request_server(
                    server_rank,
                    get_node_ids_for_rank,
                    rank=rank,
                    world_size=world_size,
                    node_type=node_type,
                )
            )
            node_ids = torch.futures.wait_all(futures)
        return node_ids

    def get_node_ids(
        self,
        node_type: Optional[NodeType] = None,
    ) -> list[torch.Tensor]:
        """
        Fetches node ids from the storage nodes for the current compute node (machine).

        The returned list are the node ids for the current compute node, by storage rank.

        For example, if there are two storage ranks, and two compute ranks, and 16 total nodes,
        In this scenario, the node ids are sharded as follows:
        Storage rank 0: [0, 1, 2, 3, 4, 5, 6, 7]
        Storage rank 1: [8, 9, 10, 11, 12, 13, 14, 15]

        NOTE: The GLT sampling enginer expects that all processes on a given compute machine
        to have the same sampling input (node ids).
        As such, the input tensors may be duplicated across all processes on a given compute machine.
        In order to save on cpu memory, pass in `mp_sharing_dict` to the `RemoteDistDataset` constructor.

        Then, for compute rank 0 (node 0, process 0), the returned list will be:
            [
                [0, 1, 3, 4], # From storage rank 0
                [8, 9, 10, 11] # From storage rank 1
            ]

        Args:
            node_type (Optional[NodeType]): The type of nodes to get.
            Must be provided for heterogeneous datasets.

        Returns:
            list[torch.Tensor]: A list of node IDs for the given node type.
        """

        def server_key(server_rank: int) -> str:
            return f"node_ids_from_server_{server_rank}"

        if self._mp_sharing_dict is not None:
            if self._local_rank == 0:
                start_time = time.time()
                print(
                    f"Compute rank {torch.distributed.get_rank()} is getting node ids from storage nodes"
                )
                node_ids = self._get_node_ids(node_type)
                for server_rank, node_id in enumerate(node_ids):
                    node_id.share_memory_()
                    self._mp_sharing_dict[server_key(server_rank)] = node_id
                print(
                    f"Compute rank {torch.distributed.get_rank()} got node ids from storage nodes in {time.time() - start_time:.2f} seconds"
                )
            torch.distributed.barrier()
            node_ids = [
                self._mp_sharing_dict[server_key(server_rank)]
                for server_rank in range(self.cluster_info.num_storage_nodes)
            ]
            print(f"node_ids: {[node.shape for node in node_ids]}")
            return node_ids
        else:
            return self._get_node_ids(node_type)

    def get_free_ports_on_storage_cluster(self, num_ports: int) -> list[int]:
        """
        Get free ports from the storage master node.

        This *must* be used with a torch.distributed process group initialized, for the *entire* training cluster.

        All compute ranks will receive the same free ports.

        Args:
            num_ports (int): Number of free ports to get.
        """
        if not torch.distributed.is_initialized():
            raise ValueError(
                "torch.distributed process group must be initialized for the entire training cluster"
            )
        compute_cluster_rank = torch.distributed.get_rank()
        if compute_cluster_rank == 0:
            ports = request_server(
                0,
                get_free_ports,
                num_ports=num_ports,
            )
            logger.info(
                f"Compute rank {compute_cluster_rank} found free ports: {ports}"
            )
        else:
            ports = [None] * num_ports
        torch.distributed.broadcast_object_list(ports, src=0)
        logger.info(f"Compute rank {compute_cluster_rank} received free ports: {ports}")
        return ports
