from collections import Counter, abc
from typing import Dict, Optional, Tuple, Union

import torch
from graphlearn_torch.channel import SampleMessage
from graphlearn_torch.distributed import DistLoader, MpDistSamplingWorkerOptions
from graphlearn_torch.sampler import NodeSamplerInput, SamplingConfig, SamplingType
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

import gigl.distributed.utils
from gigl.common.logger import Logger
from gigl.distributed.constants import DEFAULT_MASTER_INFERENCE_PORT
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.distributed.utils.neighborloader import (
    labeled_to_homogeneous,
    patch_fanout_for_sampling,
    shard_nodes_by_process,
    strip_label_edges,
)
from gigl.src.common.types.graph_data import (
    NodeType,  # TODO (mkolodner-sc): Change to use torch_geometric.typing
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
)

logger = Logger()

# When using CPU based inference/training, we default cpu threads for neighborloading on top of the per process parallelism.
DEFAULT_NUM_CPU_THREADS = 2


class DistNeighborLoader(DistLoader):
    def __init__(
        self,
        dataset: DistLinkPredictionDataset,
        num_neighbors: Union[list[int], Dict[EdgeType, list[int]]],
        input_nodes: Optional[
            Union[torch.Tensor, Tuple[NodeType, torch.Tensor]]
        ] = None,
        num_workers: int = 1,
        batch_size: int = 1,
        context: Optional[DistributedContext] = None,  # TODO: (svij) Deprecate this
        local_process_rank: Optional[int] = None,  # TODO: (svij) Deprecate this
        local_process_world_size: Optional[int] = None,  # TODO: (svij) Deprecate this
        pin_memory_device: Optional[torch.device] = None,
        worker_concurrency: int = 4,
        channel_size: str = "4GB",
        process_start_gap_seconds: float = 60.0,
        num_cpu_threads: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """
        Note: We try to adhere to pyg dataloader api as much as possible.
        See the following for reference:
        https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/loader/node_loader.html#NodeLoader
        https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/distributed/dist_neighbor_loader.html#DistNeighborLoader

        Args:
            dataset (DistLinkPredictionDataset): The dataset to sample from.
            num_neighbors (list[int] or Dict[Tuple[str, str, str], list[int]]):
                The number of neighbors to sample for each node in each iteration.
                If an entry is set to `-1`, all neighbors will be included.
                In heterogeneous graphs, may also take in a dictionary denoting
                the amount of neighbors to sample for each individual edge type.
            context (deprecated - will be removed soon) (DistributedContext): Distributed context information of the current process.
            local_process_rank (deprecated - will be removed soon) (int): Required if context provided. The local rank of the current process within a node.
            local_process_world_size (deprecated - will be removed soon)(int): Required if context provided. The total number of processes within a node.
            input_nodes (torch.Tensor or Tuple[str, torch.Tensor]): The
                indices of seed nodes to start sampling from.
                It is of type `torch.LongTensor` for homogeneous graphs.
                If set to `None` for homogeneous settings, all nodes will be considered.
                In heterogeneous graphs, this flag must be passed in as a tuple that holds
                the node type and node indices. (default: `None`)
            num_workers (int): How many workers to use (subprocesses to spwan) for
                    distributed neighbor sampling of the current process. (default: ``1``).
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            pin_memory_device (str, optional): The target device that the sampled
                results should be copied to. If set to ``None``, the device is inferred based off of
                (got by ``gigl.distributed.utils.device.get_available_device``). Which uses the
                local_process_rank and torch.cuda.device_count() to assign the device. If cuda is not available,
                the cpu device will be used. (default: ``None``).
            worker_concurrency (int): The max sampling concurrency for each sampling
                worker. Load testing has showed that setting worker_concurrency to 4 yields the best performance
                for sampling. Although, you may whish to explore higher/lower settings when performance tuning.
                (default: `4`).
            channel_size (int or str): The shared-memory buffer size (bytes) allocated
                for the channel. Can be modified for performance tuning; a good starting point is: ``num_workers * 64MB``
                (default: "4GB").
            process_start_gap_seconds (float): Delay between each process for initializing neighbor loader. At large scales,
                it is recommended to set this value to be between 60 and 120 seconds -- otherwise multiple processes may
                attempt to initialize dataloaders at overlapping times, which can cause CPU memory OOM.
            num_cpu_threads (Optional[int]): Number of cpu threads PyTorch should use for CPU training/inference
                neighbor loading; on top of the per process parallelism.
                Defaults to `2` if set to `None` when using cpu training/inference.
            shuffle (bool): Whether to shuffle the input nodes. (default: ``False``).
            drop_last (bool): Whether to drop the last incomplete batch. (default: ``False``).
        """

        # Set self._shutdowned right away, that way if we throw here, and __del__ is called,
        # then we can properly clean up and don't get extraneous error messages.
        # We set to `True` as we don't need to cleanup right away, and this will get set
        # to `False` in super().__init__()` e.g.
        # https://github.com/alibaba/graphlearn-for-pytorch/blob/26fe3d4e050b081bc51a79dc9547f244f5d314da/graphlearn_torch/python/distributed/dist_loader.py#L125C1-L126C1
        self._shutdowned = True

        node_world_size: int
        node_rank: int
        rank: int
        world_size: int
        local_rank: int
        local_world_size: int

        master_ip_address: str
        should_cleanup_distributed_context: bool = False

        if context:
            assert (
                local_process_world_size is not None
            ), "context: DistributedContext provided, so local_process_world_size must be provided."
            assert (
                local_process_rank is not None
            ), "context: DistributedContext provided, so local_process_rank must be provided."

            master_ip_address = context.main_worker_ip_address
            node_world_size = context.global_world_size
            node_rank = context.global_rank
            local_world_size = local_process_world_size
            local_rank = local_process_rank

            rank = node_rank * local_world_size + local_rank
            world_size = node_world_size * local_world_size

            if not torch.distributed.is_initialized():
                logger.info(
                    "process group is not available, trying to torch.distributed.init_process_group to communicate necessary setup information."
                )
                should_cleanup_distributed_context = True
                logger.info(
                    f"Initializing process group with master ip address: {master_ip_address}, rank: {rank}, world size: {world_size}, local_rank: {local_rank}, local_world_size: {local_world_size}."
                )
                torch.distributed.init_process_group(
                    backend="gloo",  # We just default to gloo for this temporary process group
                    init_method=f"tcp://{master_ip_address}:{DEFAULT_MASTER_INFERENCE_PORT}",
                    rank=rank,
                    world_size=world_size,
                )

        else:
            assert (
                torch.distributed.is_initialized()
            ), f"context: DistributedContext is None, so process group must be initialized before constructing this object {self.__class__.__name__}."
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            rank_ip_addresses = gigl.distributed.utils.get_internal_ip_from_all_ranks()
            master_ip_address = rank_ip_addresses[0]

            count_ranks_per_ip_address = Counter(rank_ip_addresses)
            local_world_size = count_ranks_per_ip_address[master_ip_address]
            for rank_ip_address, count in count_ranks_per_ip_address.items():
                if count != local_world_size:
                    raise ValueError(
                        f"All ranks must have the same number of processes, but found {count} processes for rank {rank} on ip {rank_ip_address}, expected {local_world_size}."
                        + f"count_ranks_per_ip_address = {count_ranks_per_ip_address}"
                    )

            node_world_size = len(count_ranks_per_ip_address)
            local_rank = rank % local_world_size
            node_rank = rank // local_world_size

        del (
            context,
            local_process_rank,
            local_process_world_size,
        )  # delete deprecated vars so we don't accidentally use them.

        device = (
            pin_memory_device
            if pin_memory_device
            else gigl.distributed.utils.get_available_device(
                local_process_rank=local_rank
            )
        )
        logger.info(
            f"Dataset Building started on {node_rank} of {node_world_size} nodes, using following node as main: {master_ip_address}"
        )

        if input_nodes is None:
            if dataset.node_ids is None:
                raise ValueError(
                    "Dataset must have node ids if input_nodes are not provided."
                )
            if isinstance(dataset.node_ids, abc.Mapping):
                raise ValueError(
                    f"input_nodes must be provided for heterogeneous datasets, received node_ids of type: {dataset.node_ids.keys()}"
                )
            input_nodes = dataset.node_ids

        # Determines if the node ids passed in are heterogeneous or homogeneous.
        self._is_labeled_heterogeneous = False
        if isinstance(input_nodes, torch.Tensor):
            node_ids = input_nodes

            # If the dataset is heterogeneous, we may be in the "labeled homogeneous" setting,
            # if so, then we should use DEFAULT_HOMOGENEOUS_NODE_TYPE.
            if isinstance(dataset.node_ids, abc.Mapping):
                if (
                    len(dataset.node_ids) == 1
                    and DEFAULT_HOMOGENEOUS_NODE_TYPE in dataset.node_ids
                ):
                    node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
                    self._is_labeled_heterogeneous = True
                else:
                    raise ValueError(
                        f"For heterogeneous datasets, input_nodes must be a tuple of (node_type, node_ids) OR if it is a labeled homogeneous dataset, input_nodes may be a torch.Tensor. Received node types: {dataset.node_ids.keys()}"
                    )
            else:
                node_type = None
        else:
            node_type, node_ids = input_nodes
            assert isinstance(
                dataset.node_ids, abc.Mapping
            ), "Dataset must be heterogeneous if provided input nodes are a tuple."

        num_neighbors = patch_fanout_for_sampling(
            dataset.get_edge_types(), num_neighbors
        )

        curr_process_nodes = shard_nodes_by_process(
            input_nodes=node_ids,
            local_process_rank=local_rank,
            local_process_world_size=local_world_size,
        )

        input_data = NodeSamplerInput(node=curr_process_nodes, input_type=node_type)

        # Sets up processes and torch device for initializing the GLT DistNeighborLoader, setting up RPC and worker groups to minimize
        # the memory overhead and CPU contention.
        logger.info(
            f"Initializing neighbor loader worker in process: {local_rank}/{local_world_size} using device: {device}"
        )
        should_use_cpu_workers = device.type == "cpu"
        if should_use_cpu_workers and num_cpu_threads is None:
            logger.info(
                "Using CPU workers, but found num_cpu_threads to be None. "
                f"Will default setting num_cpu_threads to {DEFAULT_NUM_CPU_THREADS}."
            )
            num_cpu_threads = DEFAULT_NUM_CPU_THREADS

        neighbor_loader_ports = gigl.distributed.utils.get_free_ports_from_master_node(
            num_ports=local_world_size
        )
        neighbor_loader_port_for_current_rank = neighbor_loader_ports[local_rank]

        gigl.distributed.utils.init_neighbor_loader_worker(
            master_ip_address=master_ip_address,
            local_process_rank=local_rank,
            local_process_world_size=local_world_size,
            rank=node_rank,
            world_size=node_world_size,
            master_worker_port=neighbor_loader_port_for_current_rank,
            device=device,
            should_use_cpu_workers=should_use_cpu_workers,
            # Lever to explore tuning for CPU based inference
            num_cpu_threads=num_cpu_threads,
            process_start_gap_seconds=process_start_gap_seconds,
        )
        logger.info(
            f"Finished initializing neighbor loader worker:  {local_rank}/{local_world_size}"
        )

        # Sets up worker options for the dataloader
        dist_sampling_ports = gigl.distributed.utils.get_free_ports_from_master_node(
            num_ports=local_world_size
        )
        dist_sampling_port_for_current_rank = dist_sampling_ports[local_rank]

        worker_options = MpDistSamplingWorkerOptions(
            num_workers=num_workers,
            worker_devices=[torch.device("cpu") for _ in range(num_workers)],
            worker_concurrency=worker_concurrency,
            # Each worker will spawn several sampling workers, and all sampling workers spawned by workers in one group
            # need to be connected. Thus, we need master ip address and master port to
            # initate the connection.
            # Note that different groups of workers are independent, and thus
            # the sampling processes in different groups should be independent, and should
            # use different master ports.
            master_addr=master_ip_address,
            master_port=dist_sampling_port_for_current_rank,
            # Load testing show that when num_rpc_threads exceed 16, the performance
            # will degrade.
            num_rpc_threads=min(dataset.num_partitions, 16),
            rpc_timeout=600,
            channel_size=channel_size,
            pin_memory=device.type == "cuda",
        )

        sampling_config = SamplingConfig(
            sampling_type=SamplingType.NODE,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            with_edge=True,
            collect_features=True,
            with_neg=False,
            with_weight=False,
            edge_dir=dataset.edge_dir,
            seed=None,  # it's actually optional - None means random.
        )

        if should_cleanup_distributed_context and torch.distributed.is_initialized():
            logger.info(
                f"Cleaning up process group as it was initialized inside {self.__class__.__name__}.__init__."
            )
            torch.distributed.destroy_process_group()

        super().__init__(dataset, input_data, sampling_config, device, worker_options)

    def _collate_fn(self, msg: SampleMessage) -> Union[Data, HeteroData]:
        data = super()._collate_fn(msg)
        if isinstance(data, HeteroData):
            data = strip_label_edges(data)

        if self._is_labeled_heterogeneous:
            data = labeled_to_homogeneous(DEFAULT_HOMOGENEOUS_EDGE_TYPE, data)
        return data
