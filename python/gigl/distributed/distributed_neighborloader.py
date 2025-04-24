from collections import abc
from typing import Dict, List, Optional, Tuple, Union

import graphlearn_torch
import torch
from torch_geometric.typing import EdgeType

import gigl.distributed.utils
from gigl.common.logger import Logger
from gigl.distributed import DistributedContext
from gigl.distributed.constants import (
    DEFAULT_MASTER_INFERENCE_PORT,
    DEFAULT_MASTER_SAMPLING_PORT,
)
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.src.common.types.graph_data import (
    NodeType,  # TODO (mkolodner-sc): Change to use torch_geometric.typing
)

logger = Logger()

# When using CPU based inference/training, we default cpu threads for neighborloading on top of the per process parallelism.
DEFAULT_NUM_CPU_THREADS = 2


class DistNeighborLoader(graphlearn_torch.distributed.DistNeighborLoader):
    def __init__(
        self,
        dataset: DistLinkPredictionDataset,
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        context: DistributedContext,
        local_process_rank: int,  # TODO: Move this to DistributedContext
        local_process_world_size: int,  # TODO: Move this to DistributedContext
        input_nodes: Optional[
            Union[torch.Tensor, Tuple[NodeType, torch.Tensor]]
        ] = None,
        num_workers: int = 1,
        batch_size: int = 1,
        pin_memory_device: Optional[torch.device] = None,
        worker_concurrency: int = 4,
        channel_size: str = "4GB",
        process_start_gap_seconds: int = 60,
        num_cpu_threads: Optional[int] = None,
        _main_inference_port: int = DEFAULT_MASTER_INFERENCE_PORT,
        _main_sampling_port: int = DEFAULT_MASTER_SAMPLING_PORT,
    ):
        """
        Note: We try to adhere to pyg dataloader api as much as possible.
        See the following for reference:
        https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/loader/node_loader.html#NodeLoader
        https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/distributed/dist_neighbor_loader.html#DistNeighborLoader

        Args:
            dataset (DistLinkPredictionDataset): The dataset to sample from.
            num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]):
                The number of neighbors to sample for each node in each iteration.
                If an entry is set to `-1`, all neighbors will be included.
                In heterogeneous graphs, may also take in a dictionary denoting
                the amount of neighbors to sample for each individual edge type.
            context (DistributedContext): Distributed context information of the current process.
            local_process_rank (int): The local rank of the current process within a node.
            local_process_world_size (int): The total number of processes within a node.
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
            _main_inference_port (int): WARNING: You don't need to configure this unless port conflict issues. Slotted for refactor.
                The port number to use for inference processes.
                In future, the port will be automatically assigned based on availability.
                Currently defaults to: gigl.distributed.constants.DEFAULT_MASTER_INFERENCE_PORT
            _main_sampling_port (int): WARNING: You don't need to configure this unless port conflict issues. Slotted for refactor.
                The port number to use for sampling processes.
                In future, the port will be automatically assigned based on availability.
                Currently defaults to: gigl.distributed.constants.DEFAULT_MASTER_SAMPLING_PORT
        """

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

        if isinstance(num_neighbors, abc.Mapping):
            # TODO(kmonte): We should enable this. We have two blockers:
            # 1. We need to treat `EdgeType` as a proper tuple, not the GiGL`EdgeType`.
            # 2. There are (likely) some GLT bugs around https://github.com/alibaba/graphlearn-for-pytorch/blob/26fe3d4e050b081bc51a79dc9547f244f5d314da/graphlearn_torch/python/distributed/dist_neighbor_sampler.py#L317-L318
            # Where if num_neighbors is a dict then we index into it improperly.
            raise ValueError(
                f"num_neighbors must be a list of integers, received: {num_neighbors}"
            )
        curr_process_nodes = _shard_nodes_by_process(
            input_nodes=input_nodes,
            local_process_rank=local_process_rank,
            local_process_world_size=local_process_world_size,
        )
        device = (
            pin_memory_device
            if pin_memory_device
            else gigl.distributed.utils.get_available_device(
                local_process_rank=local_process_rank
            )
        )
        # Sets up processes and torch device for initializing the GLT DistNeighborLoader, setting up RPC and worker groups to minimize
        # the memory overhead and CPU contention.
        logger.info(
            f"Initializing neighbor loader worker in process: {local_process_rank}/{local_process_world_size} using device: {device}"
        )
        should_use_cpu_workers = device.type == "cpu"
        if should_use_cpu_workers and num_cpu_threads is None:
            logger.info(
                "Using CPU workers, but found num_cpu_threads to be None. "
                f"Will default setting num_cpu_threads to {DEFAULT_NUM_CPU_THREADS}."
            )
            num_cpu_threads = DEFAULT_NUM_CPU_THREADS
        gigl.distributed.utils.init_neighbor_loader_worker(
            master_ip_address=context.main_worker_ip_address,
            local_process_rank=local_process_rank,
            local_process_world_size=local_process_world_size,
            rank=context.global_rank,
            world_size=context.global_world_size,
            master_worker_port=_main_inference_port,
            device=device,
            should_use_cpu_workers=should_use_cpu_workers,
            # Lever to explore tuning for CPU based inference
            num_cpu_threads=num_cpu_threads,
            process_start_gap_seconds=process_start_gap_seconds,
        )
        logger.info(
            f"Finished initializing neighbor loader worker:  {local_process_rank}/{local_process_world_size}"
        )

        # Sets up worker options for the dataloader
        worker_options = graphlearn_torch.distributed.MpDistSamplingWorkerOptions(
            num_workers=num_workers,
            worker_devices=[torch.device("cpu") for _ in range(num_workers)],
            worker_concurrency=worker_concurrency,
            # Each worker will spawn several sampling workers, and all sampling workers spawned by workers in one group
            # need to be connected. Thus, we need master ip address and master port to
            # initate the connection.
            # Note that different groups of workers are independent, and thus
            # the sampling processes in different groups should be independent, and should
            # use different master ports.
            master_addr=context.main_worker_ip_address,
            master_port=_main_sampling_port + local_process_rank,
            # Load testing show that when num_rpc_threads exceed 16, the performance
            # will degrade.
            num_rpc_threads=min(dataset.num_partitions, 16),
            rpc_timeout=600,
            channel_size=channel_size,
            pin_memory=device.type == "cuda",
        )

        super().__init__(
            data=dataset,
            num_neighbors=num_neighbors,
            input_nodes=curr_process_nodes,
            batch_size=batch_size,
            with_edge=True,
            edge_dir=dataset.edge_dir,
            collect_features=True,
            to_device=device,
            worker_options=worker_options,
        )


def _shard_nodes_by_process(
    input_nodes: Union[torch.Tensor, Tuple[str, torch.Tensor]],
    local_process_rank: int,
    local_process_world_size: int,
) -> Union[torch.Tensor, Tuple[str, torch.Tensor]]:
    def shard(nodes: torch.Tensor) -> torch.Tensor:
        num_node_ids_per_process = nodes.size(0) // local_process_world_size
        start_index = local_process_rank * num_node_ids_per_process
        end_index = (
            nodes.size(0)
            if local_process_rank == local_process_world_size - 1
            else start_index + num_node_ids_per_process
        )
        nodes_for_current_process = nodes[start_index:end_index]
        return nodes_for_current_process

    if isinstance(input_nodes, torch.Tensor):
        return shard(input_nodes)
    else:
        node_type, node_ids = input_nodes
        node_ids = shard(node_ids)
        return (node_type, node_ids)
