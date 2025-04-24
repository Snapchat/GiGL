from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    """
    GiGL Distributed Context
    """

    # TODO (mkolodner-sc): Investigate adding local rank and local world size

    # Main Worker's IP Address for RPC communication
    main_worker_ip_address: str

    # Rank of machine
    global_rank: int

    # Total number of machines
    global_world_size: int
