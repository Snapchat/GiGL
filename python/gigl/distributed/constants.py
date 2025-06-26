# TODO (mkolodner-sc): Set these ports dynamically while ensuring no overlap
# Ports for various purposes, we need to make sure they do not overlap.
# Note that [master_port_for_inference, master_port_for_inference + num_inference_processes).
# ports are used. Same for master port for sampling.
DEFAULT_MASTER_INFERENCE_PORT = 10_000
DEFAULT_MASTER_SAMPLING_PORT = 20_000
DEFAULT_MASTER_DATA_BUILDING_PORT = 30_000
