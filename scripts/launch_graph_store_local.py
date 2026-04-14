#!/usr/bin/env python3
"""Launch a local graph-store cluster by emulating the Vertex AI multi-pool environment.

This script spawns separate storage and compute node processes on a single machine,
wiring them together with the same environment variables and cluster spec that Vertex AI
would provide in a real multi-pool training job. Storage and compute entry-point commands
are read from the task config YAML from either ``trainerConfig`` or ``inferencerConfig``,
depending on ``--mode``.

Example - training with 3 storage nodes and 2 compute nodes::

    python scripts/launch_graph_store_local.py \\
        --storage-nodes 3 \\
        --compute-nodes 2 \\
        --compute-procs-per-node 1 \\
        --mode train \\
        --task-config-uri applied_tasks/unsupervised_user_embeddings/small_frozen_training_gs.yaml \\
        --resource-config-uri deployment/configs/unittest_resource_config.yaml

Example - inference with GPU accelerators::

    python scripts/launch_graph_store_local.py \\
        --storage-nodes 2 \\
        --compute-nodes 2 \\
        --compute-procs-per-node 1 \\
        --mode infer \\
        --gpus-per-compute-node 1 \\
        --task-config-uri path/to/task_config.yaml \\
        --resource-config-uri path/to/resource_config.yaml

Each launched process receives:

* ``CLUSTER_SPEC`` - JSON blob describing the worker-pool topology.
* ``RANK`` - global rank across both storage and compute pools.
* ``MASTER_ADDR`` / ``MASTER_PORT`` - rendezvous endpoint on the local host.
* ``CUDA_VISIBLE_DEVICES`` - restricted to the assigned accelerators (empty for storage).

Logs for every process are streamed to the console and written to individual files under
the job directory (``<job-root>/gs-job-<job-id>/``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TextIO

import yaml

DEFAULT_JOB_ROOT = Path("/tmp/gigl")
JOB_ID_SUFFIX_RE = re.compile(r"^(.*?)(\d+)$")


@dataclass(frozen=True)
class NodeProcess:
    """Configuration for one launched storage or compute node."""

    name: str
    role: str
    rank: int
    command: list[str]
    env: dict[str, str]
    log_path: Path
    cluster_spec: dict[str, Any]


RunningChild = tuple[NodeProcess, subprocess.Popen[str], TextIO, threading.Thread]


def _get_free_ports(num_ports: int) -> list[int]:
    """Reserve *num_ports* ephemeral TCP ports and return their numbers."""

    ports: list[int] = []
    sockets: list[socket.socket] = []
    try:
        for _ in range(num_ports):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", 0))
            sockets.append(sock)
            ports.append(sock.getsockname()[1])
        return ports
    finally:
        for sock in sockets:
            sock.close()


def _increment_job_id(job_id: str) -> str:
    """Bump the trailing numeric suffix of *job_id*, or append ``-1`` if none exists.

    >>> _increment_job_id("my-job-3")
    'my-job-4'
    >>> _increment_job_id("my-job")
    'my-job-1'
    """

    suffix_match = JOB_ID_SUFFIX_RE.match(job_id)
    if suffix_match is None:
        return f"{job_id}-1"
    prefix, numeric_suffix = suffix_match.groups()
    return f"{prefix}{int(numeric_suffix) + 1}"


def _allocate_job_dir(
    job_root: Path, requested_job_id: Optional[str]
) -> tuple[str, Path]:
    """Create a unique job directory under *job_root*."""

    candidate_job_id = (
        requested_job_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    )
    while True:
        candidate_job_dir = job_root / f"gs-job-{candidate_job_id}"
        try:
            candidate_job_dir.mkdir(parents=True, exist_ok=False)
            return candidate_job_id, candidate_job_dir
        except FileExistsError:
            candidate_job_id = _increment_job_id(candidate_job_id)


def _parse_gpu_ids(raw_ids: Optional[str]) -> Optional[list[int]]:
    """Parse a comma-separated string of GPU ids into a list of ints."""

    if raw_ids is None:
        return None
    tokens = [token.strip() for token in raw_ids.split(",") if token.strip()]
    return [int(token) for token in tokens]


def _detect_visible_gpus() -> list[int]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def _assign_gpus(
    *,
    compute_nodes: int,
    gpus_per_compute_node: int,
    gpu_ids: Optional[list[int]],
) -> list[list[int]]:
    """Map physical GPU ids to compute nodes."""

    if gpus_per_compute_node == 0:
        return [[] for _ in range(compute_nodes)]

    available_gpu_ids = gpu_ids if gpu_ids is not None else _detect_visible_gpus()
    if not available_gpu_ids:
        raise ValueError(
            "No CUDA accelerators are visible, but --gpus-per-compute-node was set"
        )

    total_requested = compute_nodes * gpus_per_compute_node
    if total_requested > len(available_gpu_ids):
        raise ValueError(
            f"Requested {total_requested} GPUs, but only {len(available_gpu_ids)} are visible"
        )

    assignments: list[list[int]] = []
    for compute_rank in range(compute_nodes):
        start = compute_rank * gpus_per_compute_node
        stop = start + gpus_per_compute_node
        assignments.append(available_gpu_ids[start:stop])
    return assignments


def resolve_input_uri(value: str, flag_name: str) -> str:
    """Resolve a launcher input URI.

    Args:
        value: Local path or remote URI.
        flag_name: CLI flag name used in error messages.

    Returns:
        The original URI for remote inputs, or an absolute local path.

    Raises:
        FileNotFoundError: If *value* is a local path that does not exist.
    """

    if "://" in value:
        return value

    resolved_path = Path(value).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"{flag_name} path does not exist: {value}")
    return str(resolved_path)


def _load_yaml_payload(uri: str) -> dict[str, Any]:
    """Load and return a YAML payload from a local path or remote URI."""

    if "://" not in uri:
        with open(uri, "r", encoding="utf-8") as file:
            payload = yaml.safe_load(file)
        return payload or {}

    from gigl.common import UriFactory
    from gigl.src.common.utils.file_loader import FileLoader

    temp_file = FileLoader().load_to_temp_file(
        file_uri_src=UriFactory.create_uri(uri),
        delete=False,
    )
    try:
        with open(temp_file.name, "r", encoding="utf-8") as file:
            payload = yaml.safe_load(file)
        return payload or {}
    finally:
        temp_path = Path(temp_file.name)
        temp_file.close()
        temp_path.unlink(missing_ok=True)


def load_component_config(task_config_uri: str, mode: str) -> dict[str, Any]:
    """Load storage and compute runtime config from a task config YAML.

    Args:
        task_config_uri: Local path or remote URI to the task config YAML.
        mode: ``"train"`` or ``"infer"``.

    Returns:
        A dictionary containing the selected component name, commands, and args.

    Raises:
        ValueError: If *mode* is invalid or the selected component is missing a
            required command field.
    """

    payload = _load_yaml_payload(task_config_uri)
    compute_arg_names: tuple[str, ...]
    if mode == "train":
        component_name = "trainerConfig"
        component_payload = payload.get("trainerConfig") or payload.get(
            "trainer_config"
        )
        compute_arg_names = ("trainerArgs", "trainer_args")
    elif mode == "infer":
        component_name = "inferencerConfig"
        component_payload = payload.get("inferencerConfig") or payload.get(
            "inferencer_config"
        )
        compute_arg_names = (
            "inferencerArgs",
            "inferencer_args",
            "inferenceArgs",
            "inference_args",
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if not isinstance(component_payload, dict):
        component_payload = {}

    graph_store_payload = component_payload.get(
        "graphStoreStorageConfig"
    ) or component_payload.get("graph_store_storage_config")
    if not isinstance(graph_store_payload, dict):
        graph_store_payload = {}

    compute_command = component_payload.get("command")
    if not compute_command:
        raise ValueError(f"Task config {component_name} has no 'command' field")

    storage_command = graph_store_payload.get("command")
    if not storage_command:
        raise ValueError(
            f"Task config {component_name}.graphStoreStorageConfig has no 'command' field"
        )

    compute_args: dict[str, str] = {}
    for arg_name in compute_arg_names:
        raw_args = component_payload.get(arg_name)
        if isinstance(raw_args, dict):
            compute_args = {str(key): str(value) for key, value in raw_args.items()}
            break

    storage_args: dict[str, str] = {}
    for arg_name in ("storageArgs", "storage_args"):
        raw_args = graph_store_payload.get(arg_name)
        if isinstance(raw_args, dict):
            storage_args = {str(key): str(value) for key, value in raw_args.items()}
            break

    return {
        "name": component_name,
        "compute_command": str(compute_command),
        "storage_command": str(storage_command),
        "compute_args": compute_args,
        "storage_args": storage_args,
    }


def _resolve_main_command(main: str, python_bin: str) -> list[str]:
    """Turn a user-supplied *main* specifier into an argv list."""

    if any(ch.isspace() for ch in main):
        return shlex.split(main)
    if main.endswith(".py"):
        return [python_bin, str(Path(main).expanduser().resolve())]
    return [python_bin, "-m", main]


def _build_process_command(
    *,
    main: str,
    python_bin: str,
    job_id: str,
    task_config_uri: str,
    resource_config_uri: str,
    runtime_args: dict[str, str],
) -> list[str]:
    """Assemble the argv for one launched storage or compute child process."""

    command = _resolve_main_command(main=main, python_bin=python_bin)
    command.extend(
        [
            f"--job_name={job_id}",
            f"--task_config_uri={task_config_uri}",
            f"--resource_config_uri={resource_config_uri}",
        ]
    )
    command.extend(f"--{key}={value}" for key, value in runtime_args.items())
    return command


def _build_cluster_spec(
    *,
    compute_nodes: int,
    storage_nodes: int,
    rank: int,
) -> dict[str, Any]:
    """Build a Vertex AI-style ``CLUSTER_SPEC`` JSON object for one process."""

    cluster: dict[str, list[str]] = {
        "workerpool0": ["workerpool0-0:2222"],
        "workerpool2": [f"workerpool2-{index}:2222" for index in range(storage_nodes)],
    }
    if compute_nodes > 1:
        cluster["workerpool1"] = [
            f"workerpool1-{index}:2222" for index in range(compute_nodes - 1)
        ]

    if rank == 0:
        task_type = "workerpool0"
        task_index = 0
    elif rank < compute_nodes:
        task_type = "workerpool1"
        task_index = rank - 1
    else:
        task_type = "workerpool2"
        task_index = rank - compute_nodes

    return {
        "cluster": cluster,
        "environment": "cloud",
        "task": {
            "type": task_type,
            "index": task_index,
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_node_processes(
    *,
    storage_nodes: int,
    compute_nodes: int,
    python_bin: str,
    job_id: str,
    task_config_uri: str,
    resource_config_uri: str,
    component_config: dict[str, Any],
    job_dir: Path,
    base_env: dict[str, str],
    gpu_assignments: list[list[int]],
) -> list[NodeProcess]:
    """Build the storage and compute processes for one launcher run.

    Args:
        storage_nodes: Number of storage nodes to launch.
        compute_nodes: Number of compute nodes to launch.
        python_bin: Python interpreter used for child entrypoints.
        job_id: Job identifier passed to child processes.
        task_config_uri: Task config URI passed to child processes.
        resource_config_uri: Resource config URI passed to child processes.
        component_config: Storage/compute commands and runtime args.
        job_dir: Output directory for per-process logs.
        base_env: Environment variables shared by every child.
        gpu_assignments: One GPU-id list per compute node.

    Returns:
        Storage nodes followed by compute nodes.
    """

    node_descriptors: list[tuple[str, int, int]] = []
    for storage_index in range(storage_nodes):
        node_descriptors.append(
            ("storage", storage_index, compute_nodes + storage_index)
        )
    for compute_index in range(compute_nodes):
        node_descriptors.append(("compute", compute_index, compute_index))

    processes: list[NodeProcess] = []
    for role, node_index, rank in node_descriptors:
        cluster_spec = _build_cluster_spec(
            compute_nodes=compute_nodes,
            storage_nodes=storage_nodes,
            rank=rank,
        )
        env = base_env.copy()
        env["RANK"] = str(rank)
        env["CLUSTER_SPEC"] = json.dumps(cluster_spec)

        if role == "storage":
            env["CUDA_VISIBLE_DEVICES"] = ""
            command = _build_process_command(
                main=component_config["storage_command"],
                python_bin=python_bin,
                job_id=job_id,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                runtime_args=component_config["storage_args"],
            )
        else:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu_id) for gpu_id in gpu_assignments[node_index]
            )
            command = _build_process_command(
                main=component_config["compute_command"],
                python_bin=python_bin,
                job_id=job_id,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                runtime_args=component_config["compute_args"],
            )

        processes.append(
            NodeProcess(
                name=f"{role}_node_{node_index}",
                role=role,
                rank=rank,
                command=command,
                env=env,
                log_path=job_dir / f"{role}_node_{node_index}.log",
                cluster_spec=cluster_spec,
            )
        )

    return processes


def _pump_logs(
    *,
    process_name: str,
    popen: subprocess.Popen[str],
    log_handle: TextIO,
    console_lock: threading.Lock,
) -> None:
    assert popen.stdout is not None
    try:
        for line in popen.stdout:
            log_handle.write(line)
            log_handle.flush()
            with console_lock:
                sys.stdout.write(f"[{process_name}] {line}")
                sys.stdout.flush()
    finally:
        popen.stdout.close()


def _spawn_process(
    node: NodeProcess,
    *,
    cwd: Path,
    console_lock: threading.Lock,
) -> RunningChild:
    log_handle = node.log_path.open("w", encoding="utf-8")
    popen = subprocess.Popen(
        node.command,
        cwd=str(cwd),
        env=node.env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    log_thread = threading.Thread(
        target=_pump_logs,
        kwargs={
            "process_name": node.name,
            "popen": popen,
            "log_handle": log_handle,
            "console_lock": console_lock,
        },
        daemon=True,
    )
    log_thread.start()
    return node, popen, log_handle, log_thread


def _kill_process_group(popen: subprocess.Popen[str], sig: int) -> None:
    try:
        os.killpg(popen.pid, sig)
    except ProcessLookupError:
        pass


def _shutdown_processes(
    running_children: list[RunningChild],
    term_timeout_seconds: float = 10.0,
) -> None:
    for _, popen, _, _ in running_children:
        if popen.poll() is None:
            _kill_process_group(popen, signal.SIGTERM)

    deadline = time.monotonic() + term_timeout_seconds
    while time.monotonic() < deadline:
        if all(popen.poll() is not None for _, popen, _, _ in running_children):
            return
        time.sleep(0.2)

    for _, popen, _, _ in running_children:
        if popen.poll() is None:
            _kill_process_group(popen, signal.SIGKILL)


def _launch_processes(
    *,
    nodes: list[NodeProcess],
    cwd: Path,
    job_dir: Path,
    launch_delay_s: float,
) -> int:
    """Spawn all process specs and wait for them to finish."""

    console_lock = threading.Lock()
    running_children: list[RunningChild] = []
    try:
        for index, node in enumerate(nodes):
            running_children.append(
                _spawn_process(node=node, cwd=cwd, console_lock=console_lock)
            )
            if launch_delay_s > 0 and index != len(nodes) - 1:
                time.sleep(launch_delay_s)

        while True:
            exited_count = 0
            for node, popen, _, _ in running_children:
                return_code = popen.poll()
                if return_code is None:
                    continue
                exited_count += 1
                if return_code != 0:
                    print(
                        f"{node.name} exited with code {return_code}; "
                        f"terminating remaining processes. Logs: {job_dir}",
                        file=sys.stderr,
                    )
                    _shutdown_processes(running_children)
                    return return_code

            if exited_count == len(running_children):
                return 0

            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"Interrupted; terminating job. Logs: {job_dir}", file=sys.stderr)
        _shutdown_processes(running_children)
        return 130
    except Exception:
        _shutdown_processes(running_children)
        raise
    finally:
        for _, popen, log_handle, log_thread in running_children:
            try:
                popen.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            log_thread.join(timeout=5)
            log_handle.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a local graph store cluster by emulating Vertex AI worker pools. "
            "The compute main's own config must still agree with "
            "--compute-procs-per-node."
        )
    )
    parser.add_argument("--storage-nodes", type=int, required=True)
    parser.add_argument("--compute-nodes", type=int, required=True)
    parser.add_argument(
        "--compute-procs-per-node",
        type=int,
        required=True,
        help=(
            "Must match the number of local worker processes the compute main will "
            "spawn on each compute node."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("train", "infer"),
        default="train",
        help="Read trainerConfig for train mode or inferencerConfig for infer mode.",
    )
    parser.add_argument("--task-config-uri", type=str, required=True)
    parser.add_argument("--resource-config-uri", type=str, required=True)
    parser.add_argument(
        "--gpus-per-compute-node",
        type=int,
        default=0,
        help="Number of GPUs to expose to each compute node. Use 0 for CPU-only runs.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids to expose to compute nodes.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help=(
            "Optional job id. If gs-job-<job-id> already exists under --job-root, "
            "increment a trailing number or append -1 until a free path is found."
        ),
    )
    parser.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--python", dest="python_bin", type=str, default=sys.executable)
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Working directory for launched processes. Defaults to the GiGL repo root.",
    )
    parser.add_argument(
        "--launch-delay-s",
        type=float,
        default=0.0,
        help="Optional delay between top-level child launches to reduce import spikes.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the local graph-store launcher.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.storage_nodes < 1:
        raise ValueError("--storage-nodes must be >= 1")
    if args.compute_nodes < 1:
        raise ValueError("--compute-nodes must be >= 1")
    if args.compute_procs_per_node < 1:
        raise ValueError("--compute-procs-per-node must be >= 1")
    if args.gpus_per_compute_node < 0:
        raise ValueError("--gpus-per-compute-node must be >= 0")
    if args.launch_delay_s < 0:
        raise ValueError("--launch-delay-s must be >= 0")
    if 0 < args.gpus_per_compute_node < args.compute_procs_per_node:
        raise ValueError("--gpus-per-compute-node must be >= --compute-procs-per-node")

    task_config_uri = resolve_input_uri(args.task_config_uri, "--task-config-uri")
    resource_config_uri = resolve_input_uri(
        args.resource_config_uri, "--resource-config-uri"
    )
    component_config = load_component_config(
        task_config_uri=task_config_uri, mode=args.mode
    )

    gpu_assignments = _assign_gpus(
        compute_nodes=args.compute_nodes,
        gpus_per_compute_node=args.gpus_per_compute_node,
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
    )

    cwd = args.cwd.expanduser().resolve()
    job_root = args.job_root.expanduser().resolve()
    job_id, job_dir = _allocate_job_dir(job_root, args.job_id)

    master_port = _get_free_ports(1)[0]
    world_size = args.compute_nodes + args.storage_nodes
    base_env = os.environ.copy()
    base_env.update(
        {
            "CLOUD_ML_JOB_ID": job_id,
            "MASTER_ADDR": args.host,
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(world_size),
            "RESOURCE_CONFIG_PATH": resource_config_uri,
            "COMPUTE_CLUSTER_LOCAL_WORLD_SIZE": str(args.compute_procs_per_node),
            "PYTHONUNBUFFERED": "1",
        }
    )

    nodes = build_node_processes(
        storage_nodes=args.storage_nodes,
        compute_nodes=args.compute_nodes,
        python_bin=args.python_bin,
        job_id=job_id,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        component_config=component_config,
        job_dir=job_dir,
        base_env=base_env,
        gpu_assignments=gpu_assignments,
    )

    manifest = {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "mode": args.mode,
        "cwd": str(cwd),
        "host": args.host,
        "master_port": master_port,
        "world_size": world_size,
        "topology": {
            "storage_nodes": args.storage_nodes,
            "compute_nodes": args.compute_nodes,
            "compute_procs_per_node": args.compute_procs_per_node,
        },
        "gpu_assignments": gpu_assignments,
        "component": {
            "name": component_config["name"],
            "storage_command": component_config["storage_command"],
            "compute_command": component_config["compute_command"],
            "storage_args": component_config["storage_args"],
            "compute_args": component_config["compute_args"],
        },
        "processes": [
            {
                "name": node.name,
                "role": node.role,
                "rank": node.rank,
                "command": node.command,
                "log_path": str(node.log_path),
                "cuda_visible_devices": node.env.get("CUDA_VISIBLE_DEVICES", ""),
                "cluster_spec": node.cluster_spec,
            }
            for node in nodes
        ],
    }
    _write_json(job_dir / "launcher_manifest.json", manifest)

    print(f"Launching graph store job {job_id}")
    print(f"Mode: {args.mode}")
    print(f"Logs: {job_dir}")
    print(
        "Note: the compute main's own config must still match "
        f"--compute-procs-per-node={args.compute_procs_per_node}"
    )
    print(f"GPU assignments: {gpu_assignments}")

    return _launch_processes(
        nodes=nodes,
        cwd=cwd,
        job_dir=job_dir,
        launch_delay_s=args.launch_delay_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
