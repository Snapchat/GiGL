"""
Minimum example for `launch_graph_store_enabled_job` that runs a diagnostic
script on both the compute and storage worker pools.

Usage:
    # Minimal CPU-only example (both pools on n1-standard-4, 1 replica each):
    python -m examples.diagnostic_vertex_ai_job \
        --project=my-gcp-project \
        --service_account=my-sa@my-project.iam.gserviceaccount.com \
        --staging_bucket=gs://my-staging-bucket

    # Custom machine types per pool, with GPU on compute:
    python -m examples.diagnostic_vertex_ai_job \
        --project=my-gcp-project \
        --service_account=my-sa@my-project.iam.gserviceaccount.com \
        --staging_bucket=gs://my-staging-bucket \
        --compute_machine_type=n1-standard-8 \
        --compute_gpu_type=NVIDIA_TESLA_T4 \
        --compute_gpu_count=1 \
        --compute_num_replicas=2 \
        --storage_machine_type=n2-standard-4 \
        --storage_num_replicas=1

    # With a custom Docker image:
    python -m examples.diagnostic_vertex_ai_job \
        --project=my-gcp-project \
        --service_account=my-sa@my-project.iam.gserviceaccount.com \
        --staging_bucket=gs://my-staging-bucket \
        --docker_uri=us-docker.pkg.dev/my-project/my-repo/my-image:latest
"""

import argparse
import uuid

from gigl.common import Uri
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.vertex_ai_launcher import launch_graph_store_enabled_job
from snapchat.research.gbml.gigl_resource_config_pb2 import (
    GiglResourceConfig,
    SharedResourceConfig,
    VertexAiGraphStoreConfig,
    VertexAiResourceConfig,
)

# ---------------------------------------------------------------------------
# Diagnostic code that will run on every worker in both pools.
# Written as semicolon-separated statements for readability.
# ---------------------------------------------------------------------------
DIAGNOSTIC_CODE = (
    "import sys;"
    "import os;"
    "import time;"
    "os.environ['PYTHONUNBUFFERED']='1';"
    "print('DIAGNOSTIC L1: Print started',flush=True);"
    "sys.stderr.write('DIAGNOSTIC L2: Stderr started\\n');"
    "sys.stderr.flush();"
    "print(f'DIAGNOSTIC L3: PID: {os.getpid()}',flush=True);"
    "time.sleep(5);"
    "print('DIAGNOSTIC L4: Script ending normally',flush=True);"
    "sys.exit(0)"
)


def _build_process_command(code: str) -> str:
    """Build a ``python -c <code>`` process_command string that survives
    ``_build_job_config``'s ``command_str.strip().split(" ")`` splitting.

    Spaces in *code* are replaced with tab characters (``\\t``).  Python
    treats tabs as valid whitespace for syntax purposes, so ``import\\tsys``
    is equivalent to ``import sys``.  This ensures the code stays as a single
    token after splitting on spaces.

    Note: spaces inside string literals will appear as tabs in the output,
    which is acceptable for diagnostic purposes.
    """
    # After split(" ") this yields:
    #   ["python", "-c", "import\\tsys;import\\tos;..."]
    return f"python -c {code.replace(' ', chr(9))}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a diagnostic Vertex AI graph-store-enabled job "
        "via launch_graph_store_enabled_job."
    )

    # --- Required --------------------------------------------------------
    parser.add_argument(
        "--project", required=True, help="GCP project ID."
    )
    parser.add_argument(
        "--service_account",
        required=True,
        help="GCP service account email.",
    )
    parser.add_argument(
        "--staging_bucket",
        required=True,
        help="GCS staging bucket (e.g. gs://my-bucket).",
    )

    # --- Shared ----------------------------------------------------------
    parser.add_argument(
        "--region",
        default="us-central1",
        help="GCP region (default: us-central1).",
    )
    parser.add_argument(
        "--docker_uri",
        default=None,
        help="Custom Docker image URI for both pools. "
        "If omitted, the default GiGL release images are used.",
    )
    parser.add_argument(
        "--job_name",
        default=None,
        help="Custom job name. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--component",
        default="Trainer",
        choices=["Trainer", "Inferencer"],
        help="GiGL component type (default: Trainer).",
    )

    # --- Compute pool configuration --------------------------------------
    compute = parser.add_argument_group("compute pool")
    compute.add_argument(
        "--compute_machine_type",
        default="n1-standard-4",
        help="Compute pool machine type (default: n1-standard-4).",
    )
    compute.add_argument(
        "--compute_gpu_type",
        default="",
        help="Compute pool GPU type (e.g. NVIDIA_TESLA_T4). "
        "Leave empty for CPU-only.",
    )
    compute.add_argument(
        "--compute_gpu_count",
        type=int,
        default=0,
        help="Number of GPUs per compute replica (default: 0).",
    )
    compute.add_argument(
        "--compute_num_replicas",
        type=int,
        default=1,
        help="Number of compute replicas (default: 1).",
    )
    compute.add_argument(
        "--compute_cluster_local_world_size",
        type=int,
        default=0,
        help="Number of sampling processes per compute machine. "
        "0 = auto (GPU count if GPUs present, else 1).",
    )

    # --- Storage pool configuration --------------------------------------
    storage = parser.add_argument_group("storage pool")
    storage.add_argument(
        "--storage_machine_type",
        default="n1-standard-4",
        help="Storage pool machine type (default: n1-standard-4).",
    )
    storage.add_argument(
        "--storage_gpu_type",
        default="ACCELERATOR_TYPE_UNSPECIFIED",
        help="Storage pool GPU type. Leave empty for CPU-only.",
    )
    storage.add_argument(
        "--storage_gpu_count",
        type=int,
        default=0,
        help="Number of GPUs per storage replica (default: 0).",
    )
    storage.add_argument(
        "--storage_num_replicas",
        type=int,
        default=1,
        help="Number of storage replicas (default: 1).",
    )

    args = parser.parse_args()

    # -- VertexAiGraphStoreConfig (protobuf) ------------------------------
    compute_pool = VertexAiResourceConfig(
        machine_type=args.compute_machine_type,
        gpu_type=args.compute_gpu_type,
        gpu_limit=args.compute_gpu_count,
        num_replicas=args.compute_num_replicas,
    )
    storage_pool = VertexAiResourceConfig(
        machine_type=args.storage_machine_type,
        gpu_type=args.storage_gpu_type,
        gpu_limit=args.storage_gpu_count,
        num_replicas=args.storage_num_replicas,
    )
    graph_store_config = VertexAiGraphStoreConfig(
        compute_pool=compute_pool,
        graph_store_pool=storage_pool,
        compute_cluster_local_world_size=args.compute_cluster_local_world_size,
    )

    # -- GiglResourceConfigWrapper (minimal) ------------------------------
    shared_resource_config = SharedResourceConfig(
        resource_labels={},
        common_compute_config=SharedResourceConfig.CommonComputeConfig(
            project=args.project,
            region=args.region,
            temp_assets_bucket=args.staging_bucket,
            temp_regional_assets_bucket=args.staging_bucket,
            perm_assets_bucket=args.staging_bucket,
            temp_assets_bq_dataset_name="unused",
            embedding_bq_dataset_name="unused",
            gcp_service_account_email=args.service_account,
            dataflow_runner="DirectRunner",
        ),
    )
    resource_config_wrapper = GiglResourceConfigWrapper(
        resource_config=GiglResourceConfig(
            shared_resource_config=shared_resource_config,
        )
    )

    # -- Build the diagnostic command for both pools ----------------------
    process_command = _build_process_command(DIAGNOSTIC_CODE)

    job_name = args.job_name or f"diagnostic-gs-job-{uuid.uuid4().hex[:8]}"
    component = GiGLComponents[args.component]

    # task_config_uri / resource_config_uri are required by the function
    # signature but are not consumed by the diagnostic script.
    dummy_uri = Uri("gs://unused/placeholder")

    print(f"Launching diagnostic graph-store job: {job_name}")
    print(f"  Region:               {args.region}")
    print(f"  Compute pool:         {args.compute_machine_type}, "
          f"GPU: {args.compute_gpu_type or '(none)'} x{args.compute_gpu_count}, "
          f"replicas: {args.compute_num_replicas}")
    print(f"  Storage pool:         {args.storage_machine_type}, "
          f"GPU: {args.storage_gpu_type or '(none)'} x{args.storage_gpu_count}, "
          f"replicas: {args.storage_num_replicas}")
    print(f"  Local world size:     {args.compute_cluster_local_world_size or '(auto)'}")
    print(f"  Docker:               {args.docker_uri or '(default GiGL images)'}")

    launch_graph_store_enabled_job(
        vertex_ai_graph_store_config=graph_store_config,
        job_name=job_name,
        task_config_uri=dummy_uri,
        resource_config_uri=dummy_uri,
        compute_commmand=process_command,
        compute_runtime_args={},
        storage_command=process_command,
        storage_args={},
        resource_config_wrapper=resource_config_wrapper,
        cpu_docker_uri=args.docker_uri,
        cuda_docker_uri=args.docker_uri,
        component=component,
    )

    print(f"Job {job_name} completed.")


if __name__ == "__main__":
    main()
