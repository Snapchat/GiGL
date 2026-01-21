"""Unit tests for vertex_ai_launcher module."""

import unittest
from unittest.mock import Mock, patch

from gigl.common import Uri
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.vertex_ai_launcher import (
    launch_graph_store_enabled_job,
    launch_single_pool_job,
)
from snapchat.research.gbml import gigl_resource_config_pb2


def _create_shared_resource_config(
    cost_resource_group: str = "gigl_test",
) -> gigl_resource_config_pb2.SharedResourceConfig:
    """Create a SharedResourceConfig protobuf for testing."""
    common_compute_config = (
        gigl_resource_config_pb2.SharedResourceConfig.CommonComputeConfig(
            project="test-project",
            region="us-central1",
            temp_assets_bucket="gs://test-temp-bucket",
            temp_regional_assets_bucket="gs://test-temp-regional-bucket",
            perm_assets_bucket="gs://test-perm-bucket",
            temp_assets_bq_dataset_name="test_temp_dataset",
            embedding_bq_dataset_name="test_embeddings_dataset",
            gcp_service_account_email="test-sa@test-project.iam.gserviceaccount.com",
            dataflow_runner="DataflowRunner",
        )
    )
    return gigl_resource_config_pb2.SharedResourceConfig(
        resource_labels={
            "env": "test",
            "cost_resource_group_tag": "unittest_COMPONENT",
            "cost_resource_group": cost_resource_group,
        },
        common_compute_config=common_compute_config,
    )


def _create_gigl_resource_config_with_graph_store(
    cost_resource_group: str = "gigl_test",
) -> gigl_resource_config_pb2.GiglResourceConfig:
    """Create a GiglResourceConfig with graph store training config for testing."""
    shared_resource_config = _create_shared_resource_config(cost_resource_group)

    # Define compute and storage pool configs for graph store
    compute_pool = gigl_resource_config_pb2.VertexAiResourceConfig(
        machine_type="n1-standard-16",
        gpu_type="NVIDIA_TESLA_V100",
        gpu_limit=4,
        num_replicas=1,
        gcp_region_override="us-west1",
        timeout=10800,
        scheduling_strategy="STANDARD",
    )
    storage_pool = gigl_resource_config_pb2.VertexAiResourceConfig(
        machine_type="n1-highmem-32",
        num_replicas=1,
    )
    graph_store_config = gigl_resource_config_pb2.VertexAiGraphStoreConfig(
        compute_pool=compute_pool,
        graph_store_pool=storage_pool,
        compute_cluster_local_world_size=4,
    )

    # Create TrainerResourceConfig with graph store config
    trainer_resource_config = gigl_resource_config_pb2.TrainerResourceConfig(
        vertex_ai_graph_store_trainer_config=graph_store_config,
    )

    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=shared_resource_config,
        trainer_resource_config=trainer_resource_config,
    )


def _create_gigl_resource_config_with_single_pool_inference(
    cost_resource_group: str = "gigl_test",
) -> gigl_resource_config_pb2.GiglResourceConfig:
    """Create a GiglResourceConfig with single pool CPU inference config for testing."""
    shared_resource_config = _create_shared_resource_config(cost_resource_group)

    # Define single pool CPU config for inference
    vertex_ai_config = gigl_resource_config_pb2.VertexAiResourceConfig(
        machine_type="n1-standard-8",
        num_replicas=1,
        timeout=7200,
    )

    # Create InferencerResourceConfig with single pool vertex AI config
    inferencer_resource_config = gigl_resource_config_pb2.InferencerResourceConfig(
        vertex_ai_inferencer_config=vertex_ai_config,
    )

    return gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=shared_resource_config,
        inferencer_resource_config=inferencer_resource_config,
    )


class TestVertexAILauncher(unittest.TestCase):
    """Test suite for vertex_ai_launcher module."""

    @patch("gigl.src.common.vertex_ai_launcher.VertexAIService")
    def test_launch_training_graph_store_cuda(self, mock_vertex_ai_service_class):
        """Test launching a training job with graph store enabled and GPU/CUDA configuration."""
        # Define test inputs
        job_name = "test-training-job"
        task_config_uri = Uri("gs://bucket/task_config.yaml")
        resource_config_uri = Uri("gs://bucket/resource_config.yaml")
        process_command = "python -m gigl.src.training.v2.glt_trainer"
        process_runtime_args = {"learning_rate": "0.001", "epochs": "10"}
        cpu_docker_uri = "gcr.io/project/cpu-image:tag"
        cuda_docker_uri = "gcr.io/project/cuda-image:tag"
        component = GiGLComponents.Trainer

        # Create GiglResourceConfig and wrapper
        gigl_resource_config_proto = _create_gigl_resource_config_with_graph_store(
            cost_resource_group="gigl_train"
        )
        resource_config_wrapper = GiglResourceConfigWrapper(
            resource_config=gigl_resource_config_proto
        )

        # Get the graph store config and its sub-configs for the launcher
        graph_store_config = (
            gigl_resource_config_proto.trainer_resource_config.vertex_ai_graph_store_trainer_config
        )
        compute_pool = graph_store_config.compute_pool
        storage_pool = graph_store_config.graph_store_pool
        shared_config = gigl_resource_config_proto.shared_resource_config

        # Setup mock service
        mock_service_instance = Mock()
        mock_vertex_ai_service_class.return_value = mock_service_instance

        launch_graph_store_enabled_job(
            vertex_ai_graph_store_config=graph_store_config,
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            process_command=process_command,
            process_runtime_args=process_runtime_args,
            resource_config_wrapper=resource_config_wrapper,
            storage_command="python -m gigl.distributed.graph_store.storage_main",
            storage_args={},
            cpu_docker_uri=cpu_docker_uri,
            cuda_docker_uri=cuda_docker_uri,
            component=component,
        )

        # Assert - verify VertexAIService was instantiated correctly
        mock_vertex_ai_service_class.assert_called_once_with(
            project=shared_config.common_compute_config.project,
            location=compute_pool.gcp_region_override,
            service_account=shared_config.common_compute_config.gcp_service_account_email,
            staging_bucket=shared_config.common_compute_config.temp_regional_assets_bucket,
        )

        # Verify launch_graph_store_job was called
        mock_service_instance.launch_graph_store_job.assert_called_once()

        # Verify the job configs
        call_args = mock_service_instance.launch_graph_store_job.call_args
        compute_job_config = call_args.kwargs["compute_pool_job_config"]
        storage_job_config = call_args.kwargs["storage_pool_job_config"]

        # Verify compute pool config uses CUDA
        self.assertEqual(compute_job_config.container_uri, cuda_docker_uri)
        self.assertEqual(compute_job_config.machine_type, compute_pool.machine_type)
        self.assertEqual(compute_job_config.accelerator_type, compute_pool.gpu_type)
        self.assertEqual(compute_job_config.accelerator_count, compute_pool.gpu_limit)
        self.assertEqual(compute_job_config.job_name, job_name)

        # Verify compute pool command and args
        self.assertEqual(
            compute_job_config.command,
            process_command.split(),
        )
        self.assertIsNotNone(compute_job_config.args)
        assert compute_job_config.args is not None  # Type narrowing for mypy
        self.assertIn(f"--job_name={job_name}", compute_job_config.args)
        self.assertIn(
            f"--learning_rate={process_runtime_args['learning_rate']}",
            compute_job_config.args,
        )
        self.assertIn(
            f"--epochs={process_runtime_args['epochs']}", compute_job_config.args
        )

        # Verify storage pool config
        self.assertEqual(storage_job_config.machine_type, storage_pool.machine_type)
        self.assertIn(
            "gigl.distributed.graph_store.storage_main",
            " ".join(storage_job_config.command),
        )

        # Verify environment variables
        compute_env_vars = {
            ev.name: ev.value for ev in compute_job_config.environment_variables
        }
        self.assertEqual(
            compute_env_vars["COMPUTE_CLUSTER_LOCAL_WORLD_SIZE"],
            str(graph_store_config.compute_cluster_local_world_size),
        )

        # Verify resource labels
        expected_labels = {
            "env": "test",
            "cost_resource_group_tag": "unittest_tra",
            "cost_resource_group": "gigl_train",
        }
        self.assertEqual(compute_job_config.labels, expected_labels)

    @patch("gigl.src.common.vertex_ai_launcher.VertexAIService")
    def test_launch_inference_single_pool_cpu(self, mock_vertex_ai_service_class):
        """Test launching an inference job with single pool and CPU-only configuration."""
        # Define test inputs
        job_name = "test-inference-job"
        task_config_uri = Uri("gs://bucket/inference_config.yaml")
        resource_config_uri = Uri("gs://bucket/resource_config.yaml")
        process_command = "python -m gigl.src.inference.v2.glt_inferencer"
        process_runtime_args = {
            "batch_size": "128",
            "output_path": "gs://bucket/output",
        }
        cpu_docker_uri = "gcr.io/project/cpu-image:tag"
        cuda_docker_uri = "gcr.io/project/cuda-image:tag"
        component = GiGLComponents.Inferencer
        vertex_ai_region = "us-central1"

        # Create GiglResourceConfig and wrapper
        gigl_resource_config_proto = (
            _create_gigl_resource_config_with_single_pool_inference(
                cost_resource_group="gigl_inference"
            )
        )
        resource_config_wrapper = GiglResourceConfigWrapper(
            resource_config=gigl_resource_config_proto
        )

        # Get the vertex AI config for the launcher
        vertex_ai_config = (
            gigl_resource_config_proto.inferencer_resource_config.vertex_ai_inferencer_config
        )
        shared_config = gigl_resource_config_proto.shared_resource_config

        # Setup mock service
        mock_service_instance = Mock()
        mock_vertex_ai_service_class.return_value = mock_service_instance

        launch_single_pool_job(
            vertex_ai_resource_config=vertex_ai_config,
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            process_command=process_command,
            process_runtime_args=process_runtime_args,
            resource_config_wrapper=resource_config_wrapper,
            cpu_docker_uri=cpu_docker_uri,
            cuda_docker_uri=cuda_docker_uri,
            component=component,
            vertex_ai_region=vertex_ai_region,
        )

        # Assert - verify VertexAIService was instantiated correctly
        mock_vertex_ai_service_class.assert_called_once_with(
            project=shared_config.common_compute_config.project,
            location=vertex_ai_region,
            service_account=shared_config.common_compute_config.gcp_service_account_email,
            staging_bucket=shared_config.common_compute_config.temp_regional_assets_bucket,
        )

        # Verify launch_job was called (not launch_graph_store_job)
        mock_service_instance.launch_job.assert_called_once()
        mock_service_instance.launch_graph_store_job.assert_not_called()

        # Verify the job config
        call_args = mock_service_instance.launch_job.call_args
        job_config = call_args.kwargs["job_config"]

        # Verify CPU execution settings
        self.assertEqual(job_config.container_uri, cpu_docker_uri)
        self.assertEqual(job_config.machine_type, vertex_ai_config.machine_type)
        self.assertEqual(job_config.job_name, job_name)
        self.assertEqual(job_config.accelerator_count, 0)
        self.assertEqual(job_config.accelerator_type, "")
        self.assertEqual(job_config.timeout_s, vertex_ai_config.timeout)
        self.assertEqual(job_config.replica_count, vertex_ai_config.num_replicas)

        # Verify command and args
        self.assertEqual(job_config.command, process_command.split())
        self.assertIsNotNone(job_config.args)
        assert job_config.args is not None  # Type narrowing for mypy
        self.assertIn(f"--job_name={job_name}", job_config.args)
        self.assertIn(f"--task_config_uri={task_config_uri}", job_config.args)
        self.assertIn(f"--resource_config_uri={resource_config_uri}", job_config.args)
        self.assertIn(
            f"--batch_size={process_runtime_args['batch_size']}", job_config.args
        )
        self.assertIn(
            f"--output_path={process_runtime_args['output_path']}", job_config.args
        )

        # Verify resource labels
        expected_labels = {
            "env": "test",
            "cost_resource_group_tag": "unittest_inf",
            "cost_resource_group": "gigl_inference",
        }
        self.assertEqual(job_config.labels, expected_labels)


if __name__ == "__main__":
    unittest.main()
