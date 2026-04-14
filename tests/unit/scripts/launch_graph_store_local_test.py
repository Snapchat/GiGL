"""Unit tests for scripts.launch_graph_store_local."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from scripts import launch_graph_store_local as launcher
from tests.test_assets.test_case import TestCase


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class TestLaunchGraphStoreLocal(TestCase):
    """Test suite for the local graph-store launcher."""

    def tearDown(self) -> None:
        self.kill_all_children_processes()
        super().tearDown()

    def test_load_component_config_train_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_config_path = Path(tmpdir) / "task.yaml"
            _write_yaml(
                task_config_path,
                {
                    "trainerConfig": {
                        "command": "pkg.train",
                        "trainerArgs": {"epochs": 2, "lr": 0.1},
                        "graphStoreStorageConfig": {
                            "command": "pkg.storage",
                            "storageArgs": {"num_server_sessions": 4},
                        },
                    }
                },
            )

            component_config = launcher.load_component_config(
                task_config_uri=task_config_path.as_posix(),
                mode="train",
            )

            self.assertEqual(component_config["name"], "trainerConfig")
            self.assertEqual(component_config["compute_command"], "pkg.train")
            self.assertEqual(component_config["storage_command"], "pkg.storage")
            self.assertEqual(
                component_config["compute_args"], {"epochs": "2", "lr": "0.1"}
            )
            self.assertEqual(
                component_config["storage_args"], {"num_server_sessions": "4"}
            )

    def test_load_component_config_infer_mode_accepts_snake_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_config_path = Path(tmpdir) / "task.yaml"
            _write_yaml(
                task_config_path,
                {
                    "inferencer_config": {
                        "command": "pkg.infer",
                        "inference_args": {"batch_size": 64},
                        "graph_store_storage_config": {
                            "command": "pkg.storage",
                            "storage_args": {"foo": "bar"},
                        },
                    }
                },
            )

            component_config = launcher.load_component_config(
                task_config_uri=task_config_path.as_posix(),
                mode="infer",
            )

            self.assertEqual(component_config["name"], "inferencerConfig")
            self.assertEqual(component_config["compute_command"], "pkg.infer")
            self.assertEqual(component_config["storage_command"], "pkg.storage")
            self.assertEqual(component_config["compute_args"], {"batch_size": "64"})
            self.assertEqual(component_config["storage_args"], {"foo": "bar"})

    def test_load_component_config_requires_compute_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_config_path = Path(tmpdir) / "task.yaml"
            _write_yaml(
                task_config_path,
                {
                    "trainerConfig": {
                        "graphStoreStorageConfig": {"command": "pkg.storage"}
                    }
                },
            )

            with self.assertRaisesRegex(
                ValueError, "Task config trainerConfig has no 'command' field"
            ):
                launcher.load_component_config(
                    task_config_uri=task_config_path.as_posix(),
                    mode="train",
                )

    def test_load_component_config_requires_storage_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_config_path = Path(tmpdir) / "task.yaml"
            _write_yaml(
                task_config_path,
                {
                    "trainerConfig": {
                        "command": "pkg.train",
                        "graphStoreStorageConfig": {},
                    }
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "Task config trainerConfig.graphStoreStorageConfig has no 'command' field",
            ):
                launcher.load_component_config(
                    task_config_uri=task_config_path.as_posix(),
                    mode="train",
                )

    def test_resolve_input_uri_handles_local_and_remote_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "task.yaml"
            local_path.write_text("trainerConfig: {}\n", encoding="utf-8")

            original_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                self.assertEqual(
                    launcher.resolve_input_uri("task.yaml", "--task-config-uri"),
                    local_path.resolve().as_posix(),
                )
            finally:
                os.chdir(original_cwd)

        self.assertEqual(
            launcher.resolve_input_uri("gs://bucket/task.yaml", "--task-config-uri"),
            "gs://bucket/task.yaml",
        )

    def test_assign_gpus_supports_cpu_only_and_validates_capacity(self) -> None:
        self.assertEqual(
            launcher._assign_gpus(
                compute_nodes=2,
                gpus_per_compute_node=0,
                gpu_ids=None,
            ),
            [[], []],
        )

        with self.assertRaisesRegex(
            ValueError, "Requested 4 GPUs, but only 3 are visible"
        ):
            launcher._assign_gpus(
                compute_nodes=2,
                gpus_per_compute_node=2,
                gpu_ids=[0, 1, 2],
            )

    def test_assign_gpus_errors_when_none_are_visible(self) -> None:
        with patch.object(launcher, "_detect_visible_gpus", return_value=[]):
            with self.assertRaisesRegex(
                ValueError,
                "No CUDA accelerators are visible, but --gpus-per-compute-node was set",
            ):
                launcher._assign_gpus(
                    compute_nodes=1,
                    gpus_per_compute_node=1,
                    gpu_ids=None,
                )

    def test_build_cluster_spec_and_node_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            nodes = launcher.build_node_processes(
                storage_nodes=2,
                compute_nodes=2,
                python_bin=sys.executable,
                job_id="job-123",
                task_config_uri="/tmp/task.yaml",
                resource_config_uri="/tmp/resource.yaml",
                component_config={
                    "name": "trainerConfig",
                    "storage_command": "pkg.storage",
                    "compute_command": "pkg.train",
                    "storage_args": {"storage_flag": "1"},
                    "compute_args": {"compute_flag": "2"},
                },
                job_dir=job_dir,
                base_env={"BASE_ENV": "1"},
                gpu_assignments=[[0], [1]],
            )

            self.assertEqual(
                [node.name for node in nodes],
                [
                    "storage_node_0",
                    "storage_node_1",
                    "compute_node_0",
                    "compute_node_1",
                ],
            )
            self.assertEqual([node.rank for node in nodes], [2, 3, 0, 1])
            self.assertEqual(nodes[0].env["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(nodes[2].env["CUDA_VISIBLE_DEVICES"], "0")
            self.assertIn("--storage_flag=1", nodes[0].command)
            self.assertIn("--compute_flag=2", nodes[2].command)
            self.assertEqual(
                nodes[2].cluster_spec["task"],
                {"type": "workerpool0", "index": 0},
            )
            self.assertEqual(
                nodes[3].cluster_spec["task"],
                {"type": "workerpool1", "index": 0},
            )
            self.assertEqual(
                nodes[0].cluster_spec["task"],
                {"type": "workerpool2", "index": 0},
            )

    def test_main_writes_reduced_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            task_config_path = tmp_path / "task.yaml"
            resource_config_path = tmp_path / "resource.yaml"
            _write_yaml(
                task_config_path,
                {
                    "trainerConfig": {
                        "command": "pkg.train",
                        "trainerArgs": {"epochs": 3},
                        "graphStoreStorageConfig": {
                            "command": "pkg.storage",
                            "storageArgs": {"num_server_sessions": 1},
                        },
                    }
                },
            )
            resource_config_path.write_text("project: test\n", encoding="utf-8")

            captured: dict[str, Any] = {}

            def _fake_launch_processes(
                *,
                nodes: list[launcher.NodeProcess],
                cwd: Path,
                job_dir: Path,
                launch_delay_s: float,
            ) -> int:
                captured["nodes"] = nodes
                captured["cwd"] = cwd
                captured["job_dir"] = job_dir
                captured["launch_delay_s"] = launch_delay_s
                return 0

            with (
                patch.object(launcher, "_get_free_ports", return_value=[23456]),
                patch.object(
                    launcher,
                    "_launch_processes",
                    side_effect=_fake_launch_processes,
                ),
            ):
                result = launcher.main(
                    [
                        "--storage-nodes=1",
                        "--compute-nodes=1",
                        "--compute-procs-per-node=1",
                        "--mode=train",
                        f"--task-config-uri={task_config_path.as_posix()}",
                        f"--resource-config-uri={resource_config_path.as_posix()}",
                        "--job-id=test-job",
                        f"--job-root={tmp_path.as_posix()}",
                        f"--cwd={tmp_path.as_posix()}",
                        "--launch-delay-s=1.5",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(captured["cwd"], tmp_path)
            self.assertEqual(captured["launch_delay_s"], 1.5)
            self.assertLen(captured["nodes"], 2)

            manifest_path = tmp_path / "gs-job-test-job" / "launcher_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "train")
            self.assertEqual(manifest["host"], "127.0.0.1")
            self.assertEqual(manifest["topology"]["storage_nodes"], 1)
            self.assertEqual(manifest["topology"]["compute_nodes"], 1)
            self.assertEqual(manifest["gpu_assignments"], [[]])
            self.assertEqual(manifest["component"]["name"], "trainerConfig")
            self.assertNotIn("task_config_component", manifest)
            self.assertNotIn("storage_runtime_args", manifest)
            self.assertEqual(manifest["processes"][0]["role"], "storage")

    def test_launch_processes_returns_zero_when_all_children_succeed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            nodes = [
                launcher.NodeProcess(
                    name="compute_node_0",
                    role="compute",
                    rank=0,
                    command=[sys.executable, "-c", "print('ready')"],
                    env=os.environ.copy(),
                    log_path=tmp_path / "compute_node_0.log",
                    cluster_spec={},
                )
            ]

            return_code = launcher._launch_processes(
                nodes=nodes,
                cwd=tmp_path,
                job_dir=tmp_path,
                launch_delay_s=0.0,
            )

            self.assertEqual(return_code, 0)
            self.assertIn("ready", (tmp_path / "compute_node_0.log").read_text())

    def test_launch_processes_returns_failure_code_and_shuts_down_peers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            nodes = [
                launcher.NodeProcess(
                    name="storage_node_0",
                    role="storage",
                    rank=1,
                    command=[sys.executable, "-c", "import time; time.sleep(30)"],
                    env=os.environ.copy(),
                    log_path=tmp_path / "storage_node_0.log",
                    cluster_spec={},
                ),
                launcher.NodeProcess(
                    name="compute_node_0",
                    role="compute",
                    rank=0,
                    command=[
                        sys.executable,
                        "-c",
                        "import sys; print('boom'); sys.exit(7)",
                    ],
                    env=os.environ.copy(),
                    log_path=tmp_path / "compute_node_0.log",
                    cluster_spec={},
                ),
            ]

            return_code = launcher._launch_processes(
                nodes=nodes,
                cwd=tmp_path,
                job_dir=tmp_path,
                launch_delay_s=0.0,
            )

            self.assertEqual(return_code, 7)
            self.assertIn("boom", (tmp_path / "compute_node_0.log").read_text())
