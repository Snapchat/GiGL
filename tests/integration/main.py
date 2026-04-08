import sys
from typing import Final

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri
from gigl.common.utils.test_utils import parse_args, run_tests
from gigl.src.common.utils.metrics_service_provider import initialize_metrics
from tests.test_assets.uri_constants import DEFAULT_NABLP_TASK_CONFIG_URI

# Slow test modules that must be spread across shards. Position in the tuple
# determines the shard: ``index % total_shards``.  **Append-only** — never
# reorder existing entries, or every module's shard assignment will shift.
#
# Durations measured 2026-02-27 (unsharded CI run, 77.5 min total):
INTEGRATION_TEST_SHARD_PINNED_MODULES: Final[tuple[str, ...]] = (
    "tests.integration.distributed.distributed_dataset_test",  # 14.5 min (18.7%)
    "tests.integration.distributed.utils.networking_test",  # 13.3 min (17.2%)
    "tests.integration.distributed.graph_store.graph_store_integration_test",  # 13.0 min (16.8%)
    "tests.integration.pipeline.data_preprocessor.data_preprocessor_pipeline_test",  # 11.7 min (15.1%)
    "tests.integration.pipeline.subgraph_sampler.subgraph_sampler_test",  # 8.8 min (11.4%)
    "tests.integration.common.services.vertex_ai_test",  # 6.5 min  (8.4%)
    "tests.integration.pipeline.split_generator.split_generator_pipeline_test",  # 3.8 min  (5.0%)
    "tests.integration.pipeline.inferencer.inferencer_test",  # 2.1 min  (2.8%)
)


def run(
    pattern: str = "*_test.py",
    shard_index: int = 0,
    total_shards: int = 0,
) -> bool:
    initialize_metrics(
        task_config_uri=DEFAULT_NABLP_TASK_CONFIG_URI, service_name="integration_test"
    )
    return run_tests(
        start_dir=LocalUri.join(
            local_fs_constants.get_project_root_directory(), "tests", "integration"
        ),
        pattern=pattern,
        use_sequential_execution=True,
        shard_index=shard_index,
        total_shards=total_shards,
        pinned_modules=INTEGRATION_TEST_SHARD_PINNED_MODULES,
    )


if __name__ == "__main__":
    test_args = parse_args()
    was_successful: bool = run(
        pattern=test_args.test_file_pattern,
        shard_index=test_args.shard_index,
        total_shards=test_args.total_shards,
    )
    sys.exit(not was_successful)
