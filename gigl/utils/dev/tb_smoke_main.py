"""Tiny smoke-test entrypoint that exercises GiGL's TensorBoard pipeline.

Submitted as the container command by ``submit_smoke_job.py``. Constructs a
``TensorBoardWriter`` with ``enabled=True`` (single-process smoke = always
chief), writes a few scalar events, and exits.

Configuration is plumbed via CLI flags injected by the launcher from the
smoke script's ``process_runtime_args`` map. All three are required:

    --job_name=<used as the TensorboardRun ID>
    --tensorboard_resource_name=<full Vertex AI Tensorboard resource name>
    --tensorboard_experiment_name=<TensorboardExperiment ID under that resource>

This entrypoint deliberately mirrors the production trainer/inferencer call
sites in ``examples/link_prediction/`` so the smoke test exercises the same
``TensorBoardWriter.create()`` code path.
"""

from __future__ import annotations

import argparse

from gigl.common.logger import Logger
from gigl.utils.tensorboard_writer import TensorBoardWriter

logger = Logger()

_NUM_STEPS = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job_name",
        required=True,
        help="Used as the TensorboardRun ID (must be unique per launch).",
    )
    parser.add_argument(
        "--tensorboard_resource_name",
        required=True,
        help="Full Vertex AI Tensorboard resource name.",
    )
    parser.add_argument(
        "--tensorboard_experiment_name",
        required=True,
        help="TensorboardExperiment ID under the resource above.",
    )
    # The launcher's _build_job_config always appends --task_config_uri,
    # --resource_config_uri, and (on GPU) --use_cuda. The smoke entrypoint
    # doesn't need them; use parse_known_args so they don't blow up argparse.
    args, _unrecognized = parser.parse_known_args()
    return args


def main() -> None:
    """Write a handful of scalar events and exit."""
    args = _parse_args()
    logger.info(f"Starting tb_smoke_main; job_name={args.job_name!r}")
    with TensorBoardWriter.create(
        resource_name=args.tensorboard_resource_name,
        experiment_name=args.tensorboard_experiment_name,
        experiment_run_name=args.job_name,
        enabled=True,
    ) as writer:
        for step in range(_NUM_STEPS):
            writer.log({"smoke/value": float(step)}, step=step)
            logger.info(f"Wrote smoke/value={step} at step {step}")
    logger.info("tb_smoke_main complete")


if __name__ == "__main__":
    main()
