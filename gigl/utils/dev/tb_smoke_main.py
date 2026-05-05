"""Tiny smoke-test entrypoint that exercises GiGL's TensorBoard pipeline.

Submitted as the container command by ``tools/dev_submit_tb_smoke_job.py``.
On the chief rank, instantiates :class:`gigl.utils.tensorboard_writer.TensorBoardWriter`
via ``from_env``, writes a few scalar events, and sleeps long enough for both
TensorBoard uploaders (Vertex's built-in auto-uploader and our chief-rank
``aiplatform.start_upload_tb_log``) to flush before exit.

Usage:

    python -m gigl.utils.dev.tb_smoke_main

Reads no CLI flags. All configuration comes from env vars set by Vertex AI
and GiGL's launcher (``AIP_TENSORBOARD_LOG_DIR``, ``GIGL_TENSORBOARD_*``).
"""

from __future__ import annotations

import time

from gigl.common.logger import Logger
from gigl.utils.tensorboard_writer import TensorBoardWriter

logger = Logger()

_NUM_STEPS = 3
_FLUSH_SLEEP_SECS = 60


def main() -> None:
    """Write a handful of scalar events and wait for the uploaders to flush."""
    logger.info("Starting tb_smoke_main")
    with TensorBoardWriter.from_env(enabled=True) as writer:
        for step in range(_NUM_STEPS):
            writer.log({"smoke/value": float(step)}, step=step)
            logger.info(f"Wrote smoke/value={step} at step {step}")
        logger.info(
            f"Sleeping {_FLUSH_SLEEP_SECS}s to let TensorBoard uploaders flush "
            "events to GCS + Vertex AI"
        )
        time.sleep(_FLUSH_SLEEP_SECS)
    logger.info("tb_smoke_main complete")


if __name__ == "__main__":
    main()
