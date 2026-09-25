#!/bin/bash
# Install-and-import smoke for a non-primary Python minor. `make install_dev_deps` has already
# run under UV_PYTHON=<minor>; this asserts what it produced and runs hermetic unit files.
set -euo pipefail
minor="${1:?usage: smoke_test_python_minor.sh <3.11|3.12>}"
test "${UV_PYTHON:-}" = "${minor}" || { echo "UV_PYTHON=${UV_PYTHON:-unset}, expected ${minor}"; exit 1; }
# Interpreter version, SOABI, active venv, and the native per-minor wheels import. torchrec
# imports fbgemm_gpu; the CPU build installed on CPU runners does not need libcuda.
uv run python scripts/smoke_test_image.py --python "${minor}" --imports torch,graphlearn_torch,gigl_core,torchrec
uv run python -c "import gigl, tensorflow, apache_beam, tensorflow_transform, tensorflow_data_validation, hydra"
# The test runner is called directly rather than through `make unit_test_py`, which type-checks the
# whole repository first; that belongs to the lint job, not an install check.
# tests/unit/main.py selects whole files by name, so only hermetic files (no GCS reads) are listed.
run_unit_test_file() {
    uv run python -m tests.unit.main --env=test \
        --resource_config_uri=deployment/configs/unittest_resource_config.yaml \
        --test_file_pattern="$1"
}
# tf.data.TFRecordDataset + tf.io.parse_example over tempfile TFRecords.
run_unit_test_file tf_records_iterable_dataset_test.py
# TFT tft_unit on the Beam DirectRunner: the transform extra's surface.
run_unit_test_file data_preprocessor_config_test.py
# str_to_bool parity with distutils' strtobool.
run_unit_test_file parse_test.py
