#!/bin/bash
# Builds a patched GraphLearn-Torch wheel and copies it into --output-dir.
#
# Build-only: caching, installing and verifying the wheel live in gigl/scripts/post_install.py, which
# is the entry point to use. Exits 0 without building on macOS.
#
# The clone and build happen inside --output-dir, so a failed build leaves nothing in the caller's
# directory. post_install.py passes a temp dir under $TMPDIR that it removes; set TMPDIR if /tmp is small
# (for example a tmpfs).
#
# Flags:
#   --commit           Upstream alibaba/graphlearn-for-pytorch sha to build. Required; post_install.py
#                      passes the GLT_COMMIT from glt_wheel_cache.py.
#   --cuda-arch-list   TORCH_CUDA_ARCH_LIST for a CUDA build, e.g. "7.5" (see the list of values next to
#                      GLT_CUDA_ARCH_LIST in post_install.py). Empty or omitted builds for CPU
#                      (WITH_CUDA=OFF).
#   --output-dir       Directory the built wheel is copied into. Required.
#
# This script builds exactly the variant it is told. It does not detect CUDA itself: the CPU/CUDA
# variant is part of the wheel cache key, so post_install.py decides it once, before computing the key.

set -e
set -x

if [ "$(uname)" == "Darwin" ]; then
    echo "Skipping build of GraphLearn-Torch on Mac"
    exit 0
fi

COMMIT=""
CUDA_ARCH_LIST=""
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit) COMMIT="$2"; shift 2 ;;
        --cuda-arch-list) CUDA_ARCH_LIST="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
if [[ -z "${COMMIT}" || -z "${OUTPUT_DIR}" ]]; then
    echo "--commit and --output-dir are required"
    exit 1
fi

# Resolve this script's directory BEFORE cd'ing, so the patch path survives the cd below.
GIGL_SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$( cd -- "${OUTPUT_DIR}" && pwd )"

# Without Ninja, torch's cpp_extension builds sequentially which is very slow. Ninja comes from the
# `glt-build-backend` dependency group, whose binary lives in the venv's bin dir.
export PATH="$(dirname "$(command -v python)"):${PATH}"
command -v ninja >/dev/null 2>&1 || {
    echo "FATAL: ninja not found on PATH. The GLT build needs it; install with 'uv sync --group glt-build-backend' in a dev checkout, or 'pip install ninja'."
    exit 1
}

echo "Building GraphLearn-Torch"
cd "${OUTPUT_DIR}"
# A direct run can reuse --output-dir; clear any earlier clone so git clone succeeds.
rm -rf graphlearn-for-pytorch
git clone https://github.com/alibaba/graphlearn-for-pytorch.git
cd graphlearn-for-pytorch
git checkout "${COMMIT}"
git submodule update --init
bash install_dependencies.sh
# TODO (dsaini2-sc): once a Snapchat-maintained GLT fork exists, land these patches as
# commits on the fork, pin to it, and retire this apply loop.
# Local patches applied on top of the pinned commit, in order. The CSR patch replaces the
# at::_unique distinct-count in InitCPUGraphFromCSR with an exact bitmap count: _unique's
# sort allocates ~3x the size of `indices` transiently (measured 3.00x at 200M edges), which
# is ~94 GiB at a 4.2B-edge partition and OOM-kills the graph init on hosts whose budget
# assumes init holds no copy of the topology. `patch` rather than `git apply` so failure
# modes are the familiar reject files; `--forward` + explicit failure so a re-run or an
# upstream change that absorbs the fix stops the build LOUDLY instead of shipping an
# unpatched wheel that dies 3 hours into a 16-GPU job.
for glt_patch in "${GIGL_SCRIPTS_DIR}"/patches/*.patch; do
    echo "Applying ${glt_patch}"
    patch -p1 --forward < "${glt_patch}" || { echo "FATAL: ${glt_patch} did not apply"; exit 1; }
done
if [[ -n "${CUDA_ARCH_LIST}" ]]; then
    echo "Will use CUDA for GLT..."
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" WITH_CUDA="ON" python setup.py bdist_wheel
else
    echo "Will use CPU for GLT..."
    WITH_CUDA="OFF" python setup.py bdist_wheel
fi
cp dist/*.whl "${OUTPUT_DIR}/"
cd ..
rm -rf graphlearn-for-pytorch
