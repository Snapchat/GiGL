#!/bin/bash
# Install C++ tools for gigl-core.
#
# Usage:
#   bash gigl-core/requirements/install_cpp_deps.sh        # cmake only: build and test gigl-core
#   bash gigl-core/requirements/install_cpp_deps.sh --dev  # also the clang-15 format and lint tools
#
# The default install is for CI jobs that only build or test, so they download as little as possible
# from the apt mirror. The C++ compiler (GCC) is not installed here; it comes from the host image.
#
# `make install_dev_deps` passes --dev. It runs alongside install_py_deps.sh and install_scala_deps.sh.
#
# NOTE: macOS is not supported. C++ tooling requires GLT, which does not run on macOS.

set -e
set -x

DEV=0  # Flag to install the clang-15 format and lint tools.

for arg in "$@"
do
    case $arg in
    --dev)
        DEV=1
        shift
        ;;
    *)
        echo "ERROR: unknown argument: $arg" >&2
        exit 1
        ;;
    esac
done

if [ "$(uname)" == "Darwin" ]; then
    echo "ERROR: macOS is not supported for C++ tooling (GLT does not run on macOS)." >&2
    exit 1
fi

packages=(cmake)
if [ "$DEV" == "1" ]; then
    # The Makefile format and lint targets use these:
    #   - clang-format-15: format_cpp, check_format_cpp
    #   - clangd-15:       check_lint_cpp (via gigl-core/scripts/run_cpp_lint.py)
    #   - clang-tidy-15:   fix_lint_cpp
    # clang-tidy-15 depends on clang-15, so clang++-15 is installed too.
    # libstdc++-12-dev: on Ubuntu 22.04 the clang-15 tools look for GCC 12 headers, and without
    # this package they cannot find standard headers like <cstddef>.
    packages+=(clang-format-15 clangd-15 clang-tidy-15 libstdc++-12-dev)
fi

sudo apt-get update -y
# --no-install-recommends skips about 120 MB of packages that no target uses, such as llvm-15-dev
# and clang-15-doc. A slow apt mirror makes each extra megabyte cost CI time.
sudo apt-get install -y --no-install-recommends "${packages[@]}"

# Verify cmake >= 3.18 (our CMakeLists.txt requires it; Ubuntu 20.04 apt provides 3.16).
cmake_version=$(cmake --version | awk 'NR==1{print $3}')
if ! printf '3.18\n%s\n' "$cmake_version" | sort -V -C 2>/dev/null; then
    echo "ERROR: cmake >= 3.18 required, found $cmake_version. See https://cmake.org/download/" >&2
    exit 1
fi

echo "Finished installing C++ tooling"
