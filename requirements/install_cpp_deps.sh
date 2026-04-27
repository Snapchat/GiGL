#!/bin/bash
# Install C++ development tools: clang-format, clang-tidy, cmake.
#
# Usage:
#   bash requirements/install_cpp_deps.sh
#
# Called by `make install_dev_deps` alongside install_py_deps.sh and
# install_scala_deps.sh.
#
# NOTE: macOS is not supported. C++ tooling requires GLT, which does not run on macOS.

set -e
set -x

if [ "$(uname)" == "Darwin" ]; then
    echo "ERROR: macOS is not supported for C++ tooling (GLT does not run on macOS)." >&2
    exit 1
fi

# clang++-15 requires libstdc++-12-dev: on Ubuntu 22.04, clang++-15 looks for GCC 12
# headers. Without this package clang++-15 cannot find standard headers like <cstddef>.
sudo apt-get update -y
sudo apt-get install -y clang-format-15 clang-tidy-15 clangd-15 clang++-15 libstdc++-12-dev cmake

# Verify cmake >= 3.18 (our CMakeLists.txt requires it; Ubuntu 20.04 apt provides 3.16).
cmake_version=$(cmake --version | awk 'NR==1{print $3}')
if ! printf '3.18\n%s\n' "$cmake_version" | sort -V -C 2>/dev/null; then
    echo "ERROR: cmake >= 3.18 required, found $cmake_version. See https://cmake.org/download/" >&2
    exit 1
fi

echo "Finished installing C++ tooling"
