#!/bin/bash
# Install C++ development tools: clang-format, clang-tidy, cmake.
#
# Usage:
#   bash requirements/install_cpp_deps.sh
#
# Called by `make install_dev_deps` alongside install_py_deps.sh and
# install_scala_deps.sh.

set -e
set -x

is_running_on_mac() {
    [ "$(uname)" == "Darwin" ]
    return $?
}

if is_running_on_mac; then
    # `brew install llvm` provides clang-format and clang-tidy.
    # Homebrew does not add llvm to PATH by default to avoid shadowing Apple's
    # clang, so we print an instruction for the developer to do it manually.
    brew install llvm cmake
    LLVM_PREFIX=$(brew --prefix llvm)
    set +x
    echo ""
    echo "NOTE: Add the LLVM bin directory to your PATH to use clang-format and clang-tidy:"
    echo "      export PATH=\"${LLVM_PREFIX}/bin:\$PATH\""
    echo "      (Add this to your ~/.zshrc or ~/.bashrc to make it permanent.)"
    echo ""
    set -x
else
    # Ubuntu / Debian — clang 15 is the highest version available on Ubuntu 22.04.
    apt-get install -y clang-format-15 clang-tidy-15 cmake
    # Register versioned binaries as the default so bare `clang-format` and
    # `clang-tidy` resolve to them without callers specifying the version suffix.
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-15 100
    update-alternatives --install /usr/bin/clang-tidy   clang-tidy   /usr/bin/clang-tidy-15   100
fi

echo "Finished installing C++ tooling"
