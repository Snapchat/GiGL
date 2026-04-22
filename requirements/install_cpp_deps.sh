#!/bin/bash
# Install C++ development tools: clang-format, clang-tidy, cmake.
#
# Usage:
#   bash requirements/install_cpp_deps.sh
#
# Called by `make install_dev_deps` alongside install_py_deps.sh and
# install_scala_deps.sh.
#
# NOTE: On Linux, this script calls apt-get, which requires root privileges.
# Run as root or prefix with sudo.

set -e
set -x

is_running_on_mac() {
    [ "$(uname)" == "Darwin" ]
}

if is_running_on_mac; then
    # macOS ships its own Apple Clang via Xcode Command Line Tools. Homebrew
    # intentionally does not put its llvm binaries on PATH to avoid shadowing
    # Apple's clang. We therefore have to add the Homebrew llvm bin directory
    # to PATH ourselves so that `clang-format-15` and `clang-tidy-15` resolve
    # to the correct versions rather than being missing entirely.
    brew install llvm@15 cmake
    LLVM_BIN="$(brew --prefix llvm@15)/bin"

    # Append to any shell rc files that exist and don't already include it.
    for rc_file in ~/.zshrc ~/.bashrc; do
        if [ -f "$rc_file" ] && ! grep -qF "$LLVM_BIN" "$rc_file"; then
            printf '\n# Added by GiGL install_cpp_deps.sh\nexport PATH="%s:$PATH"\n' "$LLVM_BIN" >> "$rc_file"
            echo "NOTE: Added LLVM bin to PATH in $rc_file."
            echo "      Open a new terminal (or run: source $rc_file) to pick up the change."
        fi
    done

    export PATH="$LLVM_BIN:$PATH"
else
    # On Linux, apt-get installs versioned binaries (e.g. clang-format-15) directly
    # into /usr/bin. No PATH changes are needed since /usr/bin is already on PATH.
    # Callers use the versioned names (clang-format-15, clang-tidy-15, clangd-15)
    # directly so the version is explicit and greppable across the codebase.
    # clang++-15 requires libstdc++-12-dev: on Ubuntu 22.04, clang++-15 looks for GCC 12
    # headers. Without this package clang++-15 cannot find standard headers like <cstddef>.
    # clang++-15 itself is needed because generate_compile_commands.py rewrites
    # compile_commands.json to use it so clangd natively understands the commands.
    apt-get update -y
    apt-get install -y clang-format-15 clang-tidy-15 clangd-15 clang++-15 libstdc++-12-dev cmake

    # Verify cmake >= 3.18 (our CMakeLists.txt requires it; Ubuntu 20.04 apt provides 3.16).
    # sort -V -C exits 0 if the two lines are already in ascending version order
    # (i.e. 3.18 <= cmake_version), non-zero otherwise.
    cmake_version=$(cmake --version | awk 'NR==1{print $3}')
    if ! printf '3.18\n%s\n' "$cmake_version" | sort -V -C 2>/dev/null; then
        echo "ERROR: cmake >= 3.18 required, found $cmake_version. See https://cmake.org/download/" >&2
        exit 1
    fi
fi

echo "Finished installing C++ tooling"
