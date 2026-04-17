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
    return $?
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
            echo "Added LLVM bin to PATH in $rc_file"
        fi
    done

    # NOTE: this export only affects subprocesses of this script, not the calling
    # shell or make process. Open a new terminal (or run `source ~/.zshrc`) after
    # install_dev_deps to pick up the PATH change.
    export PATH="$LLVM_BIN:$PATH"
else
    # On Linux, apt-get installs versioned binaries (e.g. clang-format-15) directly
    # into /usr/bin. No PATH changes are needed since /usr/bin is already on PATH.
    # Callers use the versioned names (clang-format-15, clang-tidy-15, clangd-15)
    # directly so the version is explicit and greppable across the codebase.
    apt-get install -y clang-format-15 clang-tidy-15 clangd-15 cmake
fi

echo "Finished installing C++ tooling"
