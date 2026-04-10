#!/bin/bash
# Install C++ development tools: clang-format, clang-tidy, cmake.
#
# Usage:
#   bash requirements/install_cpp_deps.sh
#
# Called by `make install_dev_deps` alongside install_py_deps.sh and
# install_scala_deps.sh.
#
# NOTE: On Linux, this script calls apt-get and update-alternatives, which
# require root privileges. Run as root or prefix with sudo.

set -e
set -x

CLANG_VERSION=15

is_running_on_mac() {
    [ "$(uname)" == "Darwin" ]
    return $?
}

if is_running_on_mac; then
    # macOS ships its own Apple Clang via Xcode Command Line Tools. Homebrew
    # intentionally does not put its llvm binaries on PATH to avoid shadowing
    # Apple's clang. We therefore have to add the Homebrew llvm bin directory
    # to PATH ourselves so that `clang-format` and `clang-tidy` resolve to the
    # Homebrew versions rather than being missing entirely.
    brew install llvm cmake
    LLVM_BIN="$(brew --prefix llvm)/bin"

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
    # into /usr/bin. update-alternatives wires up the bare names (clang-format,
    # clang-tidy) so callers don't need to specify the version suffix. No PATH
    # changes are needed since /usr/bin is already on PATH.
    apt-get install -y "clang-format-${CLANG_VERSION}" "clang-tidy-${CLANG_VERSION}" cmake
    update-alternatives --install /usr/bin/clang-format clang-format "/usr/bin/clang-format-${CLANG_VERSION}" 100
    update-alternatives --install /usr/bin/clang-tidy   clang-tidy   "/usr/bin/clang-tidy-${CLANG_VERSION}"   100
fi

echo "Finished installing C++ tooling"
