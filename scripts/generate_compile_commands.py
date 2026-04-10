"""Generate build/compile_commands.json for clang-tidy analysis of GiGL C++ extensions.

clang-tidy needs a compilation database to resolve include paths and compiler flags.
This script derives those paths directly from the installed torch and pybind11 packages,
avoiding the need for `bear` or a separate CMake build of the extension.

Usage::

    uv run python scripts/generate_compile_commands.py

Output: ``build/compile_commands.json`` (created or overwritten).

Note: run ``make build_cpp_extensions`` before this script (or use ``make lint_cpp``,
which does both in the correct order) so the database reflects the current build state.
"""

import json
import sys
import sysconfig

from torch.utils.cpp_extension import include_paths as torch_include_paths

from scripts.cpp_build_constants import COMPILE_ARGS, CSRC_DIR, REPO_ROOT


def main() -> None:
    # Collect all include directories needed to compile the extension.
    # torch_include_paths() returns the torch headers, which already bundle
    # pybind11 under torch/include/pybind11/ — no separate pybind11 import needed.
    include_flags: list[str] = [f"-I{path}" for path in torch_include_paths()]
    # Python C API headers (e.g. Python.h) required by pybind11.
    include_flags.append(f"-I{sysconfig.get_path('include')}")

    cpp_sources = sorted(CSRC_DIR.rglob("*.cpp")) if CSRC_DIR.exists() else []
    if not cpp_sources:
        print(f"Warning: no .cpp files found under {CSRC_DIR}", file=sys.stderr)

    cxx_flags = " ".join(COMPILE_ARGS)

    # Each entry in compile_commands.json describes how one source file is compiled.
    # clang-tidy reads this to reproduce the exact compilation environment.
    commands: list[dict[str, str]] = [
        {
            "directory": str(REPO_ROOT),
            "file": str(source),
            "command": f"c++ {cxx_flags} {' '.join(include_flags)} -c {source}",
        }
        for source in cpp_sources
    ]

    output = REPO_ROOT / "build" / "compile_commands.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(commands, indent=2))
    print(
        f"Wrote {len(commands)} entr{'y' if len(commands) == 1 else 'ies'} to {output}"
    )


if __name__ == "__main__":
    main()
