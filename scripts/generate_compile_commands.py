"""Generate build/compile_commands.json for clang-tidy analysis of GiGL C++ extensions.

clang-tidy needs a compilation database to resolve include paths and compiler flags.
This script derives those paths directly from the installed torch and pybind11 packages,
avoiding the need for `bear` or a separate CMake build of the extension.

Usage::

    uv run python scripts/generate_compile_commands.py

Output: ``build/compile_commands.json`` (created or overwritten).
"""

import json
import sys
import sysconfig
from pathlib import Path


def main() -> None:
    try:
        import pybind11
        import torch  # noqa: F401 — imported to verify it is installed
        from torch.utils.cpp_extension import include_paths as torch_include_paths
    except ImportError as exc:
        print(
            f"Error: {exc}\n"
            "Run `make build_cpp_extensions` first to ensure torch and pybind11 are available.",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_root = Path(__file__).parent.parent.resolve()

    # Collect all include directories needed to compile the extension.
    include_flags: list[str] = []
    for path in torch_include_paths():
        include_flags.append(f"-I{path}")
    include_flags.append(f"-I{pybind11.get_include()}")
    # Python C API headers (e.g. Python.h) required by pybind11.
    include_flags.append(f"-I{sysconfig.get_path('include')}")

    cpp_sources = sorted((repo_root / "gigl").rglob("*.cpp"))
    if not cpp_sources:
        print("Warning: no .cpp files found under gigl/", file=sys.stderr)

    # Each entry in compile_commands.json describes how one source file is compiled.
    # clang-tidy reads this to reproduce the exact compilation environment.
    commands: list[dict[str, str]] = [
        {
            "directory": str(repo_root),
            "file": str(source),
            "command": (
                f"c++ -std=c++17 -Wall -Wextra "
                f"{' '.join(include_flags)} "
                f"-c {source}"
            ),
        }
        for source in cpp_sources
    ]

    output = repo_root / "build" / "compile_commands.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(commands, indent=2))
    print(f"Wrote {len(commands)} entr{'y' if len(commands) == 1 else 'ies'} to {output}")


if __name__ == "__main__":
    main()
