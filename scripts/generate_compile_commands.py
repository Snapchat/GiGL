"""Generate .cache/compile_commands.json for clangd.

Delegates to CMake (which already knows all include paths and compiler flags via
find_package(Torch)) rather than manually constructing the database.

The build uses the system C++ compiler (g++) so that cmake, nvcc, and Torch's
cmake work without issues. After cmake writes compile_commands.json, the
compiler in each entry is replaced with ``clang++-15`` so that clangd natively
understands the commands without needing a ``--query-driver`` workaround.

Primary use: called by ``run_cpp_lint.py`` before running clangd checks, and
by ``make generate_compile_commands`` when you need to refresh the database
manually (e.g. after adding new source files or changing compiler flags).

Usage::

    make generate_compile_commands
"""

import json
import subprocess
from pathlib import Path

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_CMAKE_BUILD_DIR: Path = _REPO_ROOT / ".cache" / "cmake_build_lint"
_COMPILE_COMMANDS: Path = _REPO_ROOT / ".cache" / "compile_commands.json"


def write_compile_commands() -> None:
    """Run CMake to generate .cache/compile_commands.json."""
    _CMAKE_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(_REPO_ROOT),
            "-B",
            str(_CMAKE_BUILD_DIR),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ],
        check=True,
    )

    raw_path = _CMAKE_BUILD_DIR / "compile_commands.json"
    entries: list[dict] = json.loads(raw_path.read_text())

    # Replace the compiler for .cpp entries with clang++-15 so clangd uses
    # clang-native implicit include paths instead of guessing GCC's.
    # Leave .cu entries unchanged so nvcc handles them correctly.
    for entry in entries:
        if not entry.get("file", "").endswith(".cu"):
            tokens = entry["command"].split()
            tokens[0] = "clang++-15"
            entry["command"] = " ".join(tokens)

    _COMPILE_COMMANDS.write_text(json.dumps(entries, indent=2))


def main() -> None:
    write_compile_commands()
    print(f"Wrote {_COMPILE_COMMANDS}")


if __name__ == "__main__":
    main()
