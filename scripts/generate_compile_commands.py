"""Generate .cache/compile_commands.json for clangd.

Delegates to CMake (which already knows all include paths and compiler flags via
find_package(Torch)) rather than manually constructing the database.

Primary use: called by ``run_cpp_lint.py`` before running clangd checks, and
by ``make generate_compile_commands`` when you need to refresh the database
manually (e.g. after adding new source files or changing compiler flags).

Usage::

    make generate_compile_commands
"""

import shutil
import subprocess
from pathlib import Path

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_CMAKE_BUILD_DIR: Path = _REPO_ROOT / ".cache" / "cmake_build"
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
    shutil.copy(_CMAKE_BUILD_DIR / "compile_commands.json", _COMPILE_COMMANDS)


def main() -> None:
    write_compile_commands()
    print(f"Wrote {_COMPILE_COMMANDS}")


if __name__ == "__main__":
    main()
