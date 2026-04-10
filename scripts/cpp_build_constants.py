"""Shared C++ build constants for build_cpp_extensions.py and generate_compile_commands.py.

This is the single source of truth for C++ compiler flags and source paths.
Both scripts import from here so clang-tidy always analyzes with the same flags
used during the actual build.
"""

from pathlib import Path

# REPO_ROOT is derived from this file's location — this file must live in scripts/.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CSRC_DIR: Path = REPO_ROOT / "gigl" / "csrc"

# Flags passed to every C++ compilation unit. Applies to both the extension
# build (build_cpp_extensions.py) and the compile_commands.json used by
# clang-tidy (generate_compile_commands.py).
COMPILE_ARGS: list[str] = ["-O3", "-std=c++17", "-Wall", "-Wextra"]
