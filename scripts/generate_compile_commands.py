"""Generate .cache/compile_commands.json for clangd.

clangd requires a compilation database to resolve include paths and compiler
flags. This script derives those paths from the installed torch and pybind11
packages and writes ``.cache/compile_commands.json``.

Primary use: called by ``run_cpp_lint.py`` before running clangd checks, and
by ``make generate_compile_commands`` when you need to refresh the database
manually (e.g. after adding new source files or changing compiler flags).

Usage::

    make generate_compile_commands
"""

import json
import subprocess
import sys
import sysconfig
import warnings
from pathlib import Path

from gigl.csrc.cpp_compile_constants import COMPILE_ARGS

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_CSRC_DIR: Path = _REPO_ROOT / "gigl" / "csrc"
_COMPILE_COMMANDS: Path = _REPO_ROOT / ".cache" / "compile_commands.json"


def _get_cxx_system_include_flags() -> list[str]:
    """Return -isystem flags for the C++ standard library headers.

    clang++-15 on some machines is not configured with the correct GCC toolchain
    path, so it cannot find standard headers like <cstddef> on its own. We ask
    the system g++ compiler for its include search paths and pass them explicitly,
    which is the reliable cross-machine approach.
    """
    try:
        result = subprocess.run(
            ["c++", "-v", "-x", "c++", "-E", "/dev/null"],
            capture_output=True,
            text=True,
        )
        includes: list[str] = []
        in_section = False
        for line in result.stderr.splitlines():
            if "#include <...> search starts here:" in line:
                in_section = True
                continue
            if "End of search list." in line:
                break
            if in_section and line.strip():
                includes.append(f"-isystem{line.strip()}")
        return includes
    except FileNotFoundError:
        print(
            "Warning: 'c++' not found; C++ system headers will be missing",
            file=sys.stderr,
        )
        return []


def write_compile_commands() -> None:
    """Write .cache/compile_commands.json for clangd."""
    # Suppress PyTorch's CUDA-not-found warning emitted at import time.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from torch.utils.cpp_extension import include_paths as torch_include_paths

    # -isystem marks these as system headers so clangd skips deep analysis of
    # PyTorch internals.
    include_flags: list[str] = [f"-isystem{p}" for p in torch_include_paths()]
    include_flags.append(f"-isystem{sysconfig.get_path('include')}")
    include_flags.extend(_get_cxx_system_include_flags())

    cpp_sources = sorted(_CSRC_DIR.rglob("*.cpp")) if _CSRC_DIR.exists() else []
    if not cpp_sources:
        print(f"Warning: no .cpp files found under {_CSRC_DIR}", file=sys.stderr)

    cxx_flags = " ".join(COMPILE_ARGS)
    commands: list[dict[str, str]] = [
        {
            "directory": str(_REPO_ROOT),
            "file": str(source),
            "command": f"clang++ {cxx_flags} {' '.join(include_flags)} -c {source}",
        }
        for source in cpp_sources
    ]

    _COMPILE_COMMANDS.parent.mkdir(exist_ok=True)
    _COMPILE_COMMANDS.write_text(json.dumps(commands, indent=2))


def main() -> None:
    write_compile_commands()
    print(f"Wrote {_COMPILE_COMMANDS}")


if __name__ == "__main__":
    main()
