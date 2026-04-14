"""Run C++ lint on source files using clangd.

Generates compile_commands.json, then runs clangd --check on each file in
parallel and prints a clean summary.

Usage::

    # Lint specific files (also regenerates compile_commands.json):
    uv run python scripts/run_cpp_lint.py <file1.cpp> [file2.cpp] ...

    # Regenerate build/compile_commands.json only (for IDE / clangd setup):
    uv run python scripts/run_cpp_lint.py
"""

import json
import re
import subprocess
import sys
import sysconfig
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_CSRC_DIR: Path = _REPO_ROOT / "gigl" / "csrc"
_COMPILE_COMMANDS: Path = _REPO_ROOT / "build" / "compile_commands.json"
_COMPILE_ARGS: list[str] = [
    "-O3",
    "-std=c++17",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
]

# Matches real clang-tidy diagnostics emitted by clangd:
#   E[HH:MM:SS.mmm] [check-name] Line N: message
_DIAGNOSTIC_RE = re.compile(r"^E\[[\d:.]+\] (\[.+\] .+)$")


def _get_cxx_system_include_flags() -> list[str]:
    """Return -isystem flags for C++ standard library headers.

    clang++-15 on some machines is not configured with the correct GCC toolchain
    path and cannot find standard headers on its own. We derive them from the
    system g++ and pass them explicitly.
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
        return []


def _generate_compile_commands() -> None:
    """Write build/compile_commands.json for clangd."""
    # Suppress PyTorch's CUDA-not-found warning emitted at import time.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from torch.utils.cpp_extension import include_paths as torch_include_paths

    # -isystem (not -I) marks these as system headers so clangd's checks skip
    # AST nodes originating from them, avoiding deep analysis of PyTorch internals.
    include_flags: list[str] = [f"-isystem{p}" for p in torch_include_paths()]
    include_flags.append(f"-isystem{sysconfig.get_path('include')}")
    include_flags.extend(_get_cxx_system_include_flags())

    cpp_sources = sorted(_CSRC_DIR.rglob("*.cpp")) if _CSRC_DIR.exists() else []
    cxx_flags = " ".join(_COMPILE_ARGS)

    commands = [
        {
            "directory": str(_REPO_ROOT),
            "file": str(source),
            "command": f"clang++-15 {cxx_flags} {' '.join(include_flags)} -c {source}",
        }
        for source in cpp_sources
    ]

    _COMPILE_COMMANDS.parent.mkdir(exist_ok=True)
    _COMPILE_COMMANDS.write_text(json.dumps(commands, indent=2))


def _check_file(source: Path) -> tuple[Path, list[str]]:
    result = subprocess.run(
        [
            "clangd",
            f"--check={source}",
            f"--compile-commands-dir={_COMPILE_COMMANDS.parent}",
        ],
        capture_output=True,
        text=True,
    )
    diagnostics: list[str] = []
    for line in result.stderr.splitlines():
        m = _DIAGNOSTIC_RE.match(line)
        if m:
            diagnostics.append(m.group(1))
    return source, diagnostics


def main() -> None:
    sources = [Path(s) for s in sys.argv[1:]]

    _generate_compile_commands()

    if not sources:
        print(f"Wrote {_COMPILE_COMMANDS}")
        return

    failures: dict[Path, list[str]] = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_check_file, s): s for s in sources}
        for future in as_completed(futures):
            source, diagnostics = future.result()
            if diagnostics:
                failures[source] = diagnostics

    if not failures:
        print("\033[32mC++ lint passed.\033[0m")
    else:
        for source in sorted(failures):
            print(f"  FAIL  {source}")
            for d in failures[source]:
                print(f"        {d}")
        sys.exit(1)


if __name__ == "__main__":
    main()
