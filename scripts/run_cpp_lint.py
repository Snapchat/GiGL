"""Run C++ lint on source files using clangd.

Generates compile_commands.json, then runs clangd --check on each file in
parallel and prints a clean summary.

Usage::

    uv run python scripts/run_cpp_lint.py <file1.cpp> [file2.cpp] ...
"""

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.generate_compile_commands import _COMPILE_COMMANDS, write_compile_commands

# Matches real clang-tidy diagnostics emitted by clangd:
#   E[HH:MM:SS.mmm] [check-name] Line N: message
_DIAGNOSTIC_RE = re.compile(r"^E\[[\d:.]+\] (\[.+\] .+)$")


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
    if not sources:
        sys.exit(0)

    write_compile_commands()

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
