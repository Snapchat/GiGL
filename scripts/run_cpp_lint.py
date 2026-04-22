"""Run C++ lint on source files using clang-tidy.

Generates compile_commands.json, then runs clang-tidy-15 on each file in
parallel and prints a clean summary.

Usage::

    uv run python scripts/run_cpp_lint.py <file1.cpp> [file2.cpp] ...
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.generate_compile_commands import COMPILE_COMMANDS


def _check_file(source: Path) -> tuple[Path, str, int]:
    result = subprocess.run(
        [
            "clang-tidy-15",
            f"-p={COMPILE_COMMANDS}",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    return source, (result.stdout + result.stderr).strip(), result.returncode


def main() -> None:
    sources = [Path(s) for s in sys.argv[1:]]
    if not sources:
        sys.exit(0)

    failures: dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 8)) as executor:
        futures = {executor.submit(_check_file, s): s for s in sources}
        for future in as_completed(futures):
            source, output, returncode = future.result()
            if returncode != 0:
                failures[source] = output

    if not failures:
        print("\033[32mC++ lint passed.\033[0m")
    else:
        for source in sorted(failures):
            print(f"  FAIL  {source}")
            if failures[source]:
                print(failures[source])
        print(
            "\nRun \033[1mmake fix_lint_cpp\033[0m to auto-fix violations where possible."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
