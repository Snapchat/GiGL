"""Run C++ lint on source files using clangd.

Runs clangd --check on each file in parallel and prints a clean summary.
Expects compile_commands.json to already exist at
gigl-core/.cache/cmake_build/compile_commands.json; call
``make build_cpp_extensions`` first if it is absent or stale
(``make check_lint_cpp`` does this automatically via a Makefile prerequisite).

Usage::

    uv run python scripts/run_cpp_lint.py <file1.cpp> [file2.cpp] ...
"""

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
COMPILE_COMMANDS = (
    _REPO_ROOT / "gigl-core" / ".cache" / "cmake_build" / "compile_commands.json"
)

# Matches real clang-tidy diagnostics emitted by clangd:
#   E[HH:MM:SS.mmm] [check-name] Line N: message
_DIAGNOSTIC_RE = re.compile(r"^E\[[\d:.]+\] (\[.+\] .+)$")


def _check_file(source: Path) -> list[str]:
    result = subprocess.run(
        [
            "clangd-15",
            f"--check={source}",
            f"--compile-commands-dir={COMPILE_COMMANDS.parent}",
            "--query-driver=/usr/bin/clang++-15,/usr/bin/g++",
        ],
        capture_output=True,
        text=True,
    )
    diagnostics: list[str] = []
    completed_normally = False
    for line in result.stderr.splitlines():
        if "All checks completed" in line:
            completed_normally = True
        m = _DIAGNOSTIC_RE.match(line)
        if m:
            diagnostics.append(m.group(1))
    # Only treat a non-zero exit as a crash if clangd didn't reach its normal
    # completion message. A non-zero exit with "All checks completed" means
    # clangd found only IDE-action probe failures (tweak: ... ==> FAIL), which
    # are not lint violations and should be ignored.
    if not completed_normally and result.returncode != 0:
        diagnostics = [
            f"clangd exited with code {result.returncode} (tool error or crash)"
        ]
    return diagnostics


def main() -> None:
    sources = [Path(s) for s in sys.argv[1:]]
    if not sources:
        sys.exit(0)

    failures: dict[Path, list[str]] = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_check_file, s): s for s in sources}
        for future in as_completed(futures):
            source = futures[future]
            try:
                diagnostics = future.result()
            except Exception as exc:
                diagnostics = [f"linter error: {exc}"]
            if diagnostics:
                failures[source] = diagnostics

    if not failures:
        print("\033[32mC++ lint passed.\033[0m")
    else:
        for source in sorted(failures):
            print(f"  FAIL  {source}")
            for d in failures[source]:
                print(f"        {d}")
        print(
            "\nRun \033[1mmake fix_lint_cpp\033[0m to auto-fix violations where possible."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
