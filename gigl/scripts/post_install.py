#!/usr/bin/env python3
"""
Once GiGL is installed w/ `pip install gigl`, this script can be executed by running:
`gigl-post-install`

This script is used to install the dependencies for GIGL:
- Installs GLT by running install_glt.sh.
- Builds pybind11 C++ extensions in-place so they are importable without a separate build step.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_command_and_stream_stdout(cmd: str) -> Optional[int]:
    """
    Executes a command and streams the stdout output.

    Args:
        cmd (str): The command to be executed.

    Returns:
        Optional[int]: The return code of the command, or None if the command failed to execute.
    """
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    while True:
        output = process.stdout.readline()  # type: ignore
        if output == b"" and process.poll() is not None:
            break
        if output:
            print(output.strip())
    return_code: Optional[int] = process.poll()
    return return_code


def main():
    """Main entry point for the post-install script."""
    print("Running GIGL post-install script...")

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    # Step 1: Install GLT
    install_glt_script = script_dir / "install_glt.sh"
    if not install_glt_script.exists():
        print(f"Error: install_glt.sh not found at {install_glt_script}")
        sys.exit(1)

    try:
        print(f"Executing bash {install_glt_script}...")
        result = run_command_and_stream_stdout(f"bash {install_glt_script}")
        print("GLT install finished with return code:", result)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    # Step 2: Build pybind11 C++ extensions in-place so they are importable
    # without requiring a separate `make build_cpp_extensions` call.
    # subprocess.run streams stdout/stderr to the terminal and raises
    # CalledProcessError on a non-zero exit code.
    try:
        print("Building C++ extensions...")
        subprocess.run(
            [
                sys.executable,
                "scripts/build_cpp_extensions.py",
                "build_ext",
                "--inplace",
            ],
            cwd=repo_root,
            check=True,
        )
        print("C++ extension build finished.")
    except subprocess.CalledProcessError as e:
        print(f"Error building C++ extensions: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error building C++ extensions: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
