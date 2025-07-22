#!/usr/bin/env python3
"""
Post-install script for GIGL that runs install_glt.sh
"""

from typing import Optional

import os
import subprocess
import sys
from pathlib import Path


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

    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Path to the install_glt.sh script
    install_glt_script = script_dir / "install_glt.sh"

    if not install_glt_script.exists():
        print(f"Error: install_glt.sh not found at {install_glt_script}")
        sys.exit(1)

    cmd = f"bash {install_glt_script}"

    try:
        print(f"Executing {cmd}...")
        result = run_command_and_stream_stdout(cmd)
        print("Post-install script finished running, with return code: ", result)
        return result

    except subprocess.CalledProcessError as e:
        print(f"Error running install_glt.sh: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
