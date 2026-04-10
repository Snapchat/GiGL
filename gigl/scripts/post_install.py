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
        print(f"Installing GLT via {install_glt_script}...")
        subprocess.run(["bash", str(install_glt_script)], check=True)
        print("GLT install finished.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing GLT: {e}")
        sys.exit(1)

    # Step 2: Build pybind11 C++ extensions in-place so they are importable
    # without requiring a separate `make build_cpp_extensions` call.
    # subprocess.run streams stdout/stderr to the terminal and raises
    # CalledProcessError on a non-zero exit code.
    build_cpp_script = repo_root / "scripts" / "build_cpp_extensions.py"
    if not build_cpp_script.exists():
        print(f"Error: build_cpp_extensions.py not found at {build_cpp_script}")
        sys.exit(1)

    try:
        print("Building C++ extensions...")
        subprocess.run(
            [sys.executable, str(build_cpp_script), "build_ext", "--inplace"],
            cwd=repo_root,
            check=True,
        )
        print("C++ extension build finished.")
    except subprocess.CalledProcessError as e:
        print(f"Error building C++ extensions: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
