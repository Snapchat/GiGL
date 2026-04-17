#!/usr/bin/env python3
"""
Once GiGL is installed w/ `pip install gigl`, this script can be executed by running:
`gigl-post-install`

This script installs GLT, which cannot be distributed as a standard wheel.
C++ extensions are built automatically by scikit-build-core during `pip install`
and do not require a separate step here.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Main entry point for the post-install script."""
    print("Running GIGL post-install script...")

    script_dir = Path(__file__).parent

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


if __name__ == "__main__":
    main()
