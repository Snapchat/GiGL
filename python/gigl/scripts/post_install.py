#!/usr/bin/env python3
"""
Post-install script for GIGL that runs install_glt.sh
"""

import os
import subprocess
import sys
from pathlib import Path


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

    try:
        # Make the script executable
        os.chmod(install_glt_script, 0o755)

        # Run the install_glt.sh script
        print(f"Executing {install_glt_script}...")
        result = subprocess.run(
            [str(install_glt_script)],
            cwd=script_dir,
            check=True,
            capture_output=False
        )

        print("Post-install script completed successfully!")
        return result.returncode

    except subprocess.CalledProcessError as e:
        print(f"Error running install_glt.sh: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
