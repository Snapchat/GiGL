"""Fails if the local GLT patches no longer apply to the pinned upstream GLT commit.

Applies each patch in ``gigl/scripts/patches/`` in sorted filename order, with the same tool and
flags ``install_glt.sh`` uses, to a fresh checkout of ``GLT_COMMIT`` (later patches may depend on
earlier ones, so each is really applied). Catches a bumped pin or an edited patch before an image
build or install does.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

from gigl.scripts.glt_wheel_cache import GLT_COMMIT

GLT_REPO_URL: Final = "https://github.com/alibaba/graphlearn-for-pytorch.git"
# We're in GiGL/tests/config_tests, so we need to go up two levels to find GiGL/gigl/scripts/patches
PATCHES_DIR: Final = (
    Path(__file__).parent.parent.parent / "gigl" / "scripts" / "patches"
)


def _git(args: list[str], cwd: Path) -> None:
    # Git's stderr goes to the terminal so its failure reason is visible when check=True raises.
    subprocess.run(["git", *args], cwd=cwd, check=True)


if __name__ == "__main__":
    patches = sorted(PATCHES_DIR.glob("*.patch"), key=lambda p: p.name)
    assert patches, f"No patches found in {PATCHES_DIR}"
    with tempfile.TemporaryDirectory() as checkout_dir:
        checkout = Path(checkout_dir)
        _git(["init", "-q"], checkout)
        _git(["remote", "add", "origin", GLT_REPO_URL], checkout)
        _git(["fetch", "-q", "--depth", "1", "origin", GLT_COMMIT], checkout)
        _git(["checkout", "-q", "FETCH_HEAD"], checkout)
        for patch in patches:
            with open(patch, "rb") as patch_file:
                result = subprocess.run(
                    ["patch", "-p1", "--forward"],
                    cwd=checkout,
                    stdin=patch_file,
                    capture_output=True,
                    text=True,
                )
            if result.returncode != 0:
                print(result.stdout + result.stderr)
                sys.exit(
                    f"Patch {patch.name} does not apply to GLT commit {GLT_COMMIT}"
                )
    print(
        f"GLT commit {GLT_COMMIT}: all patches apply: {', '.join(p.name for p in patches)}"
    )
