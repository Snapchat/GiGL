"""Asserts a freshly built ``gigl-core`` wheel is the one the release job meant to build.

Checks that there is exactly one wheel, that its filename carries the ABI tag of ``--python``, and that its
extension links ``libtorch_cuda.so`` exactly when ``--torch-variant`` is ``cu128``.

Example:
    $ uv run --no-sync python .github/scripts/check_gigl_core_wheel.py \
        --wheel "gigl-core/dist/*.whl" --python 3.12 --torch-variant cu128
"""

import argparse
import glob
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

_CUDA_NEEDED = re.compile(r"NEEDED.*\[libtorch_cuda\.so\]")


def find_wheel(pattern: str) -> Path:
    """Returns the only wheel matching ``pattern`` (a glob, or a directory holding wheels)."""
    if Path(pattern).is_dir():
        pattern = str(Path(pattern) / "*.whl")
    wheels = glob.glob(pattern)
    if len(wheels) != 1:
        raise AssertionError(
            f"expected exactly one wheel for {pattern}, found {wheels}"
        )
    return Path(wheels[0])


def check_abi_tag(wheel_name: str, python_minor: str) -> None:
    """``gigl_core-0.3.1-cp312-cp312-linux_x86_64.whl`` carries ``cp312-cp312`` for ``3.12``."""
    tag = "cp" + python_minor.replace(".", "")
    if f"-{tag}-{tag}-" not in wheel_name:
        raise AssertionError(f"{wheel_name} does not carry the ABI tag {tag}-{tag}")


def links_libtorch_cuda(readelf_dynamic: str) -> bool:
    """Whether the ``readelf -d`` output lists ``libtorch_cuda.so`` as a needed library."""
    return bool(_CUDA_NEEDED.search(readelf_dynamic))


def check_torch_variant(wheel: zipfile.ZipFile, torch_variant: str) -> None:
    """The extension must link libtorch_cuda.so for cu128 and must not for cpu."""
    with tempfile.TemporaryDirectory() as extract_dir:
        sos = [
            n
            for n in wheel.namelist()
            if n.startswith("gigl_core/") and n.endswith(".so")
        ]
        if not sos:
            raise AssertionError("the wheel contains no gigl_core/*.so")
        for so in sos:
            wheel.extract(so, extract_dir)
            # check=True: a failing readelf must fail the check, not read as "not linked".
            dynamic = subprocess.run(
                ["readelf", "-d", str(Path(extract_dir) / so)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            linked = "cu128" if links_libtorch_cuda(dynamic) else "cpu"
            if linked != torch_variant:
                raise AssertionError(
                    f"{so} links torch {linked}, expected {torch_variant}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--python", required=True, help="Python minor, e.g. 3.12")
    parser.add_argument("--torch-variant", required=True, choices=["cpu", "cu128"])
    args = parser.parse_args()

    wheel_path = find_wheel(args.wheel)
    check_abi_tag(wheel_path.name, args.python)
    with zipfile.ZipFile(wheel_path) as wheel:
        check_torch_variant(wheel, args.torch_variant)
    print(f"{wheel_path.name}: ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        sys.exit(f"error: {error}")
