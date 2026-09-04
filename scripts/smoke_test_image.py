"""Asserts that a Python environment is the one a GiGL image is supposed to ship.

Runs inside the base images, which contain no GiGL source, so the only imports at module
scope are the standard library. The packages to assert are named by the caller through
``--imports`` and imported dynamically; importing ``gigl`` here would make the script
unusable for the images it exists to check.

The caller declares the import set because the images do not ship the same packages:

  - Base images install ``gigl-core`` from a build manifest carrying no C++ sources, so
    ``gigl_core`` is a metadata-only distribution there and does not import. Only a
    ``src`` image can assert it.
  - The Dataflow base skips the GraphLearn-for-PyTorch post-install step, so
    ``graphlearn_torch`` is absent from that image by design.

Every check raises on mismatch, so a zero exit status is the only signal of success.

Example:
    Inside a CUDA base image::

        $ python scripts/smoke_test_image.py --python 3.11 \
            --imports torch,graphlearn_torch \
            --venv-prefix /gigl_deps/.venv --min-glibc 2.39

    Inside a Dataflow base image::

        $ python scripts/smoke_test_image.py --python 3.11 \
            --imports torch,apache_beam \
            --venv-prefix /gigl_deps/.venv --beam 2.56.0 --boot-env --min-glibc 2.39

    Inside a src image, or on a developer checkout where the venv path varies by
    checkout and so cannot be asserted::

        $ python scripts/smoke_test_image.py --python 3.11 \
            --imports torch,graphlearn_torch,gigl_core
"""

import argparse
import importlib
import os
import platform
import sys
import sysconfig
from pathlib import Path

SUPPORTED_PYTHON_VERSIONS: list[str] = ["3.11", "3.12", "3.13"]

# The packages GiGL images install outside of GiGL's own source tree. Restricting
# --imports to this set turns a typo into a usage error instead of a failing import that
# reads like a broken image.
CHECKABLE_IMPORTS: list[str] = ["torch", "graphlearn_torch", "gigl_core", "apache_beam"]

# Exported by the graphlearn_torch pybind11 extension only when it is compiled with
# WITH_CUDA=ON, so its absence identifies a CPU-only GLT wheel in a CUDA image.
GLT_CUDA_ONLY_SYMBOL = "cuda_stitch_sample_results"


def assert_equal(what: str, expected: object, actual: object) -> None:
    """Raises unless ``actual`` equals ``expected``.

    Args:
        what (str): Name of the checked property, used in the failure message.
        expected (object): The value the image is supposed to have.
        actual (object): The value the image actually has.

    Raises:
        AssertionError: When the values differ.
    """
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected!r}, got {actual!r}")
    print(f"OK  {what} == {actual!r}")


def parse_version(version: str) -> tuple[int, ...]:
    """Splits a dotted numeric version into ints so it compares by value.

    Comparing versions as strings ranks ``"2.9"`` above ``"2.39"``, which would let a
    glibc floor pass on an older libc than requested.

    Args:
        version (str): Dotted version, e.g. ``"2.39"``.

    Returns:
        tuple[int, ...]: The components, e.g. ``(2, 39)``.

    Raises:
        ValueError: When any component is not an integer.
    """
    return tuple(int(part) for part in version.split("."))


def check_python(python_version: str) -> None:
    """Asserts the running interpreter is the requested minor version, inside a venv.

    SOABI is the ABI tag of the running interpreter, so it is the tag every compiled
    extension in the image has to carry to be importable. Requiring the delimiter after
    the minor version rejects the free-threaded build, whose tag is ``cpython-313t-`` and
    which no GiGL image is built against.

    Args:
        python_version (str): Requested version as ``major.minor``, e.g. ``"3.13"``.

    Raises:
        AssertionError: When the version, SOABI, or venv state does not match.
    """
    major, minor = (int(part) for part in python_version.split("."))
    assert_equal("sys.version_info[:2]", (major, minor), sys.version_info[:2])

    soabi = sysconfig.get_config_var("SOABI")
    expected_soabi_prefix = f"cpython-{major}{minor}-"
    if not isinstance(soabi, str) or not soabi.startswith(expected_soabi_prefix):
        raise AssertionError(
            f"SOABI: expected a value starting with {expected_soabi_prefix!r}, got {soabi!r}"
        )
    print(f"OK  SOABI == {soabi!r}")

    if sys.prefix == sys.base_prefix:
        raise AssertionError(
            "virtual environment: expected sys.prefix != sys.base_prefix, got both "
            f"equal to {sys.prefix!r}"
        )
    print(f"OK  virtual environment active at {sys.prefix!r}")


def check_venv_prefix(venv_prefix: str) -> None:
    """Asserts the running interpreter comes from inside ``venv_prefix``.

    Containment is by path component, not string prefix: a string prefix accepts a
    sibling directory whose name merely starts the same way, so ``/gigl_deps/.venv``
    would be satisfied by an interpreter in ``/gigl_deps/.venv-broken``.

    Paths are normalized but not resolved. A venv's ``bin/python`` is a symlink to the
    interpreter uv installed elsewhere, so resolving it walks out of the venv and no
    prefix inside the venv could ever match.

    Args:
        venv_prefix (str): Directory the interpreter is expected to live under.

    Raises:
        AssertionError: When ``sys.executable`` lies outside ``venv_prefix``.
    """
    executable = Path(os.path.abspath(sys.executable))
    expected_prefix = Path(os.path.abspath(venv_prefix))
    if not executable.is_relative_to(expected_prefix):
        raise AssertionError(
            f"sys.executable: expected a path under {str(expected_prefix)!r}, got {str(executable)!r}"
        )
    print(f"OK  sys.executable == {sys.executable!r}")


def check_imports(module_names: list[str]) -> None:
    """Asserts each named package is importable.

    Args:
        module_names (list[str]): Import names the image is required to provide.

    Raises:
        ImportError: When any of them is missing or fails to load.
    """
    for module_name in module_names:
        importlib.import_module(module_name)
        print(f"OK  import {module_name}")


def check_cuda() -> None:
    """Asserts CUDA is usable by torch and that GLT is built against it.

    Beyond the presence of the CUDA-only GLT symbol, this builds a ``CUDA``-mode graph
    and initializes it, which copies the topology onto the device and therefore fails if
    the runtime CUDA stack is broken rather than merely compiled in.

    Raises:
        AssertionError: When CUDA is unavailable or GLT lacks its CUDA symbol.
    """
    import torch

    assert_equal("torch.cuda.is_available()", True, torch.cuda.is_available())

    import graphlearn_torch as glt

    if not hasattr(glt.py_graphlearn_torch, GLT_CUDA_ONLY_SYMBOL):
        raise AssertionError(
            f"graphlearn_torch.py_graphlearn_torch.{GLT_CUDA_ONLY_SYMBOL}: expected the "
            "symbol to exist, got a CPU-only graphlearn_torch build"
        )
    print(f"OK  graphlearn_torch exports {GLT_CUDA_ONLY_SYMBOL}")

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    graph = glt.data.Graph(glt.data.Topology(edge_index=edge_index), mode="CUDA")
    graph.lazy_init()
    assert_equal("CUDA-mode graph edge count", 3, graph.edge_count)


def check_beam(beam_version: str) -> None:
    """Asserts the installed Apache Beam matches the version the worker harness expects.

    Args:
        beam_version (str): Exact expected value of ``apache_beam.__version__``.

    Raises:
        AssertionError: When the installed version differs.
    """
    import apache_beam

    assert_equal("apache_beam.__version__", beam_version, apache_beam.__version__)


def check_boot_env() -> None:
    """Asserts the Beam boot harness is told to use the image's own environment.

    Raises:
        AssertionError: When ``RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT`` is not ``"1"``,
            which makes boot build a per-worker venv that cannot see the image venv's
            site-packages.
    """
    assert_equal(
        "RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT",
        "1",
        os.environ.get("RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT"),
    )


def check_min_glibc(min_glibc: str) -> None:
    """Asserts the running glibc is at least ``min_glibc``.

    The glibc floor is an installability constraint, not a preference: current
    ``tensorflow-data-validation`` and ``tfx-bsl`` releases publish only
    ``manylinux_2_39_x86_64`` wheels, so a base below that floor has no candidate to
    resolve at all. Asserting it here names the OS as the cause; otherwise a base that
    drops to an older Ubuntu surfaces as nothing but an unresolvable
    ``tensorflow-data-validation`` requirement during ``uv sync``.

    Args:
        min_glibc (str): Lowest acceptable glibc version, e.g. ``"2.39"``.

    Raises:
        AssertionError: When glibc is older than requested, or when the running libc does
            not report a version at all — a non-glibc libc has no bearing on this floor,
            so it cannot satisfy it.
    """
    libc, actual = platform.libc_ver()
    if not actual:
        raise AssertionError(
            f"glibc: expected at least {min_glibc}, got no version from "
            f"platform.libc_ver(), which reported libc {libc!r}"
        )
    if parse_version(actual) < parse_version(min_glibc):
        raise AssertionError(f"glibc: expected at least {min_glibc}, got {actual}")
    print(f"OK  glibc == {actual} (>= {min_glibc})")


def parse_imports(raw: str) -> list[str]:
    """Parses the ``--imports`` value into the list of packages to require.

    Args:
        raw (str): Comma-separated import names.

    Returns:
        list[str]: The requested import names, in the order given.

    Raises:
        argparse.ArgumentTypeError: When the value is empty or names a package outside
            ``CHECKABLE_IMPORTS``.
    """
    module_names = [part.strip() for part in raw.split(",") if part.strip()]
    if not module_names:
        raise argparse.ArgumentTypeError("expected at least one import name")
    unknown = [name for name in module_names if name not in CHECKABLE_IMPORTS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown import name(s) {unknown}; choose from {CHECKABLE_IMPORTS}"
        )
    return module_names


def main() -> None:
    """Parses arguments and runs the requested checks.

    Raises:
        AssertionError: When any check fails.
    """
    parser = argparse.ArgumentParser(
        description="Assert that the current environment matches a GiGL image contract."
    )
    parser.add_argument(
        "--python",
        required=True,
        choices=SUPPORTED_PYTHON_VERSIONS,
        help="Python minor version the environment must be running.",
    )
    parser.add_argument(
        "--imports",
        required=True,
        type=parse_imports,
        help="Comma-separated packages the environment must be able to import, chosen "
        f"from {','.join(CHECKABLE_IMPORTS)}. Required: image package sets differ, so "
        "the caller states which one it expects rather than the script guessing.",
    )
    parser.add_argument(
        "--venv-prefix",
        default=None,
        help="Directory the interpreter must live under, e.g. /gigl_deps/.venv. Omit "
        "outside images, where the venv path varies by checkout.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Also require a working CUDA runtime and a CUDA-enabled graphlearn_torch.",
    )
    parser.add_argument(
        "--beam",
        default=None,
        help="Exact apache_beam version the environment must have installed.",
    )
    parser.add_argument(
        "--boot-env",
        action="store_true",
        help="Also require RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT=1.",
    )
    parser.add_argument(
        "--min-glibc",
        default=None,
        help="Lowest glibc version the image may ship, e.g. 2.39.",
    )
    args = parser.parse_args()

    check_python(python_version=args.python)
    if args.venv_prefix is not None:
        check_venv_prefix(venv_prefix=args.venv_prefix)
    if args.min_glibc is not None:
        check_min_glibc(min_glibc=args.min_glibc)
    check_imports(module_names=args.imports)
    if args.cuda:
        check_cuda()
    if args.beam is not None:
        check_beam(beam_version=args.beam)
    if args.boot_env:
        check_boot_env()

    print("All checks passed.")


if __name__ == "__main__":
    main()
