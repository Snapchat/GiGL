"""Build script for GiGL pybind11 C++ extensions.

Invoked by ``make build_cpp_extensions`` and automatically during ``make install_dev_deps``
via ``post_install.py``.  Not a general-purpose setup.py — only builds C++ extensions.

Usage::

    python build_cpp_extensions.py build_ext --inplace
"""

from setuptools import setup
from setuptools.extension import Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

from scripts.cpp_build_constants import COMPILE_ARGS, CSRC_DIR


def find_cpp_extensions() -> list[Extension]:
    """Auto-discover pybind11 extension modules under ``gigl/csrc/``.

    Following PyTorch's csrc convention, only files named ``python_*.cpp`` are
    compiled as Python extension modules.

    Returns an empty list if ``gigl/csrc/`` does not yet exist.
    """
    if not CSRC_DIR.exists():
        return []
    extensions = []
    for cpp_file in sorted(CSRC_DIR.rglob("python_*.cpp")):
        parts = list(cpp_file.with_suffix("").parts)
        parts[-1] = parts[-1].removeprefix("python_")
        module_name = ".".join(parts)
        impl_file = cpp_file.parent / (parts[-1] + ".cpp")
        sources = [str(cpp_file)]
        if impl_file.exists():
            sources.append(str(impl_file))
        extensions.append(
            CppExtension(
                name=module_name,
                sources=sources,
                extra_compile_args=COMPILE_ARGS,
            )
        )
    return extensions


setup(
    ext_modules=find_cpp_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
