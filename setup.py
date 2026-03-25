from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def find_cpp_extensions() -> list[CppExtension]:
    """Auto-discover pybind11 extension modules.

    Any .cpp file anywhere under ``gigl/`` is compiled as a Python C++
    extension.  The module name is derived from the file path, so the
    extension is importable at the same location as its Python neighbours.

    Example::

        gigl/distributed/cpp_extensions/ppr_forward_push.cpp
            → importable as ``gigl.distributed.cpp_extensions.ppr_forward_push``

    To add a new extension, drop a .cpp file anywhere under ``gigl/`` —
    no changes to this file required.
    """
    extensions = []
    for cpp_file in sorted(Path("gigl").rglob("*.cpp")):
        # Convert path separators to dots and strip the .cpp suffix to get the
        # fully-qualified Python module name, e.g.:
        #   gigl/distributed/cpp_extensions/ppr_forward_push.cpp
        #   → "gigl.distributed.cpp_extensions.ppr_forward_push"
        module_name = ".".join(cpp_file.with_suffix("").parts)
        extensions.append(
            CppExtension(
                name=module_name,
                sources=[str(cpp_file)],
                extra_compile_args=["-O3", "-std=c++17", "-Wall", "-Wextra"],
            )
        )
    return extensions


setup(
    ext_modules=find_cpp_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
