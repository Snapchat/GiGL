from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def find_cpp_extensions() -> list[CppExtension]:
    """Auto-discover pybind11 extension modules under ``gigl/cpp_extensions/``.

    The module name is derived from the file path, so each extension is
    importable at the Python path corresponding to its location.
    """
    extensions = []
    for cpp_file in sorted(Path("gigl/cpp_extensions").rglob("*.cpp")):
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
