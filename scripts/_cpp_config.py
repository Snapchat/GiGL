"""Shared C++ build configuration used by build_cpp_extensions.py and generate_compile_commands.py."""

COMPILE_ARGS: list[str] = ["-O3", "-std=c++17", "-Wall", "-Wextra", "-Wno-unused-parameter"]
